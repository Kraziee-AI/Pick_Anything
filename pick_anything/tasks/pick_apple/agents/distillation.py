"""蒸馏算法定制：teacher 标签 clamp 到环境实际执行的动作范围。

照抄 IsaacLabTutorial place_vial 的 ``BoundedTeacherDistillation``
（agents/distillation.py）：PPO teacher 的高斯均值无界，饱和时远超环境
[-1, 1] 动作裁剪——回归原始均值会把 student 容量浪费在环境根本不执行的
幅值上，因此标签 clamp 到实际执行值。其余语义同 rsl_rl 5.5.1 标准
``Distillation``：student 采样执行、teacher 对每个到访状态打确定性均值
标签、student 对标签做行为克隆。
"""

from __future__ import annotations

import os

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

STUDENT_CKPT_ENV_VAR = "PICK_DISTILL_STUDENT_CKPT"
"""混合蒸馏（v7）：student 热启动 checkpoint 路径。

支持两种格式：相机 PPO checkpoint（取 ``actor_state_dict``，如 cam_v9
model_11999）或蒸馏 checkpoint（取 ``student_state_dict``，如某次蒸馏的
model_XXXX——用于从上一段蒸馏成果继续精修）。"""

STUDENT_STD_ENV_VAR = "PICK_DISTILL_STUDENT_STD"
"""混合蒸馏（v7）：可选的 rollout σ 覆写（如 0.05）。

蒸馏 loss 无 std 梯度，热启动后 σ 即固定采样噪声；默认继承 checkpoint
的 log_std。收窄 σ 的用途（2026-10-06 model_2000 判定门实证）：σ=0.096
的噪训分布与确定性部署轨迹的协变量差把确定性成功率压在 9.4%（训练侧
GRASP-born 已 0.95），且噪声让学生自身难完成 HOME 长链（随机口径仅
2.7%）——σ→0.05 同时收拢两处。"""


class BoundedTeacherDistillation(Distillation):
    """标签为 teacher *被执行*（clamp 后）动作的 RSL-RL 蒸馏。"""

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        return actions


class WarmStartBoundedTeacherDistillation(BoundedTeacherDistillation):
    """v7 混合蒸馏：student 用相机 PPO checkpoint 热启动，teacher 标签精修。

    2026-10-04 定案（v0-v6 七轮单变量逐一证伪后）：从零 BC/DAgger 在本任务
    的长链精度伺服上复合误差结构性不可学（GRASP-born 钉 0.20、HOME 恒 0，
    闭包相机修复也零增量），而同一 CNN/同一观测组的相机 PPO 直训
    （Camera-v7，cam_v9 model_11999）已达确定性 91%。混合路线 = student 从
    cam_v9 热启动（RL 已给粗伺服），teacher（model_4250，96.1%）对 student
    到访状态打确定性标签精修（DAgger 补教师级精度）。

    机制：训练入口 ``--checkpoint`` 仍指 teacher 的 PPO checkpoint（走 rsl_rl
    默认「只加载 teacher」分支）；本类在该分支上额外把环境变量
    ``PICK_DISTILL_STUDENT_CKPT`` 指向的相机 PPO checkpoint 之
    ``actor_state_dict``（含 obs_normalizer buffers 与 log_std）strict 载入
    student——两线 CNN/MLP 结构与观测组名完全同构，state_dict 键集一致
    （obs_set 标签 "actor"/"student" 不进键名）。载入蒸馏 checkpoint
    （eval/play/续训）时不触发热启动分支，无需设置环境变量。载入的
    log_std 只作 DAgger rollout 采样噪声（蒸馏 loss 无 std 梯度、推理走
    确定性均值），σ 取 cam_v9 的 PPO 训练值，到访状态散布比 v1-v6 钉死的
    0.05 宽——DAgger 纠正性标注的前提。
    """

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        is_rl_checkpoint = load_cfg is None and any("actor_state_dict" in key for key in loaded_dict)
        if is_rl_checkpoint:
            ckpt_path = os.environ.get(STUDENT_CKPT_ENV_VAR, "")
            if not ckpt_path:
                raise RuntimeError(
                    f"混合蒸馏要求环境变量 {STUDENT_CKPT_ENV_VAR} 指向相机 PPO checkpoint"
                    "（Camera-v7 训练的 model_*.pt，如 cam_v9_closure_512/model_11999.pt）"
                    "或蒸馏 checkpoint（student_state_dict）；"
                    "缺失则 student 从零初始化，即已被 v0-v6 证伪的从零 BC 路线。"
                )
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "actor_state_dict" in payload:
                actor_sd = payload["actor_state_dict"]
            elif "student_state_dict" in payload:
                actor_sd = payload["student_state_dict"]
            else:
                raise RuntimeError(
                    f"{STUDENT_CKPT_ENV_VAR} 指向的 checkpoint 既无 actor_state_dict 也无"
                    " student_state_dict——需要相机 PPO 或蒸馏 checkpoint。"
                )
            self._raw_student.load_state_dict(actor_sd, strict=strict)
            std_override = os.environ.get(STUDENT_STD_ENV_VAR, "")
            if std_override:
                with torch.no_grad():
                    self._raw_student.distribution.log_std_param.fill_(
                        float(torch.log(torch.tensor(float(std_override))))
                    )
            state = self._raw_student.state_dict()
            conv_key = next(key for key in state if key.startswith("cnns.") and key.endswith("weight"))
            std_now = float(state["distribution.log_std_param"].exp().mean())
            print(
                f"[DISTILL-V7] student 热启动自 {ckpt_path}（iter={payload.get('iter', -1)}，"
                f"{len(actor_sd)} 张量，{conv_key} 权重和={float(state[conv_key].sum()):.6f}，"
                f"rollout σ={std_now:.4f}"
                + (f"（覆写自 {std_override}）" if std_override else "")
                + "）"
            )
        return super().load(loaded_dict, load_cfg, strict)
