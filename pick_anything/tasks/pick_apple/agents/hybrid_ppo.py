"""混合 PPO：actor 从蒸馏 student 热启动，critic 从相机 PPO checkpoint 热启动。

2026-10-07 蒸馏十三版终审后的路线 A（用户批准）：纯 BC 的确定性均值天花板
（最佳 14.06%，随训练递减）只有奖励锚能补——拿 v9 蒸馏学生
（distill_v9_hd256/model_500，随机口径 GRASP-born 0.982/总 success 0.812）
做 PPO 初始化继续训练。cam_v12 已实证同配方（低 LR 3e-5 + 固定课程 +
确定性守护收割）在纯 PPO 血统上拿到 92.19%。

机制（rsl_rl 5.5.1 PPO.load 的 load_cfg 分支）：
* actor ← 环境变量 ``PICK_PPO_ACTOR_FROM_DISTILL`` 指向的蒸馏 checkpoint 之
  ``student_state_dict``（与 CNN actor 键集完全一致——obs_set 标签不进键名，
  组名同为 wrist_rgb/proprioception，位级验证过）；
* critic ← ``--checkpoint``（相机 PPO 档）的 ``critic_state_dict``——critic
  是 49 维状态输入、与图像分辨率无关，直接复用 cam_v9 的 value 函数热身；
* optimizer/iteration 跳过（fresh Adam，LR 由 runner cfg 控制；迭代从 0 计）。
无环境变量时行为=原生 PPO，不影响任何现有用法。
"""

from __future__ import annotations

import os

import torch
from rsl_rl.algorithms import PPO

ACTOR_FROM_DISTILL_ENV_VAR = "PICK_PPO_ACTOR_FROM_DISTILL"
"""混合 PPO（Camera-v9 任务）：actor 热启动用的蒸馏 checkpoint 路径。"""


class HybridPPO(PPO):
    """actor←蒸馏 student / critic←PPO checkpoint / 优化器全新的混合 PPO。"""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        actor_path = os.environ.get(ACTOR_FROM_DISTILL_ENV_VAR, "")
        if not actor_path:
            return super().load(loaded_dict, load_cfg, strict)

        payload = torch.load(actor_path, map_location=self.device, weights_only=False)
        if "student_state_dict" in payload:
            actor_sd = payload["student_state_dict"]
        elif "actor_state_dict" in payload:
            actor_sd = payload["actor_state_dict"]
        else:
            raise RuntimeError(
                f"{ACTOR_FROM_DISTILL_ENV_VAR} 指向的 checkpoint 既无 student_state_dict"
                " 也无 actor_state_dict——需要蒸馏或相机 PPO checkpoint。"
            )
        self._raw_actor.load_state_dict(actor_sd, strict=strict)
        # critic 仍取 --checkpoint（相机 PPO 档）；optimizer/iteration 全新
        hybrid_cfg = {"actor": False, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
        load_iteration = super().load(loaded_dict, hybrid_cfg, strict)
        state = self._raw_actor.state_dict()
        conv_key = next(key for key in state if key.startswith("cnns.") and key.endswith("weight"))
        print(
            f"[HYBRID-PPO] actor 热启动自 {actor_path}（iter={payload.get('iter', -1)}，"
            f"{len(actor_sd)} 张量，{conv_key} 权重和={float(state[conv_key].sum()):.6f}）；"
            f"critic 自 --checkpoint 热启动；optimizer/iteration 全新（LR 由 runner cfg 控制）"
        )
        return load_iteration
