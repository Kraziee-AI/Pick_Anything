"""RSL-RL 蒸馏配置：状态版 teacher（已知坐标策略）→ 腕部相机 student。

第三个版本（IsaacLab 教程 anymal_d ``rsl_rl_distillation_cfg`` 同款机制）：

* ``obs_groups``：student 用 [wrist_rgb, proprioception]（实机可得的纯视觉
  输入，CNN 编码），teacher 用 [teacher]（staged 任务 policy 组的 111 维
  逐字复用）——rsl_rl ``Distillation`` 里 student 采样执行、teacher 对同一
  状态回归动作均值（在线 DAgger）；
* teacher 必须**逐字** ``STAGED_ACTOR_CFG``：checkpoint 的
  ``actor_state_dict`` 含 ``obs_normalizer.*`` / ``distribution.log_std_param``
  / ``mlp.0~6.*``，结构漂移会 strict load 失败（防错配特性）；
* student 原样复用相机版 CNN actor；其 std=0.2 仅作为 DAgger rollout 的
  采样噪声（不被蒸馏 loss 训练），推理用确定性均值；
* ``gradient_length=16`` 整除 ``num_learning_epochs×num_steps_per_env``
  =2×64=128，不留被累积却永不回传的尾步梯度；
* lr 5e-4：教程 anymal 蒸馏（MLP student）用 1e-3，CNN 从零起折半起步；
  若 loss 震荡可降 1e-4，若收敛过慢可升 1e-3（不动其余项）。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
)
from isaaclab_tasks.utils import PresetCfg

from pick_anything.tasks.pick_apple.agents.camera_ppo_cfg import CAMERA_ACTOR_CFG, CameraBoundedGaussianCfg
from pick_anything.tasks.pick_apple.agents.staged_ppo_cfg import STAGED_ACTOR_CFG

DISTILL_ALGO_CFG = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=2,
    learning_rate=5.0e-4,
    gradient_length=16,
    max_grad_norm=1.0,
    loss_type="mse",
)


@configclass
class CameraDistillBaseRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 42
    num_steps_per_env = 64
    max_iterations = 2000  # 行为克隆收敛快；每 250 存档，可按 Loss/behavior 早停
    save_interval = 250
    experiment_name = "pick_apple_staged_so101_camera_distill"
    run_name = ""
    obs_groups = {"student": ["wrist_rgb", "proprioception"], "teacher": ["teacher"]}
    clip_actions = 1.0
    student = CAMERA_ACTOR_CFG
    teacher = STAGED_ACTOR_CFG
    algorithm = DISTILL_ALGO_CFG


@configclass
class CameraDistillRunnerCfg(PresetCfg):
    default = CameraDistillBaseRunnerCfg()


# --------------------------------------------------------------------- #
# v1：对照 IsaacLabTutorial place_vial 蒸馏的三项 agent 侧修复（2026-09-30
# distill_v0 复盘：17249 轮平台 Loss≈0.17 / success≈0.51 / mean_stage≈2.7，
# 后 14000 轮零提升；参照同款 SO101+64×48 腕摄蒸馏 1024 envs × 1600 轮
# = 31 min 达 98.4%）
# --------------------------------------------------------------------- #


DISTILL_STUDENT_CFG = CAMERA_ACTOR_CFG.replace(
    distribution_cfg=CameraBoundedGaussianCfg(init_std=0.05, std_type="log")
)
"""v1 student：init_std 0.05（参照口径；v0 为 0.2）。

蒸馏 loss 只训均值、std 无梯度（rsl_rl 5.5.1 语义），init_std 即 DAgger
rollout 的固定采样噪声——0.2 在动作 scale 0.06/0.10 下每步抖 ±0.012/
±0.02 rad，把到访状态分布撒得太散；参照取 0.05。推理走确定性均值，
std 不影响部署。
"""

DISTILL_V1_ALGO_CFG = RslRlDistillationAlgorithmCfg(
    class_name="pick_anything.tasks.pick_apple.agents.distillation:BoundedTeacherDistillation",
    num_learning_epochs=4,
    learning_rate=5.0e-4,
    gradient_length=1,
    max_grad_norm=1.0,
    loss_type="mse",
)
"""v1 算法：teacher 标签 clamp 到 [-1,1]（agents/distillation.py，参照
BoundedTeacherDistillation——teacher 饱和均值远超动作裁剪，回归原始均值
浪费 student 容量）；4 epochs × 32 步、gradient_length=1 全部对齐参照
（v0 为 2 × 64 / 16）。"""


@configclass
class CameraDistillV1BaseRunnerCfg(RslRlDistillationRunnerCfg):
    """v1 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v1）。

    student 24 维本体（含 joint_target，教程四项口径）；teacher 逐字
    STAGED_ACTOR_CFG 不变（model_1750.pt strict load）。2000 轮：参照
    1600 轮收敛，行为克隆收敛快，每 250 存档可按 Loss/behavior 早停。
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 250
    experiment_name = "pick_apple_staged_so101_camera_distill"
    run_name = ""
    obs_groups = {"student": ["wrist_rgb", "proprioception"], "teacher": ["teacher"]}
    clip_actions = 1.0
    student = DISTILL_STUDENT_CFG
    teacher = STAGED_ACTOR_CFG
    algorithm = DISTILL_V1_ALGO_CFG


@configclass
class CameraDistillV1RunnerCfg(PresetCfg):
    default = CameraDistillV1BaseRunnerCfg()


# --------------------------------------------------------------------- #
# v2：配套细粒度教师（StagedFineTeacherEnvCfg，gym id
# Pick-Apple-SO101-Staged-TeacherFine-v0）。2026-10-01 distill_v1 平台期
# 定因=旧教师粗动作粒度（0.06/0.10）像素下不可模仿 + 窗口带标签 OOD。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV2BaseRunnerCfg(CameraDistillV1BaseRunnerCfg):
    """v2 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v2）。

    算法/student/teacher 全部继承 v1（已与 IsaacLabTutorial 逐字对齐）。
    唯一约束变化：``--checkpoint`` 必须指向**细粒度教师**
    （Pick-Apple-SO101-Staged-TeacherFine-v0 训练的 model）——旧
    model_1750 的动作尺度（0.06/0.10）与 v2 环境（0.033/0.02）不兼容，
    标签会把 student 教回粗粒度行为。教师 MLP 结构不变（同一
    STAGED_ACTOR_CFG，观测 111 维含 history=3），strict load 不受影响。
    """


@configclass
class CameraDistillV2RunnerCfg(PresetCfg):
    default = CameraDistillV2BaseRunnerCfg()


@configclass
class CameraDistillV3BaseRunnerCfg(CameraDistillV2BaseRunnerCfg):
    """v3 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v3）。

    与 v2 完全一致，仅环境侧腕摄 64×48→128×96（见 distill_env_cfg 的
    CameraDistillV3PickAppleEnvCfg docstring：教程对照定因=信息几何，
    分辨率 ×2 是「物体更小更远更散」的按比例补偿）。CNN/MLP 维度由
    rsl_rl 按输入尺寸自适应，配置无需改；teacher 仍是状态 MLP（与
    分辨率无关），--checkpoint 沿用 teacher_fine_v4/model_4250。
    """


@configclass
class CameraDistillV3RunnerCfg(PresetCfg):
    default = CameraDistillV3BaseRunnerCfg()


@configclass
class CameraDistillV4BaseRunnerCfg(CameraDistillV3BaseRunnerCfg):
    """v4 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v4）。

    与 v3 完全一致；环境侧加 P7 扫视 HOME 姿态 + 盒子可见窗口（出生画面
    橘子+盒子同框，见 distill_env_cfg 的 CameraDistillV4PickAppleEnvCfg
    docstring）。--checkpoint 必须指向新姿态重训的教师（teacher_fine_v5），
    旧 model_4250 与新 HOME 分布不匹配。
    """


@configclass
class CameraDistillV4RunnerCfg(PresetCfg):
    default = CameraDistillV4BaseRunnerCfg()


@configclass
class CameraDistillV5BaseRunnerCfg(CameraDistillV4BaseRunnerCfg):
    """v5 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v5）。

    算法/student/teacher 全部继承 v4（已与 IsaacLabTutorial 逐字对齐），
    唯一变化在环境侧 episode 20s（见 distill_env_cfg 的
    CameraDistillV5PickAppleEnvCfg）。--checkpoint 沿用 teacher_fine_v4/
    model_4250（O 姿态实测 96.1%，无需重训教师）。
    """


@configclass
class CameraDistillV5RunnerCfg(PresetCfg):
    default = CameraDistillV5BaseRunnerCfg()


@configclass
class CameraDistillV6BaseRunnerCfg(CameraDistillV5BaseRunnerCfg):
    """v6 蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v6）。

    算法/student/teacher 全部继承 v5；唯一变化在环境侧相机（合爪盲区修复：
    俯角 -12° + FOV ×1.5，见 camera_env_cfg.CameraClosureHdSceneCfg）。
    --checkpoint 沿用 teacher_fine_v4/model_4250（教师读状态，与相机无关）。
    """


@configclass
class CameraDistillV6RunnerCfg(PresetCfg):
    default = CameraDistillV6BaseRunnerCfg()


# --------------------------------------------------------------------- #
# v7：混合路线——cam_v9 热启动 student + teacher 标签 DAgger 精修（gym id
# Pick-Apple-SO101-Staged-Camera-Distill-v7）。2026-10-04 蒸馏线定案：从零
# BC 不可学（v0-v6 七轮单变量全零增量：GRASP-born 钉 0.20、HOME≡0、闭包
# 相机也无效——瓶颈是 DAgger 复合误差本身，不是可见性/episode/分辨率/配方
# ——与 IsaacLabTutorial 配方逐字一致仍失败即是铁证）；同 CNN 同观测的相机
# PPO 直训 cam_v9 已达确定性 91%。v7 只换掉「从零」这一前提：student 从
# cam_v9 model_11999 热启动（环境变量 PICK_DISTILL_STUDENT_CKPT 传入，
# agents/distillation.py 的 WarmStartBoundedTeacherDistillation 加载），
# --checkpoint 仍指 teacher model_4250（96.1%）；环境侧 episode 20s→90s
# （cam_v9 原生口径，见 distill_env_cfg 的 CameraDistillV7PickAppleEnvCfg）。
# --------------------------------------------------------------------- #


DISTILL_V7_ALGO_CFG = DISTILL_V1_ALGO_CFG.replace(
    class_name="pick_anything.tasks.pick_apple.agents.distillation:WarmStartBoundedTeacherDistillation",
    learning_rate=5.0e-5,
)
"""v7 算法：v1 的 clamp 蒸馏四件套 + student 热启动钩子；lr 5e-4→5e-5。

首跑 distill_v7_hybrid（lr 5e-4）iter118 定因：热启动优势（iter16
GRASP-born 0.73）在 ~100 迭代内被 BC 回归吸引子整个覆盖——中段出生成功
率收敛到从零 v2 同款签名（HOME 0/PREGRASP 0.03/GRASP 0.46/LIFT 0.54），
loss 同落 0.03 误差地板。lr 5e-4 × 128 Adam 步/迭代（gradient_length=1
× 4 epochs × 32 步）对已收敛策略是「重写」不是「精修」。5e-5 = 慢一个
数量级的打磨，配 save_interval 50 密集存档、按确定性评估收割最佳档。"""


@configclass
class CameraDistillV7BaseRunnerCfg(CameraDistillV6BaseRunnerCfg):
    """v7 混合蒸馏配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v7）。

    算法（4 epochs × 32 步 / gradient_length=1 / mse）继承 v6 不动；lr
    5e-5 见 DISTILL_V7_ALGO_CFG docstring。student 的 init_std 0.05 仅在
    无热启动时生效：正常启动会被 PICK_DISTILL_STUDENT_CKPT 载入的
    cam_v9 log_std（0.096，PPO 训练值）覆盖，只作 DAgger rollout 噪声，
    不影响 loss 与部署（推理=确定性均值）。max_iterations 4000 /
    save_interval 50：lr 5e-5 下 model_0≈纯热启动（128 步 × 5e-5 的移动
    可忽略），model_50/100/150/…是「热启动+渐进教师打磨」的密集序列，
    用确定性评估在外部挑最佳档（cam_v11 教训：精修期必须周期做确定性
    验收，随机口径优势不必然转化为确定性收益；判据见
    logs/night_watch_2026-10-04.md）。
    """

    max_iterations = 4000
    save_interval = 50
    algorithm = DISTILL_V7_ALGO_CFG


@configclass
class CameraDistillV7RunnerCfg(PresetCfg):
    default = CameraDistillV7BaseRunnerCfg()


# --------------------------------------------------------------------- #
# v9：纯蒸馏·官方式从零 + 256×192（gym id Pick-Apple-SO101-Staged-Camera-
# Distill-v9，见 distill_env_cfg 的 CameraDistillV9PickAppleEnvCfg docstring
# 的差异终表论证）。从零初始化 → 用 v1 的 BoundedTeacherDistillation
# （无 WarmStart 环境变量要求），超参全教程口径；--checkpoint 必须指平滑
# 教师 teacher_smooth_v1/model_2250（确定性 100%）。256×192 渲染 ×4 像素
# → rollout 存储 (32,512,3,192,256)×4B ≈ 9.7GB，num_envs 上限 512。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV9BaseRunnerCfg(RslRlDistillationRunnerCfg):
    """v9 纯蒸馏 runner：v1 全套（从零 BC、clamp 标签、教程超参）+ 4000 轮。"""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 250
    experiment_name = "pick_apple_staged_so101_camera_distill"
    run_name = ""
    obs_groups = {"student": ["wrist_rgb", "proprioception"], "teacher": ["teacher"]}
    clip_actions = 1.0
    student = DISTILL_STUDENT_CFG
    teacher = STAGED_ACTOR_CFG
    algorithm = DISTILL_V1_ALGO_CFG


@configclass
class CameraDistillV9RunnerCfg(PresetCfg):
    default = CameraDistillV9BaseRunnerCfg()


@configclass
class CameraDistillUnclippedBaseRunnerCfg(CameraDistillV9BaseRunnerCfg):
    """Distill raw means from teachers trained without action clipping.

    Both the BC labels and the student's executed actions must retain the
    teacher's action range. Tutorial place_vial clips its PPO teacher already;
    our staged teachers have clip_actions=None and action-term clip=None.
    Keep the v9 recipe otherwise unchanged for a controlled comparison.
    """

    clip_actions = None
    algorithm = DISTILL_V1_ALGO_CFG.replace(class_name="Distillation")


@configclass
class CameraDistillUnclippedRunnerCfg(PresetCfg):
    default = CameraDistillUnclippedBaseRunnerCfg()
