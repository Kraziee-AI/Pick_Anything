"""RSL-RL PPO 配置：腕部相机版八阶段任务（CNN actor + 特权 MLP critic）。

* obs_groups：actor 用 [wrist_rgb, proprioception]（图像组经 CNN 编码、
  本体组拼接后进 MLP 头——rsl_rl CNNModel 按观测维度自动路由），critic
  用 [critic]（staged 特权组原样复用）；
* CNN 结构取教程 64×48 腕图的紧凑编码器（16/32/32 通道、5/3/3 核、
  stride 2——下采样到 6×4 特征图），MLP 头 [256, 128]；
* std 钳制 (0.05, 0.3)、init 0.2——对齐教程视觉版口径（cam_v1 复盘
  2026-09-27：照搬状态任务的 (0.05, 0.15) 后，home_born_success 在
  147k episodes 上全程 0.000，std 钉上限、自适应 LR 自 it~500 起钉死
  1e-5 下限——高斯 KL≈Δμ²/2σ²，σ=0.15 对同样均值更新产生 4× KL，
  desired_kl=0.01 调度被反复打到底，学习饿死；表面成功率先升后降
  纯为出生课程权重经过。教程同 PPO 超参仅 std 上界 0.3 即确定性审计
  99%，其注释明言上界须保在「approach 仍可学」的水平）。staged 状态
  任务的 (0.05, 0.15) 不动——已收敛且断层修复实测有效；
* PPO 超参与 critic 网络原样复用 staged 配置；num_steps_per_env=64
  （教程视觉任务同值，图像任务需要更大的 rollout）；
* cam_v2 复盘（2026-09-28，11833/20000 停）：std 上界放到 0.3 后 LR 仍全程
  钉死 ~1.2e-5 下限（mean_std 恒 0.3、entropy 恒 1.29=6 维 σ0.3 理论值；
  对照组状态线 det_align_v1 同期 LR ~1e-4，同 reward 同 algo 收敛正常）——
  KL 超标的真因是 CNN 在图像噪声/光度 DR/部分可观测下的梯度噪声，不是 σ
  本身。cam_v3 冒烟实测 desired_kl 0.01→0.02 仍全程贴 1e-5 下限（每次更新
  KL>2×容差，自适应调度在这种噪声下救不回），故改固定 LR 1e-4（有效 LR
  抬 10 倍、行为确定，介于饿死地板与激进 3e-4 之间），entropy 0.005→0.001
  让 std 不再被顶死上界、可随精度需求下漂（下界 0.05 保探索）。std 上下界
  不动。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from isaaclab_tasks.utils import PresetCfg

from pick_anything.tasks.pick_apple.agents.staged_ppo_cfg import (
    STAGED_ALGO_CFG,
    STAGED_CRITIC_CFG,
    StagedBoundedGaussianCfg,
)

CAMERA_WRIST_CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[16, 32, 32],
    kernel_size=[5, 3, 3],
    stride=[2, 2, 2],
    activation="elu",
)
"""64×48 腕图的紧凑编码器（下采样到 6×4 特征图），教程同款。"""


@configclass
class CameraBoundedGaussianCfg(StagedBoundedGaussianCfg):
    """相机版 std 钳制：上界放宽到 0.3（教程视觉口径，依据见模块 docstring）。

    仅覆盖 std_range；其余（std_type="log" 钳制语义）继承 staged 实现。
    """

    std_range: tuple[float, float] = (0.05, 0.3)


CAMERA_ACTOR_CFG = RslRlCNNModelCfg(
    cnn_cfg=CAMERA_WRIST_CNN_CFG,
    hidden_dims=[256, 128],
    activation="elu",
    # 归一化只作用于 1D 本体组（图像已 [0,1]），CNNModel 内建行为
    obs_normalization=True,
    distribution_cfg=CameraBoundedGaussianCfg(init_std=0.2, std_type="log"),
)


@configclass
class CameraPPOBoundedGaussianCfg(CameraBoundedGaussianCfg):
    """PPO 专用 std 配置：上界回教程口径 0.3（cam_v4 与小动作 scale 配套）。

    演化：03:17 夜间裁决曾收到 0.2（大动作 scale 下 0.3×0.35 夹爪噪声
    过大）；cam_v4 教程对照定位出真因是动作粒度（臂 0.033/夹爪 0.02，
    见 camera_env_cfg.py 的 CameraVisibleHomeEnvCfg docstring）——小 scale
    下 0.3×0.02≈0.006/步，噪声无害且从头训练需要足量探索，回 (0.05,0.3)
    init 0.2（教程原文）。只挂 CAMERA_PPO_ACTOR_CFG；共享的
    CameraBoundedGaussianCfg 不动（蒸馏 student 继续用它）。
    """

    std_range: tuple[float, float] = (0.05, 0.3)


# 相机 PPO 专用 actor（仅分布上界与 CAMERA_ACTOR_CFG 不同；网络结构一致，
# cam_v3 checkpoint 的 log_std 参数可直接加载、clamp 到 0.2 起步）
CAMERA_PPO_ACTOR_CFG = CAMERA_ACTOR_CFG.replace(
    distribution_cfg=CameraPPOBoundedGaussianCfg(init_std=0.2, std_type="log")
)

# 相机专用算法参数（依据见模块 docstring cam_v2 复盘条目）：固定 LR 1e-4
# （自适应调度在 CNN 梯度噪声下必钉 1e-5 下限）+ entropy 0.001（std 不再钉
# 0.3 上界）。状态线的 STAGED_ALGO_CFG 不动。
CAMERA_ALGO_CFG = STAGED_ALGO_CFG.replace(
    entropy_coef=0.001,
    learning_rate=1.0e-4,
    schedule="fixed",
)


@configclass
class CameraPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 64
    max_iterations = 20000
    save_interval = 250
    experiment_name = "pick_apple_staged_so101_camera"
    run_name = ""
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = CAMERA_PPO_ACTOR_CFG
    critic = STAGED_CRITIC_CFG
    algorithm = CAMERA_ALGO_CFG


@configclass
class CameraStagedPPORunnerCfg(PresetCfg):
    default = CameraPPOBaseRunnerCfg()


@configclass
class CameraTutorialPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """v3 PPO（gym id: Pick-Apple-SO101-Staged-Camera-v3）：IsaacLabTutorial 配方原样。

    algorithm 直接用 STAGED_ALGO_CFG——它与教程 PPO_ALGORITHM_CFG 逐参数
    一致（adaptive 3e-4 / desired_kl 0.01 / entropy 0.005 / 5 epochs /
    8 mini-batches / gamma 0.995 / lam 0.95 / clip 0.2）。

    相机线 v0-v2 的固定 LR 1e-4 + entropy 0.001 是 cam_v2/v3 时代（粗动作
    + tanh 奖励表）下 adaptive 触底 1e-5 的权宜之计。教程日志（vision run）
    实测同款 CNN + 同款图像 DR/光度随机化下 adaptive 稳定运行于
    3e-5~1.7e-4（均值 ≈1e-4，即我们固定值附近）——触底复发的风险前提
    （粗动作语义）已在 cam_v4 起消除。std 钉 0.3 上界在教程同样全程存在
    且无害（见 camera_env_cfg.CameraTutorialEnvCfg docstring），actor
    沿用 CAMERA_PPO_ACTOR_CFG。

    必须搭配 CameraTutorialEnvCfg（奖励全表 ×1/30）：entropy 0.005 只有
    在教程量级回报（~7/局）下才是教程的相对压力；200+ 回报下会过强。
    max_iterations 8000（教程 5000 迭代即 93% 随机口径；本任务更难给
    1.6× 预算）；save_interval 100 便于按 success 峰值挑 checkpoint。
    """

    seed = 42
    num_steps_per_env = 64
    max_iterations = 8000
    save_interval = 100
    experiment_name = "pick_apple_staged_so101_camera"
    run_name = ""
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = CAMERA_PPO_ACTOR_CFG
    critic = STAGED_CRITIC_CFG
    algorithm = STAGED_ALGO_CFG


@configclass
class CameraTutorialPPORunnerCfg(PresetCfg):
    default = CameraTutorialPPOBaseRunnerCfg()


@configclass
class CameraTutorialFixedLRPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """v4 PPO（gym id: Pick-Apple-SO101-Staged-Camera-v4）：v3 缩放奖励 + cam_v5 固定 LR 优化器。

    两个已单独验证组件的合并。对照 IsaacLabTutorial（参照/现状/差异理由）：
    * 参照：教程 vision run adaptive 3e-4/desired_kl 0.01，实际运行区间
      3e-5~1.7e-4、均值 ≈1e-4。
    * 现状：cam_v6（Camera-v3，缩放奖励+adaptive）6474 轮复盘——
      Loss/value 329→0.06（缩放修复完全成功）、862 轮即 53%（起步 6×
      于 v5），但 LR 自 iter~647 起钉死 1e-5 下限，success 53%→41%
      回落、home_born≡0、episode 拖长到 1535 步——学习饿死。
    * 差异理由：adaptive 触底在本环境是 **CNN 相机 actor 特有**（第三次
      实测：cam_v1/cam_v2/cam_v6，均已排除回报量级因素）——同期
      teacher_fine_v4（同缩放奖励+adaptive，状态 MLP actor）LR 在
      1e-5~2.5e-4 震荡可自恢复，3304 轮 success 96.4%/home_born 92.7%。
      相机线回固定 LR 1e-4（= 教程 adaptive 均值水平；cam_v5 用它唯一
      实证到 87.6%）+ entropy 0.001，即 CAMERA_ALGO_CFG。
    env 侧沿用 CameraTutorialEnvCfg（奖励 ×1/30）。max_iterations 12000
    /save 100（cam_v5 峰值在 15125；缩放修复后预期显著提前）。
    """

    seed = 42
    num_steps_per_env = 64
    max_iterations = 12000
    save_interval = 100
    experiment_name = "pick_apple_staged_so101_camera"
    run_name = ""
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = CAMERA_PPO_ACTOR_CFG
    critic = STAGED_CRITIC_CFG
    algorithm = CAMERA_ALGO_CFG


@configclass
class CameraTutorialFixedLRPPORunnerCfg(PresetCfg):
    default = CameraTutorialFixedLRPPOBaseRunnerCfg()


# --------------------------------------------------------------------- #
# 相机 v9（gym id: Pick-Apple-SO101-Staged-Camera-v9）：蒸馏学生 PPO 收尾
# （2026-10-07 路线 A，用户批准）——actor 从 v9 蒸馏学生热启动（环境变量
# PICK_PPO_ACTOR_FROM_DISTILL，见 agents/hybrid_ppo.HybridPPO），critic 从
# --checkpoint（cam_v9 model_11999）热启动，optimizer 全新。**LR 3e-5**
# （cam_v12 实证的防漂移档：1e-4 续训两次跑飞 91→61%，3e-5 登顶 92.19%）。
# max_iterations 4000（256×192 渲染 ~12-15s/轮 ≈ 14-17h；蒸馏初始化起步
# 即强，峰预期早现，save 100 供守护收割）。
# --------------------------------------------------------------------- #


@configclass
class CameraFineTuneLowLRPPOBaseRunnerCfg(CameraTutorialFixedLRPPOBaseRunnerCfg):
    """v9 混合续训 runner：v4/v5/v7/v8 的 PPO 配方 + LR 3e-5 + HybridPPO。"""

    max_iterations = 4000
    algorithm = CAMERA_ALGO_CFG.replace(
        class_name="pick_anything.tasks.pick_apple.agents.hybrid_ppo:HybridPPO",
        learning_rate=3.0e-5,
    )


@configclass
class CameraFineTuneLowLRPPORunnerCfg(PresetCfg):
    default = CameraFineTuneLowLRPPOBaseRunnerCfg()
