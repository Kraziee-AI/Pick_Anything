"""RSL-RL PPO 配置：八阶段抓苹果任务（非对称 Actor-Critic）。

* obs_groups：actor 用 policy 组（相对量+里程碑），critic 用 critic 组
  （policy + 接触力/阶段 one-hot 特权）——与 staged_env_cfg 的两个观测组
  一一对应；
* std 上下界钳制（IsaacLabTutorial BoundedGaussian 配方）：std_type="log"
  + std_range=(0.05, 0.15)。fix2 训练实测无下界 log std 会单调收缩
  （0.144→0.109、entropy -3.5→-5.2），策略提前收敛进「开爪悬停」局部
  最优；下界 0.05 保住探索。上界 2026-09-26 由 0.3 降到 0.15：entropy
  0.005 会把 std 全程顶死在上限（旧 run 实测恒 0.30），训练行为=均值+
  σ0.3 白噪声激励，而 play 只跑均值（rsl_rl get_policy 确定性输出）——
  造成 train 90% vs 确定性推理 20-26% 的断层（对照探针实测）。上界 0.15
  让采样行为≈均值行为；若 GRASP 段进度停滞（旧 0.109 收缩的教训），
  优先回调上界到 0.2 而不是动下界；
* 其余超参对齐教程 PPO（entropy 0.005 / lr 3e-4 / desired_kl 0.01 /
  lam 0.95 / 8 mini-batch），entropy 系数在 std 有界后可安全调高。
"""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_tasks.utils import PresetCfg


@configclass
class StagedBoundedGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """std 钳制在 std_range 内的高斯策略头（教程 BoundedGaussianDistributionCfg 同款）。"""

    # 上界 0.15 的依据见模块 docstring（train/play 行为一致性）；init_std
    # 对齐上界，开局即目标值，不经历 0.2→0.15 的钳制瞬态
    std_range: tuple[float, float] = (0.05, 0.15)


STAGED_ACTOR_CFG = RslRlMLPModelCfg(
    distribution_cfg=StagedBoundedGaussianCfg(init_std=0.15, std_type="log"),
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

STAGED_CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

STAGED_ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,  # std 有界后安全值（教程配方）；无界时代 0.008 曾失控
    num_learning_epochs=5,
    num_mini_batches=8,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class StagedPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class StagedPPORunnerCfg(PresetCfg):
    default = StagedPPOBaseRunnerCfg().replace(
        experiment_name="pick_apple_staged_so101",
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=STAGED_ACTOR_CFG,
        critic=STAGED_CRITIC_CFG,
        algorithm=STAGED_ALGO_CFG,
    )


# --------------------------------------------------------------------- #
# 细粒度特权教师专用 PPO（TeacherFine-v0 任务）。演化史（2026-10-01）：
# * v1：staged 默认 tanh 奖励表 → 细粒度远场零梯度，home_born≡0；
# * v2：粗粒度线的 σ≤0.15+自适应 LR → KL×4 调度饿死（std 钉 0.15、LR
#   触底、mean_stage 3.0→1.1 崩塌）；
# * v3：std (0.05,0.3)+固定 LR 1e-4+entropy 0.001 → 回报量级 30× 压垮
#   critic（Loss/value 500-1000 vs 教程 0.1，cam_v5 复盘同病根），
#   success 0.32→0.11 下滑；
# * v4（当前）：与 cam_v6/CameraTutorialEnvCfg 同步——环境侧全表奖励
#   ×1/30 对齐教程回报量级（~7/局），算法侧直接用 STAGED_ALGO_CFG
#   （它本就是教程逐参数配方 entropy 0.005/adaptive 3e-4/desired_kl
#   0.01；此前 adaptive 触底是回报量级问题而非调度本身）。std 钉 0.3
#   上界在教程日志同样全程存在、无害。actor 仍是 STAGED_ACTOR_CFG 的
#   MLP（512/256/128 + obs_norm）仅分布参数不同——蒸馏 teacher strict
#   load 不受影响。粗粒度状态线 StagedPPORunnerCfg 不动。
# --------------------------------------------------------------------- #


@configclass
class FineTeacherBoundedGaussianCfg(StagedBoundedGaussianCfg):
    """细粒度教师的 std 钳制：上界放宽到 0.3（教程口径）。"""

    std_range: tuple[float, float] = (0.05, 0.3)


@configclass
class StagedFineTeacherPPORunnerCfg(PresetCfg):
    default = StagedPPOBaseRunnerCfg().replace(
        experiment_name="pick_apple_staged_so101",
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=STAGED_ACTOR_CFG.replace(
            distribution_cfg=FineTeacherBoundedGaussianCfg(init_std=0.2, std_type="log")
        ),
        critic=STAGED_CRITIC_CFG,
        algorithm=STAGED_ALGO_CFG,
        num_steps_per_env=64,
    )
