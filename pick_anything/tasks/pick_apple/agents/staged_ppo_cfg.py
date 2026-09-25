"""RSL-RL PPO 配置：八阶段抓苹果任务（非对称 Actor-Critic）。

* obs_groups：actor 用 policy 组（相对量+里程碑），critic 用 critic 组
  （policy + 接触力/阶段 one-hot 特权）——与 staged_env_cfg 的两个观测组
  一一对应；
* std 上下界钳制（IsaacLabTutorial BoundedGaussian 配方）：std_type="log"
  + std_range=(0.05, 0.3)。fix2 训练实测无下界 log std 会单调收缩
  （0.144→0.109、entropy -3.5→-5.2），策略提前收敛进「开爪悬停」局部
  最优；下界 0.05 保住探索，上界 0.3 防历史版本的无界膨胀（std 1.0→7.7）；
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

    std_range: tuple[float, float] = (0.05, 0.3)


STAGED_ACTOR_CFG = RslRlMLPModelCfg(
    distribution_cfg=StagedBoundedGaussianCfg(init_std=0.2, std_type="log"),
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
