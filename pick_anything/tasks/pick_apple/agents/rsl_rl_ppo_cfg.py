"""RSL-RL PPO configuration for the apple-picking task.

Recipe follows the proven Isaac-Lift-SO101 run on this machine (rsl-rl 5.5.1):
obs-normalized ELU MLPs for actor and critic, adaptive-KL PPO. The task is
low-dimensional (proprio + object poses), so the same [512, 256, 128] trunk
is comfortable.
"""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_tasks.utils import PresetCfg

STATE_POLICY_CFG = RslRlMLPModelCfg(
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

STATE_CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    # 2026-09-23: entropy_coef 0.008 + unbounded scalar std drove a runaway
    # entropy spiral (std 1.0 -> 7.7), but 0.001 collapsed exploration too
    # far the other way (std froze at 0.17 and the policy dead-locked in a
    # hover optimum); 0.003 sits between the two observed failure modes
    entropy_coef=0.003,
    num_learning_epochs=5,
    num_mini_batches=4,
    # smaller step + a KL band wide enough that the adaptive schedule stops
    # chain-halving the LR into the 1e-5 mud (observed on run 11-50: the
    # entropy fix made each update bigger, KL persistently overshot 2x the
    # old 0.008 target and the LR collapsed 5e-4 -> 1e-5)
    learning_rate=5.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.90,
    desired_kl=0.016,
    max_grad_norm=1.0,
)


@configclass
class PickApplePPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class PickApplePPORunnerCfg(PresetCfg):
    default = PickApplePPOBaseRunnerCfg().replace(
        experiment_name="pick_apple_so101",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=STATE_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG,
    )
