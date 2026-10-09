"""Camera-v8 PPO configuration and shared camera student network.

The actor consumes RGB and proprioception; the critic consumes privileged state.
PPO uses a fixed learning rate of 1e-4 and Gaussian std bounds (0.05, 0.3).
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
"""Shared compact encoder; input dimensions follow the task camera resolution."""


@configclass
class CameraBoundedGaussianCfg(StagedBoundedGaussianCfg):
    """Gaussian std bounds shared by the validated camera network."""

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
    """Gaussian std bounds shared by the validated camera network."""

    std_range: tuple[float, float] = (0.05, 0.3)


# Keep the checkpoint-compatible actor distribution configuration.
CAMERA_PPO_ACTOR_CFG = CAMERA_ACTOR_CFG.replace(
    distribution_cfg=CameraPPOBoundedGaussianCfg(init_std=0.2, std_type="log")
)

# Validated fixed-rate PPO recipe.
CAMERA_ALGO_CFG = STAGED_ALGO_CFG.replace(
    entropy_coef=0.001,
    learning_rate=1.0e-4,
    schedule="fixed",
)


@configclass
class CameraPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Validated Camera-v8 PPO recipe: fixed LR, CNN actor and privileged critic."""

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
class CameraPPORunnerCfg(PresetCfg):
    default = CameraPPOBaseRunnerCfg()
