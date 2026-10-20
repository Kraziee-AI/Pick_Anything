"""Supported apple-picking tasks and validated camera PPO entry points."""

import gymnasium as gym
from . import agents

gym.register(
    id="Pick-Apple-SO101-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.staged_env_cfg:StagedPickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.staged_ppo_cfg:StagedPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v8",
    entry_point=f"{__name__}.camera_env:CameraPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraClosureFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Pick-Apple-SO101-Staged-Camera-Distill-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_distill_v10_cfg:CameraDistillV10PickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.camera_distill_v10_cfg:CameraDistillV10RunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="Pick-Apple-SO101-Staged-Camera-Filtered-v0",
    entry_point=f"{__name__}.camera_env:CameraPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smooth_camera_cfg:CameraFilteredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.smooth_camera_cfg:CameraSmoothPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
