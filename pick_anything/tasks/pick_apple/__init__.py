"""Gym registration for the apple-picking-into-box task."""

import gymnasium as gym

from . import agents
from .pick_env_cfg import PickAppleEnvCfg

gym.register(
    id="Pick-Apple-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_env_cfg:PickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickApplePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 八阶段版（物理验证 reset 状态库 + 课程学习 + 非对称 AC）；v0 保持不变
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
