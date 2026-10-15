"""Opt-in language/mask picking task; existing entry points never import this package."""

import gymnasium as gym

TASK_ID = "Pick-Orange-SO101-Grounded-v0"


def register_task() -> None:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="pick_anything.grounded_pick.env:GroundedPickEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "pick_anything.grounded_pick.config:GroundedPickEnvCfg",
                "rsl_rl_cfg_entry_point": "pick_anything.grounded_pick.config:GroundedPickRunnerCfg",
                "default_agent": "rsl_rl",
            },
        )
