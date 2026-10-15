"""Opt-in multi-object, mask-conditioned PPO grasp-and-lift task."""

import gymnasium as gym

TASK_ID = "Pick-Anything-SO101-OpenVocab-v0"


def register_task():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID, entry_point="pick_anything.open_vocab_pick.env:OpenVocabPickEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "pick_anything.open_vocab_pick.config:OpenVocabPickEnvCfg",
                "rsl_rl_cfg_entry_point": "pick_anything.open_vocab_pick.config:OpenVocabPickRunnerCfg",
                "default_agent": "rsl_rl",
            },
        )
