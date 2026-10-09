"""RL task definitions for Pick_Anything.

Importing this package registers the eight-stage coordinate PPO task,
validated camera PPO tasks and camera distillation v10.
"""

from . import pick_apple  # noqa: F401  -- importing registers the gym task

__all__ = ["pick_apple"]
