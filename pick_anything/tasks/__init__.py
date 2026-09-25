"""RL task definitions for Pick_Anything.

Importing this package registers all gym environments (currently the
apple-picking-into-box task ``Pick-Apple-SO101-v0``).
"""

from . import pick_apple  # noqa: F401  -- importing registers the gym task

__all__ = ["pick_apple"]
