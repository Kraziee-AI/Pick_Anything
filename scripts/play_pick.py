"""Visualize a trained apple-picking policy (unified isaaclab play CLI).

Run from the Pick_Anything root:

    python scripts/play_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-v0 \\
        --num_envs 4 --device cuda:0 --checkpoint /path/to/model_XXXX.pt --viz kit
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pick_anything.tasks  # noqa: F401,E402  -- importing registers Pick-Apple-SO101-v0

from isaaclab_rl.entrypoints.dispatch import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
