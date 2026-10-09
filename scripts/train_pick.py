"""Train the apple-picking policy (registers the task, then runs the unified isaaclab train CLI).

Run from the Pick_Anything root so logs land in ./logs:

    conda activate isaaclab_61
    cd ~/IsaacLab_6.1/Pick_Anything

八阶段版（reset 状态库 + 课程学习 + 非对称 AC）：

    python scripts/train_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \\
        --num_envs 1024 --device cuda:0 --seed 42 --max_iterations 13000 --viz none

小规模调试：

    python scripts/train_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \\
        --num_envs 32 --max_iterations 5 --viz none

"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pick_anything.tasks  # noqa: F401,E402  -- importing registers the supported tasks

from isaaclab_rl.entrypoints.dispatch import run_train_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_train_cli())
