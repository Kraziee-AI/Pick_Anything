"""Visualize a trained apple-picking policy (unified isaaclab play CLI).

Run from the Pick_Anything root:

    python scripts/play_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \\
        --num_envs 4 --device cuda:2 --checkpoint /path/to/model_XXXX.pt --viz kit

注意：`--viz kit` 必须配合 LD_PRELOAD 预加载 conda 的 libcudart（kit 与
conda 自带 CUDA 运行库冲突），见 scripts/view_scene.sh 同款写法：
    export LD_PRELOAD=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _prelaunch_kit_if_requested(argv: list[str]) -> None:
    """`--viz kit` 时先启动 SimulationApp，再进入 CLI 注册流程。

    Staged 任务的环境配置在模块顶层导入 isaaclab_newton，其内部
    `from pxr import Usd` 若发生在 SimulationApp 之前，会以 site-packages
    的 pxr（usd_exchange 轮子）抢先加载；GUI kit 随后加载自带 USD 扩展时
    双方绑定冲突（"extension class wrapper ... has not been created yet"），
    kit 启动即崩（约 4% 处）。kitless 的训练 / --viz none 路径不受影响。
    先把 kit 拉起来，pxr 统一解析到 kit 内部实现，冲突消失；
    后续 launch_simulation 见 has_kit() 为真会跳过二次启动。
    """
    if "-h" in argv or "--help" in argv:
        return
    for index, token in enumerate(argv):
        is_kit = (token == "--viz" and index + 1 < len(argv) and argv[index + 1] == "kit") or (
            token.startswith("--viz=") and token.split("=", 1)[1] == "kit"
        )
        if not is_kit:
            continue
        parser = argparse.ArgumentParser(add_help=False)
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        launcher_args, _ = parser.parse_known_args(argv)
        AppLauncher(launcher_args)
        return


_prelaunch_kit_if_requested(sys.argv[1:])

import pick_anything.tasks  # noqa: F401,E402  -- importing registers the Pick-Apple tasks

from isaaclab_rl.entrypoints.dispatch import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
