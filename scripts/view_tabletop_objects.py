"""Opt-in multi-object scene preview; --inspect runs on CPU without Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pick_anything.tabletop.catalog import (
    BOX_ASSET, BOX_KEY, DEFAULT_OBJECTS, OBJECTS, object_keys, resolve_target, sample_layout,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", action="store_true", help="仅打印布局／语言目标，不启动仿真")
    p.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS), choices=list(OBJECTS))
    p.add_argument("--instruction", default="抓小黄鸭", help="单目标词汇映射，预览不会执行抓取")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-box", action="store_true", help="显式隐藏纸盒；默认保留原纸盒")
    p.add_argument("--steps", type=int, default=600, help="仿真步数；0 表示持续预览")
    return p


def main():
    p = parser()
    preliminary, _ = p.parse_known_args()
    if not preliminary.inspect:
        cache = PROJECT_ROOT / "outputs/tabletop_objects/cache"
        for key, directory in {
            "XDG_CACHE_HOME": "xdg", "WARP_CACHE_PATH": "warp", "TORCH_HOME": "torch",
            "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda",
            "MPLCONFIGDIR": "matplotlib",
        }.items():
            os.environ[key] = str(cache / directory)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    keys = object_keys(args.objects)
    try:
        target = resolve_target(args.instruction, keys)
        layout = sample_layout(keys, args.seed, include_box=not args.no_box)
        if args.steps < 0:
            raise ValueError("steps 必须非负。")
        for key in keys:
            if not OBJECTS[key].asset.is_file():
                raise ValueError(f"缺少资产：{OBJECTS[key].asset}")
        if not args.no_box and not BOX_ASSET.is_file():
            raise ValueError(f"缺少资产：{BOX_ASSET}")
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps({
        "target": {"key": target.key, "label": target.label, "caption": target.caption,
                   "prim_path": f"/World/envs/env_0/{target.prim_name}"},
        "objects": {key: {"asset": str(OBJECTS[key].asset), **layout[key]} for key in keys},
        "cardboard_box": None if args.no_box else {"asset": str(BOX_ASSET), **layout[BOX_KEY]},
        "mode": "scene_preview_only",
    }, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return

    launcher = AppLauncher(args)
    app = launcher.app
    try:
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from pick_anything.tabletop.scene_cfg import make_scene_cfg

        sim_cfg = SimulationCfg(dt=1 / 120, physics=PhysxCfg(), device=args.device)
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(make_scene_cfg(keys, args.seed, include_box=not args.no_box))
        sim.set_camera_view(eye=[0.95, -1.2, 0.85], target=[0.35, -0.43, 0.08])
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        defaults = robot.data.default_joint_pos
        defaults = defaults.torch if hasattr(defaults, "torch") else defaults
        tick = 0
        while app.is_running() and (args.steps == 0 or tick < args.steps):
            robot.set_joint_position_target(defaults)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            tick += 1
        for key in layout:
            position = scene[key].data.root_pos_w
            position = position.torch if hasattr(position, "torch") else position
            print(f"{key} settled position: {position[0].cpu().tolist()}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
