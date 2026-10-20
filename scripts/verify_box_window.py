"""验证蒸馏 v4 的 HOME 出生：盒子/橘子窗口落位 + 同框实拍（真实 StagedReset 路径）。

不手摆位姿——直接构建 CameraDistillV4PickAppleEnvCfg，env.reset() 即训练真实
出生（P7 姿态 + 橘子窗口 + 盒子窗口）。打印每次出生的橘子/盒子坐标是否落窗、
橙色像素数，并存帧到 photos/box_window/ 供人工确认同框。用法（GPU2）：

    CUDA_VISIBLE_DEVICES=2 python scripts/verify_box_window.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

WARMUP_STEPS, SETTLE_STEPS, UPSCALE = 10, 10, 5
R_MINUS_B, R_MINUS_G, R_MIN = 45, 15, 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify v4 HOME spawn windows via real StagedReset.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--resets", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV4PickAppleEnvCfg,
    )

    torch.manual_seed(0)
    cfg = CameraDistillV4PickAppleEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = 8
    cfg.events.reset_staged.params["forced_stage"] = 0
    env = ManagerBasedRLEnv(cfg)

    try:
        cam = env.scene.sensors["wrist_camera"]
        state = getattr(env, "staged_pick_state", None)
        for _ in range(WARMUP_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        out_dir = PROJECT_ROOT / "photos" / "box_window"
        out_dir.mkdir(parents=True, exist_ok=True)
        for spawn_no in range(args.resets):
            env.reset()
            for _ in range(SETTLE_STEPS):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            o = state.spawn_orange_xy[0].tolist()
            b = state.spawn_box_xy[0].tolist()
            o_in = 0.31 <= o[0] <= 0.38 and -0.38 <= o[1] <= -0.34
            b_in = 0.40 <= b[0] <= 0.48 and -0.50 <= b[1] <= -0.44
            sep = float(np.hypot(o[0] - b[0], o[1] - b[1]))
            buf = cam.data.output
            rgb = (buf["rgb"].torch if hasattr(buf["rgb"], "torch") else buf["rgb"])[0, :, :, :3]
            frame = rgb.cpu().numpy()
            r, g, bb = (frame[..., i].astype(np.int16) for i in range(3))
            oj = int(((r - bb > R_MINUS_B) & (r - g > R_MINUS_G) & (r > R_MIN)).sum())
            print(
                f"[spawn{spawn_no}] 橘子=({o[0]:+.3f},{o[1]:+.3f}){'✓' if o_in else '✗出窗'}  "
                f"盒子=({b[0]:+.3f},{b[1]:+.3f}){'✓' if b_in else '✗出窗'}  "
                f"果盒距={sep:.3f}m  橙px={oj}",
                flush=True,
            )
            try:
                from PIL import Image

                h, w = frame.shape[:2]
                Image.fromarray(frame).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST).save(
                    out_dir / f"spawn{spawn_no}.png"
                )
            except Exception as exc:
                print(f"      [save 失败] {exc}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
