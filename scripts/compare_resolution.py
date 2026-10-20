"""对比 64×48(v2) 与 128×96(v3) 腕摄下学生看到的橘子像素量（HOME 出生+窗口）。

蒸馏 v3 的核心假设：45mm 橘子在 22cm 外、64×48 下只有 ~8 像素圆斑撑不起
长程 BC；128×96 面积 ×4。本脚本 headless 实拍两种分辨率的真实传感器帧
（策略输入同口径），按颜色分割橘子连通像素并打印实测数字，--save 存放大
对比图。用法：

    conda activate isaaclab_61 && cd ~/IsaacLab_6.1/Pick_Anything
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_resolution.py --variant v2 --resets 6 --save
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_resolution.py --variant v3 --resets 6 --save
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

WARMUP_STEPS = 10
SETTLE_STEPS = 10
UPSCALE = 6
# 橘子颜色分割阈值（照片贴图的橙 R>>B；桌面/纸盒/黑爪均不满足）
R_MINUS_B = 45
R_MINUS_G = 15
R_MIN = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare orange pixel counts across camera resolutions.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--variant", choices=["v2", "v3"], required=True, help="v2=64x48, v3=128x96")
    parser.add_argument("--resets", type=int, default=6, help="抽样出生次数")
    parser.add_argument("--save", action="store_true", help="存放大对比图到 photos/res_compare/")
    return parser.parse_args()


def orange_mask(frame: np.ndarray) -> np.ndarray:
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return (r - b > R_MINUS_B) & (r - g > R_MINUS_G) & (r > R_MIN)


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV2PickAppleEnvCfg,
        CameraDistillV3PickAppleEnvCfg,
    )

    cfg = CameraDistillV3PickAppleEnvCfg() if args.variant == "v3" else CameraDistillV2PickAppleEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = 8
    cfg.events.reset_staged.params["forced_stage"] = 0  # HOME 出生（窗口内刷橘子）
    env = ManagerBasedRLEnv(cfg)

    try:
        cam = env.scene.sensors["wrist_camera"]
        state = getattr(env, "staged_pick_state", None)
        for _ in range(WARMUP_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        stats: list[int] = []
        for spawn_no in range(args.resets):
            env.reset()
            for _ in range(SETTLE_STEPS):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            buf = cam.data.output
            if not buf or "rgb" not in buf:
                print(f"[{args.variant}] spawn{spawn_no}: 传感器空帧", flush=True)
                continue
            rgb = buf["rgb"].torch if hasattr(buf["rgb"], "torch") else buf["rgb"]
            frame = rgb[0, :, :, :3].cpu().numpy()
            h, w = frame.shape[:2]
            mask = orange_mask(frame)
            n = int(mask.sum())
            stats.append(n)
            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if n else None
            xy = state.spawn_orange_xy[0].tolist() if state is not None else [0.0, 0.0]
            print(
                f"[{args.variant}] spawn{spawn_no}: {w}x{h}  橘子xy=({xy[0]:+.3f},{xy[1]:+.3f})"
                f"  橙色像素={n}px ({100.0 * n / (w * h):.2f}% 画面)  bbox(x0,x1,y0,y1)={bbox}",
                flush=True,
            )
            if args.save and n > 0:
                try:
                    from PIL import Image

                    img = Image.fromarray(frame).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)
                    out_dir = PROJECT_ROOT / "photos" / "res_compare"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{args.variant}_{w}x{h}_spawn{spawn_no}.png"
                    img.save(out)
                    print(f"      [save] {out.relative_to(PROJECT_ROOT)}", flush=True)
                except Exception as exc:  # PIL 缺失不致命
                    print(f"      [save 失败] {exc}", flush=True)

        if stats:
            print(
                f"\n== {args.variant} 汇总: {cfg.scene.wrist_camera.width}x{cfg.scene.wrist_camera.height}"
                f"  橙色像素 均值={np.mean(stats):.0f}px  中位={np.median(stats):.0f}px  "
                f"范围=[{min(stats)},{max(stats)}]  (n={len(stats)}) ==",
                flush=True,
            )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
