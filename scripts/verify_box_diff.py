"""差分渲染铁证：盒子在窗口位 vs 挪走，像素差 = 盒子的真实可见轮廓。

不再用颜色分割/肉眼判断（两轮误判教训）。每次真实出生（StagedReset）后：
渲染帧 A（盒子在出生位）→ 把盒子挪到桌角远位 → 渲染帧 B → 差分掩码
（|A-B|>阈值）即盒子+阴影的真实像素集合，打印 bbox 与面积，存三联图
A|B|diff 到 photos/box_diff/。用法（GPU2）：

    CUDA_VISIBLE_DEVICES=2 python scripts/verify_box_diff.py --resets 8
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

WARMUP_STEPS, SETTLE_STEPS, MOVE_SETTLE = 10, 10, 8
DIFF_THRESH = 18
UPSCALE = 5
PARK_XY = (0.12, -0.59)  # 桌角远位（避开视野）


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff-render ground truth for box visibility.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--resets", type=int, default=8)
    parser.add_argument(
        "--env",
        choices=["distill_v4", "camera_v5"],
        default="distill_v4",
        help="distill_v4=蒸馏 v4 环境(128×96)；camera_v5=相机直训 v5 环境(64×48，同款 O 姿态/双窗口)",
    )
    return parser.parse_args()


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple import mdp as v0  # noqa: E402
    from pick_anything.tasks.pick_apple.camera_env_cfg import (  # noqa: E402
        CameraScanHomeEnvCfg,
    )
    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV4PickAppleEnvCfg,
    )

    torch.manual_seed(0)
    cfg = CameraScanHomeEnvCfg() if args.env == "camera_v5" else CameraDistillV4PickAppleEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = 4
    cfg.events.reset_staged.params["forced_stage"] = 0
    env = ManagerBasedRLEnv(cfg)

    try:
        cam = env.scene.sensors["wrist_camera"]
        box = env.scene["cardboard_box"]
        robot = env.scene["robot"]
        state = getattr(env, "staged_pick_state", None)

        def run_steps(k: int) -> None:
            for _ in range(k):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)

        def grab() -> np.ndarray:
            buf = cam.data.output
            rgb = t(buf["rgb"]) if buf and "rgb" in buf else None
            return rgb[0, :, :, :3].cpu().numpy()

        run_steps(WARMUP_STEPS)
        out_dir = PROJECT_ROOT / "photos" / "box_diff"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for spawn_no in range(args.resets):
            env.reset()
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            run_steps(SETTLE_STEPS)
            frame_a = grab()
            o = state.spawn_orange_xy[0].tolist()
            b = state.spawn_box_xy[0].tolist()

            # 把盒子挪去桌角
            ids = torch.arange(cfg.scene.num_envs, device=env.device)
            origins = t(env.scene.env_origins)
            quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(len(ids), 1)
            pos = torch.zeros(len(ids), 3, device=env.device)
            pos[:, 0], pos[:, 1], pos[:, 2] = PARK_XY[0], PARK_XY[1], v0.BOX_SPAWN_Z
            box.write_root_pose_to_sim_index(
                root_pose=torch.cat([pos + origins, quat], dim=-1), env_ids=ids
            )
            run_steps(MOVE_SETTLE)
            frame_b = grab()

            # 橘子质心(颜色分割,橘子可靠)
            r_, g_, b_ = (frame_a[..., i].astype(np.int16) for i in range(3))
            om = (r_ - b_ > 45) & (r_ - g_ > 15) & (r_ > 70)
            oc = (float(np.nonzero(om)[1].mean()), float(np.nonzero(om)[0].mean())) if om.any() else (float("nan"),) * 2

            diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16)).max(axis=-1)
            mask = diff > DIFF_THRESH
            n = int(mask.sum())
            if n > 0:
                ys, xs = np.nonzero(mask)
                bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
                h, w = mask.shape
                results.append(n)
                print(
                    f"[spawn{spawn_no}] 橘子=({o[0]:+.3f},{o[1]:+.3f}) 盒=({b[0]:+.3f},{b[1]:+.3f})"
                    f"  盒px={n} 盒心≈({(bbox[0]+bbox[1])/2:.0f},{(bbox[2]+bbox[3])/2:.0f}) 橘心≈({oc[0]:.0f},{oc[1]:.0f})"
                    f"  底缘裁切={'是' if bbox[3] >= h-2 else '否'}",
                    flush=True,
                )
            else:
                results.append(0)
                print(f"[spawn{spawn_no}] 差分=0 —— 盒子完全不可见!", flush=True)

            try:
                from PIL import Image

                h, w = frame_a.shape[:2]
                sheet = Image.new("RGB", (w * UPSCALE * 3, h * UPSCALE), (24, 24, 24))
                for j, f in enumerate(
                    (frame_a, frame_b, np.stack([mask * 255] * 3, axis=-1).astype(np.uint8))
                ):
                    img = Image.fromarray(f).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)
                    sheet.paste(img, (j * w * UPSCALE, 0))
                sheet.save(out_dir / f"spawn{spawn_no}_A_B_diff.png")
            except Exception as exc:
                print(f"      [save 失败] {exc}", flush=True)

        vis = sum(1 for r in results if r > 50)
        print(
            f"\n== 汇总: {vis}/{len(results)} 出生盒子可见(>50px)，"
            f"像素均值={np.mean(results) if results else 0:.0f}px ==",
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
