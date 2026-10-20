"""扫掠 HOME 姿态候选：找「橘子+盒子同框」的出生相机视角（蒸馏 v4 前置探查）。

2026-10-02 用户定方向：出生画面必须同时含橘子与盒子（盒子可部分）——对齐
教程 place_vial「出生即见任务目标」的原则。当前 HOME（workshop 初始姿态）
相机只能看到桌沿 22×3.8cm 窄带：橘子贴上边缘、盒子完全不可见。

本脚本在蒸馏 v3 环境上（窗口+128×96+细动作），把 8 个候选抬臂姿态逐 env
写入关节，橘子摆窗口中心、盒子摆在候选位，实拍画面存拼图供人工选位。
用法（GPU2）：

    conda activate isaaclab_61 && cd ~/IsaacLab_6.1/Pick_Anything
    CUDA_VISIBLE_DEVICES=2 python scripts/probe_scan_pose.py
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
SETTLE_STEPS = 12
UPSCALE = 5

# workshop 初始姿态（robots.py WORKSHOP_INITIAL_JOINT_POSITION，臂 5 关节）
BASE_ARM = (-0.1221070742, -0.9066845838, 0.1900876486, 1.4797928525, -0.8044013083)
GRIPPER = 0.2555162024919699

# 候选姿态（在 BASE 上叠加：肩抬升/肘伸直/腕下压）——目标：相机抬高俯视桌面
P7 = (-0.122, -0.407, 0.090, 1.100, -0.804)
CANDIDATES: list[tuple[str, tuple[float, ...]]] = [
    ("W0_P7_o左_盒中深", P7),
    ("W1_P7_o中_盒中深", P7),
    ("W2_P7_o右_盒中深", P7),
    ("W3_P7_o左_盒左浅", P7),
    ("W4_P7_o中_盒左浅", P7),
    ("W5_P7_o右_盒左浅", P7),
]

# 盒子候选位（盒中心，桌内合法带 y∈[-0.52,-0.36]）；橘子按 env 序号取 3 个窗口位置
BOX_SPOTS = {
    "deep": (0.35, -0.50),
}
ORANGE_XYS = [(0.26, -0.32), (0.35, -0.32), (0.44, -0.32)]

R_MINUS_B, R_MINUS_G, R_MIN = 45, 15, 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe candidate HOME scan poses.")
    AppLauncher.add_app_launcher_args(parser)
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
    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV3PickAppleEnvCfg,
    )

    torch.manual_seed(0)
    n = len(CANDIDATES)
    cfg = CameraDistillV3PickAppleEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = n
    cfg.events.reset_staged.params["forced_stage"] = 0
    env = ManagerBasedRLEnv(cfg)

    try:
        cam = env.scene.sensors["wrist_camera"]
        robot = env.scene["robot"]
        arm_ids, arm_names = robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        )
        grip_ids, _ = robot.find_joints(["gripper"])
        print(f"臂关节序: {arm_names}", flush=True)

        def warmup() -> None:
            for _ in range(WARMUP_STEPS):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)

        def settle() -> None:
            for _ in range(SETTLE_STEPS):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)

        def write_pose(env_ids: torch.Tensor, arm: tuple[float, ...]) -> None:
            m = torch.zeros(len(env_ids), len(arm_ids), device=env.device)
            for j, v in enumerate(arm):
                m[:, j] = v
            robot.write_joint_position_to_sim(m, env_ids=env_ids, joint_ids=arm_ids)
            robot.write_joint_velocity_to_sim(
                torch.zeros_like(m), env_ids=env_ids, joint_ids=arm_ids
            )
            g = torch.full((len(env_ids), len(grip_ids)), GRIPPER, device=env.device)
            robot.write_joint_position_to_sim(g, env_ids=env_ids, joint_ids=grip_ids)
            robot.set_joint_position_target(m, env_ids=env_ids, joint_ids=arm_ids)
            robot.set_joint_position_target(g, env_ids=env_ids, joint_ids=grip_ids)

        def place_objects(box_xy: tuple[float, float]) -> None:
            ids = torch.arange(n, device=env.device)
            origins = t(env.scene.env_origins)
            # 橘子按 env 序号取 3 个窗口位置；盒子前 3 env 用 A 位、后 3 env 用 B 位
            yaw = torch.zeros(n, device=env.device)
            quat_o = torch.stack(
                [torch.zeros(n, device=env.device), torch.zeros(n, device=env.device),
                 torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1
            )
            pos_o = torch.zeros(n, 3, device=env.device)
            for i in range(n):
                pos_o[i, 0], pos_o[i, 1] = ORANGE_XYS[i % len(ORANGE_XYS)]
            pos_o[:, 2] = v0.FRUIT_SPAWN_Z
            env.scene["orange"].write_root_pose_to_sim_index(
                root_pose=torch.cat([pos_o + origins, quat_o], dim=-1), env_ids=ids
            )
            # 盒子：box_xy 给前半，浅位 (0.15,-0.40) 给后半（桌内合法带）
            quat_b = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(n, 1)
            pos_b = torch.zeros(n, 3, device=env.device)
            alt = (0.15, -0.40)
            for i in range(n):
                xy = box_xy if i < n // 2 else alt
                pos_b[i, 0], pos_b[i, 1] = xy
            pos_b[:, 2] = v0.BOX_SPAWN_Z
            env.scene["cardboard_box"].write_root_pose_to_sim_index(
                root_pose=torch.cat([pos_b + origins, quat_b], dim=-1), env_ids=ids
            )

        warmup()
        from PIL import Image, ImageDraw, ImageFont

        out_dir = PROJECT_ROOT / "photos" / "scan_pose"
        out_dir.mkdir(parents=True, exist_ok=True)
        font = ImageFont.load_default()

        for spot_name, box_xy in BOX_SPOTS.items():
            env.reset()
            for i, (name, arm) in enumerate(CANDIDATES):
                write_pose(torch.tensor([i], device=env.device), arm)
            place_objects(box_xy)
            settle()
            buf = cam.data.output
            rgb = t(buf["rgb"]) if buf and "rgb" in buf else None
            frames = rgb[:, :, :, :3].cpu().numpy()
            h, w = frames.shape[1:3]
            # 2 列拼图，每格放大 + 烧标签
            cols = 2
            rows = (n + cols - 1) // cols
            tw, th = w * UPSCALE, h * UPSCALE
            sheet = Image.new("RGB", (tw * cols, th * rows), (24, 24, 24))
            for i, (name, _) in enumerate(CANDIDATES):
                f = frames[i]
                r = f[..., 0].astype(np.int16)
                g = f[..., 1].astype(np.int16)
                b = f[..., 2].astype(np.int16)
                oj = int(((r - b > R_MINUS_B) & (r - g > R_MINUS_G) & (r > R_MIN)).sum())
                img = Image.fromarray(f).resize((tw, th), Image.NEAREST)
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, tw, 26), fill=(0, 0, 0))
                draw.text((6, 6), f"{name}  orange={oj}px", fill=(255, 255, 0), font=font)
                sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
                print(f"[{spot_name}] {name}: orange={oj}px  mean={f.mean():.0f}", flush=True)
            out = out_dir / f"round_{spot_name}.png"
            sheet.save(out)
            print(f"[save] {out.relative_to(PROJECT_ROOT)}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
