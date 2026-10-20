"""出生视角几何探针：量化「离桌子太近、视野小」并扫掠更高位 HOME 候选。

2026-10-02 用户反馈：distill v4 的 P7 扫视 HOME 出生画面仍嫌相机贴桌、
可见桌面太小。相机 prim（含光学）与教程同一 USD，FOV 无差异可抄——
能调的只有姿态（相机离地高度/俯角）或显式改光圈（偏离教程同参，慎选）。

本脚本在蒸馏 v3 环境上把候选臂姿态逐 env 写入，打印每个姿态的：
* 相机内参（fx/fy → 水平/垂直 FOV，USD 光学直读）；
* 相机相对桌面的高度与俯角；
* 四边缘射线与桌面（z=TABLE_TOP_Z）的交点四边形 = 实际可见桌面足迹；
并实拍橘子/盒子同框画面存 photos/cam_geometry/ 供人工选位。

用法（GPU2）：

    conda activate isaaclab_61 && cd ~/IsaacLab_6.1/Pick_Anything
    CUDA_VISIBLE_DEVICES=2 python scripts/probe_camera_geometry.py
"""

from __future__ import annotations

import argparse
import math
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
TABLE_TOP_Z = 0.041

# 关节序：shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
# 注意 wrist_flex 上限 95°=1.658rad（USD 关节限位），候选不得超出。
WORKSHOP = (-0.1221070742, -0.9066845838, 0.1900876486, 1.4797928525, -0.8044013083)
SCAN_HOME = (-0.05, -0.45, -0.250, 1.658, 0.0)  # 现状 P7
# 定姿态扫盒位：A 方案选定 O 姿态（workshop roll=-0.804 族，pan 居中，
# wf=1.58；实测相机 23.0cm 高、俯角 26.4°，橘盒同列居中且对投影约定误差
# 免疫）。本扫掠在 O 姿态下逐 env 摆不同盒位（v0 盒矩形 x[0.14,0.56]
# y[-0.51,-0.39] ∩ 可达环 r[0.16,0.24] 内取样），找盒子投影离开底缘/
# 相机正下方的窗口。第三元保留但不再用于 FOV 分组（运行时改光圈无效已实测）。
POSE_O = (-0.05, -0.907, 0.190, 1.58, -0.804)
FOCAL = 13.6
BOX_CANDIDATES = [
    ("C0_现状0.44-0.47", (0.44, -0.47)),   # r=0.242 现状(底缘/爪后)
    ("C1_0.42-0.42", (0.42, -0.42)),       # r=0.231 深
    ("C2_0.46-0.42", (0.46, -0.42)),       # r=0.246 深右
    ("C3_0.48-0.41", (0.48, -0.41)),       # r=0.264 深右(超环)
    ("C4_0.44-0.40", (0.44, -0.40)),       # r=0.256 最深(超环)
    ("C5_0.40-0.40", (0.40, -0.40)),       # r=0.245 深左
    ("C6_0.52-0.44", (0.52, -0.44)),       # r=0.262 最右深(超环)
    ("C7_0.38-0.41", (0.38, -0.41)),       # r=0.232 深左(贴橘)
]
CANDIDATES: list[tuple[str, tuple[float, ...], bool]] = [
    (name, POSE_O, False) for name, _ in BOX_CANDIDATES
]
GRIPPER = 0.2555162024919699

ORANGE_XYS = [(0.345, -0.36)]  # 橘子固定橘窗中心，单变量扫盒位

R_MINUS_B, R_MINUS_G, R_MIN = 45, 15, 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe camera geometry of HOME poses.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(x,y,z,w) → 3x3 旋转矩阵（IsaacLab quat_w_world 的顺序）。"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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

        # USD 光学直读（env_0；所有 env 同参数）
        stage = env.sim.stage
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/wowrobo_2MP_camera")
        focal = cam_prim.GetAttribute("focalLength").Get()
        h_ap = cam_prim.GetAttribute("horizontalAperture").Get()
        v_ap = cam_prim.GetAttribute("verticalAperture").Get()
        clip = cam_prim.GetAttribute("clippingRange").Get()
        hfov = 2 * math.degrees(math.atan(h_ap / (2 * focal)))
        vfov = 2 * math.degrees(math.atan(v_ap / (2 * focal)))
        print(
            f"[USD光学] focalLength={focal:.3f}mm hAp={h_ap:.3f}mm vAp={v_ap:.3f}mm "
            f"clip={clip} → HFOV={hfov:.1f}° VFOV={vfov:.1f}°",
            flush=True,
        )

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

        def place_objects() -> None:
            ids = torch.arange(n, device=env.device)
            origins = t(env.scene.env_origins)
            quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(n, 1)
            pos_o = torch.zeros(n, 3, device=env.device)
            for i in range(n):
                pos_o[i, 0], pos_o[i, 1] = ORANGE_XYS[i % len(ORANGE_XYS)]
            pos_o[:, 2] = v0.FRUIT_SPAWN_Z
            env.scene["orange"].write_root_pose_to_sim_index(
                root_pose=torch.cat([pos_o + origins, quat], dim=-1), env_ids=ids
            )
            pos_b = torch.zeros(n, 3, device=env.device)
            for i in range(n):
                pos_b[i, 0], pos_b[i, 1] = BOX_CANDIDATES[i][1]
            pos_b[:, 2] = v0.BOX_SPAWN_Z
            env.scene["cardboard_box"].write_root_pose_to_sim_index(
                root_pose=torch.cat([pos_b + origins, quat], dim=-1), env_ids=ids
            )

        warmup()
        from PIL import Image, ImageDraw, ImageFont

        out_dir = PROJECT_ROOT / "photos" / "cam_geometry"
        out_dir.mkdir(parents=True, exist_ok=True)
        font = ImageFont.load_default()

        env.reset()
        for i, (_, arm, _) in enumerate(CANDIDATES):
            write_pose(torch.tensor([i], device=env.device), arm)
        place_objects()
        settle()

        # ---- 几何量 ----
        pos_w = cam.data.pos_w.cpu().numpy()          # (N,3) 世界系
        quat_w = cam.data.quat_w_world.cpu().numpy()  # (N,4) xyzw，world 惯例 +X 朝前/+Z 朝上
        origins = t(env.scene.env_origins).cpu().numpy()
        K = cam.data.intrinsic_matrices.cpu().numpy()  # (N,3,3)
        img_h, img_w = cam.image_shape[0], cam.image_shape[1]
        actual_joints = robot.data.joint_pos[:, arm_ids].cpu().numpy()

        print(f"\n{'姿态':<22}{'相机高(cm)':>10}{'俯角(°)':>9}{'HFOV':>7}{'VFOV':>7}  桌面可见足迹(env局部 m)", flush=True)
        for i, (name, _, _) in enumerate(CANDIDATES):
            p = pos_w[i] - origins[i]
            h_cm = (p[2] - TABLE_TOP_Z) * 100
            R = quat_to_mat(quat_w[i])
            fwd = R @ np.array([1.0, 0.0, 0.0])  # world 惯例相机朝 +X
            tilt = math.degrees(math.acos(np.clip(-fwd[2], -1, 1)))  # 与竖直向下的夹角
            fx, fy = K[i][0, 0], K[i][1, 1]
            cx, cy = K[i][0, 2], K[i][1, 2]
            hf = 2 * math.degrees(math.atan(img_w / (2 * fx)))
            vf = 2 * math.degrees(math.atan(img_h / (2 * fy)))
            # 图像四角射线（world 相机系：+X 前 / +Y 左 / +Z 上）与桌面求交
            corners = []
            for u, v in [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]:
                d_cam = np.array([1.0, -(u - cx) / fx, -(v - cy) / fy])
                d = R @ d_cam
                if d[2] >= -1e-6:
                    corners.append(None)
                    continue
                s = (TABLE_TOP_Z - p[2]) / d[2]
                hit = p + s * d
                corners.append((round(float(hit[0]), 3), round(float(hit[1]), 3)))
            aj = " ".join(f"{v:+.2f}" for v in actual_joints[i])
            axis_hit = None
            if fwd[2] < -1e-6:
                s0 = (TABLE_TOP_Z - p[2]) / fwd[2]
                ah = p + s0 * fwd
                axis_hit = (round(float(ah[0]), 3), round(float(ah[1]), 3))
            # 把橘子/盒子中心投到像素坐标（world 相机系 +X前/+Y左/+Z上）
            def to_px(xy: tuple[float, float], z: float) -> tuple[int, int] | None:
                c = R.T @ (np.array([xy[0], xy[1], z]) - p)
                if c[0] <= 1e-6:
                    return None
                return (int(cx - fx * c[1] / c[0]), int(cy - fy * c[2] / c[0]))
            o_xy = ORANGE_XYS[i % len(ORANGE_XYS)]
            b_xy = BOX_CANDIDATES[i][1]
            print(
                f"{name:<22}{h_cm:>9.1f}{tilt:>9.1f}{hf:>7.1f}{vf:>7.1f}  {corners}\n"
                f"  相机xy=({p[0]:.3f},{p[1]:.3f}) 光轴落点={axis_hit} "
                f"橘子{o_xy}→px{to_px(o_xy, v0.FRUIT_SPAWN_Z)} "
                f"盒子{b_xy}→px{to_px(b_xy, v0.BOX_SPAWN_Z)} (图{img_w}x{img_h})\n"
                f"  实际关节: {aj}",
                flush=True,
            )

        # ---- 实拍拼图 ----
        buf = cam.data.output
        rgb = t(buf["rgb"]) if buf and "rgb" in buf else None
        frames = rgb[:, :, :, :3].cpu().numpy()
        h, w = frames.shape[1:3]
        cols = 2
        rows = (n + cols - 1) // cols
        tw, th = w * UPSCALE, h * UPSCALE
        sheet = Image.new("RGB", (tw * cols, th * rows), (24, 24, 24))
        for i, (name, _, _) in enumerate(CANDIDATES):
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
        out = out_dir / "poses.png"
        sheet.save(out)
        print(f"[save] {out.relative_to(PROJECT_ROOT)}", flush=True)

        # ---- 盒子差分（铁证）：盒子挪去桌角后重拍，像素差=盒子真实可见轮廓 ----
        PARK_XY = (0.12, -0.59)
        ids = torch.arange(n, device=env.device)
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(n, 1)
        pos_p = torch.zeros(n, 3, device=env.device)
        pos_p[:, 0], pos_p[:, 1], pos_p[:, 2] = PARK_XY[0], PARK_XY[1], v0.BOX_SPAWN_Z
        env.scene["cardboard_box"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_p + t(env.scene.env_origins), quat], dim=-1), env_ids=ids
        )
        settle()
        buf = cam.data.output
        frames_b = t(buf["rgb"])[:, :, :, :3].cpu().numpy()

        diff_sheet = Image.new("RGB", (tw * cols, th * rows), (24, 24, 24))
        print("\n盒子差分可见性（掩码 bbox / 像素数，图 %dx%d）：" % (w, h), flush=True)
        for i, (name, _, _) in enumerate(CANDIDATES):
            diff = np.abs(frames[i].astype(np.int16) - frames_b[i].astype(np.int16)).max(axis=-1)
            mask = diff > 18
            cnt = int(mask.sum())
            if cnt > 0:
                ys, xs = np.nonzero(mask)
                bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
                print(f"  {name:<22} 盒px={cnt:>5} bbox=u[{bbox[0]},{bbox[1]}] v[{bbox[2]},{bbox[3]}]", flush=True)
            else:
                print(f"  {name:<22} 盒px=0 —— 完全不可见!", flush=True)
            img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").resize((tw, th), Image.NEAREST)
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, tw, 26), fill=(0, 0, 0))
            draw.text((6, 6), f"{name} box={cnt}px", fill=(255, 255, 0), font=font)
            diff_sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
        out = out_dir / "poses_boxdiff.png"
        diff_sheet.save(out)
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
