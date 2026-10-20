"""Route B 相机拟合探针：dump 闭包相机在 Isaac 端的精确世界位姿。

背景（sim2sim Gate 1）：MuJoCo 直搬闭包参数后 v 对上（0.135≈0.14）但 u 偏
（0.298 vs 预期带 0.38-0.64）、盒差分 2×——怀疑挂载朝向有残余误差。本脚本在
**闭包相机任务**（CameraClosureScanHomeEnvCfg = cam_v12 同款相机口径）上
写死 O 姿态 + 固定橘/盒位，dump：
* cam pos_w / quat_w_world（Isaac 惯例 xyzw、+X 前/+Z 上）
* gripper body pos_w / quat_w（wxyz）
* 内参 K 与橘/盒像素投影（Isaac 真值，MuJoCo 对拍用）
* 换算好的 MJCF 本地挂载（pos + quat wxyz + fovy）——可直接粘进 XML

用法（GPU 空闲卡）：
  conda activate isaaclab_61 && cd ~/IsaacLab_6.1/Pick_Anything
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_closure_camera.py
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

# O 姿态（SCAN_HOME_ARM_JOINTS，camera_env_cfg.py:352-359）+ 第二姿态
# （挂载常量性检验：两姿态下 Isaac 相机世界位姿应与 MJCF FK·拟合挂载一致）
POSE_O = (-0.05, -0.907, 0.190, 1.580, -0.804)
POSE_2 = (-0.30, -0.600, 0.500, 1.400, -0.200)
POSES = [POSE_O, POSE_2]
GRIPPER = 0.2555162024919699
# 固定出生：橘窗中心 + 一个角点（sanity），盒窗中心
ORANGE_XYS = [(0.305, -0.36), (0.292, -0.378)]
BOX_XY = (0.465, -0.425)
SETTLE_STEPS = 12
TABLE_TOP_Z = 0.041


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def quat_to_mat_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_mat_wxyz(q):
    w, x, y, z = q
    return quat_to_mat_xyzw((x, y, z, w))


def mat_to_quat_wxyz(R):
    """旋转矩阵 → (w,x,y,z)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q  # w≥0 规范化
    return q / np.linalg.norm(q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe closure camera world pose.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple import mdp as v0  # noqa: E402
    from pick_anything.tasks.pick_apple.camera_env_cfg import (  # noqa: E402
        CameraClosureScanHomeEnvCfg,
    )

    torch.manual_seed(0)
    n = len(ORANGE_XYS)
    cfg = CameraClosureScanHomeEnvCfg()
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
        grip_body_id = int(robot.find_bodies(["gripper"])[0][0])
        print(f"臂关节序: {arm_names}", flush=True)

        # USD 光学直读（闭包手术后应为 focal 9.067 / HFOV≈98）
        stage = env.sim.stage
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/wowrobo_2MP_camera")
        focal = cam_prim.GetAttribute("focalLength").Get()
        h_ap = cam_prim.GetAttribute("horizontalAperture").Get()
        v_ap = cam_prim.GetAttribute("verticalAperture").Get()
        clip = cam_prim.GetAttribute("clippingRange").Get()
        print(f"[USD光学] focal={focal:.4f}mm hAp={h_ap:.3f} vAp={v_ap:.3f} clip={clip}", flush=True)

        # 写死 O 姿态 + 固定橘/盒，沉降
        for _ in range(SETTLE_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)
        env.reset()
        ids = torch.arange(n, device=env.device)
        m = torch.tensor(POSES, device=env.device).float()  # 每 env 一个姿态
        robot.write_joint_position_to_sim(m, env_ids=ids, joint_ids=arm_ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(m), env_ids=ids, joint_ids=arm_ids)
        g = torch.full((n, len(grip_ids)), GRIPPER, device=env.device)
        robot.write_joint_position_to_sim(g, env_ids=ids, joint_ids=grip_ids)
        robot.set_joint_position_target(m, env_ids=ids, joint_ids=arm_ids)
        robot.set_joint_position_target(g, env_ids=ids, joint_ids=grip_ids)

        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(n, 1)
        origins = t(env.scene.env_origins)
        pos_o = torch.zeros(n, 3, device=env.device)
        for i in range(n):
            pos_o[i, 0], pos_o[i, 1] = ORANGE_XYS[i]
        pos_o[:, 2] = v0.FRUIT_SPAWN_Z
        env.scene["orange"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_o + origins, quat], dim=-1), env_ids=ids
        )
        pos_b = torch.zeros(n, 3, device=env.device)
        pos_b[:, 0], pos_b[:, 1], pos_b[:, 2] = BOX_XY[0], BOX_XY[1], v0.BOX_SPAWN_Z
        env.scene["cardboard_box"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_b + origins, quat], dim=-1), env_ids=ids
        )
        for _ in range(SETTLE_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        # ---- dump ----
        pos_w = cam.data.pos_w.cpu().numpy()
        quat_w = cam.data.quat_w_world.cpu().numpy()  # xyzw, +X前/+Z上
        grip_pos = robot.data.body_pos_w.cpu().numpy()[:, grip_body_id]
        grip_quat = robot.data.body_quat_w.cpu().numpy()[:, grip_body_id]  # wxyz
        K = cam.data.intrinsic_matrices.cpu().numpy()
        img_h, img_w = cam.image_shape[0], cam.image_shape[1]
        actual = robot.data.joint_pos[:, arm_ids].cpu().numpy()
        origins_np = origins.cpu().numpy()

        # ROS-cam(x前/y左/z上) → MuJoCo-cam(x右/y上/z后) 的基变换
        C = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

        for i in range(n):
            p = pos_w[i] - origins_np[i]
            gp = grip_pos[i] - origins_np[i]
            R_ros = quat_to_mat_xyzw(quat_w[i])
            R_grip = quat_to_mat_wxyz(grip_quat[i])
            R_cam_mj_world = R_ros @ C
            R_mount = R_grip.T @ R_cam_mj_world
            pos_local = R_grip.T @ (p - gp)
            q_mount = mat_to_quat_wxyz(R_mount)
            fx, fy = K[i][0, 0], K[i][1, 1]
            cx, cy = K[i][0, 2], K[i][1, 2]
            vfov = 2 * math.degrees(math.atan(img_h / (2 * fy)))
            fwd_mj = -R_cam_mj_world[:, 2]
            tilt = math.degrees(math.acos(np.clip(-fwd_mj[2], -1.0, 1.0)))

            def to_px(xy, z):
                c = R_ros.T @ (np.array([xy[0], xy[1], z]) - p)
                if c[0] <= 1e-6:
                    return None
                return (cx - fx * c[1] / c[0], cy - fy * c[2] / c[0])

            o_px = to_px(ORANGE_XYS[i], v0.FRUIT_SPAWN_Z)
            b_px = to_px(BOX_XY, v0.BOX_SPAWN_Z)
            print(f"\n===== env{i}（姿态 {POSES[i]}，橘 {ORANGE_XYS[i]}）=====", flush=True)
            print(f"cam_w pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) quat_ros(xyzw)={np.round(quat_w[i],6).tolist()}",
                  flush=True)
            print(f"grip_w pos=({gp[0]:.4f},{gp[1]:.4f},{gp[2]:.4f}) quat(wxyz)={np.round(grip_quat[i],6).tolist()}",
                  flush=True)
            print(f"相机高(桌上)= {(p[2]-TABLE_TOP_Z)*100:.1f}cm  MuJoCo口径俯角={tilt:.2f}°  "
                  f"fx={fx:.1f} fy={fy:.1f}  vfov={vfov:.2f}°  图 {img_w}x{img_h}", flush=True)
            if o_px:
                print(f"橘投影 u={o_px[0]/img_w:.3f} v={o_px[1]/img_h:.3f}（像素 {o_px[0]:.0f},{o_px[1]:.0f}）",
                      flush=True)
            if b_px:
                print(f"盒投影 u={b_px[0]/img_w:.3f} v={b_px[1]/img_h:.3f}", flush=True)
            print(f"实际关节: {' '.join(f'{v:+.3f}' for v in actual[i])}", flush=True)
            print(f"\n[MJCF 挂载] <camera name=\"wrist_camera\" "
                  f"pos=\"{pos_local[0]:.7f} {pos_local[1]:.7f} {pos_local[2]:.7f}\" "
                  f"quat=\"{q_mount[0]:.7f} {q_mount[1]:.7f} {q_mount[2]:.7f} {q_mount[3]:.7f}\" "
                  f"fovy=\"{vfov:.2f}\"/>", flush=True)

        # ---- 存帧（含「无盒」差分帧：定位盒子的真实像素位置）----
        from PIL import Image
        buf = cam.data.output
        rgb = t(buf["rgb"])[:, :, :, :3].cpu().numpy()
        out_dir = PROJECT_ROOT / "photos" / "closure_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = Image.fromarray(rgb[i].astype(np.uint8))
            img.save(out_dir / f"env{i}_closure_home.png")
            img.resize((img_w * 4, img_h * 4), Image.NEAREST).save(
                out_dir / f"env{i}_closure_home_x4.png")
        # 盒挪走再拍 env0
        pos_p = torch.zeros(1, 3, device=env.device)
        pos_p[:, 0], pos_p[:, 1], pos_p[:, 2] = 0.12, -0.59, v0.BOX_SPAWN_Z
        env.scene["cardboard_box"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_p + origins[:1], quat[:1]], dim=-1),
            env_ids=torch.tensor([0], device=env.device))
        for _ in range(SETTLE_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)
        buf = cam.data.output
        rgb_nb = t(buf["rgb"])[0, :, :, :3].cpu().numpy()
        Image.fromarray(rgb_nb.astype(np.uint8)).save(out_dir / "env0_nobox.png")
        print("[save] env0_nobox.png（盒差分用）", flush=True)
        print(f"[save] {out_dir}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
