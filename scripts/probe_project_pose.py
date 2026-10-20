"""姿态扫掠 v2（严格几何版）：投影橘子/盒子到相机像素，判定框内外。

2026-10-02 用户实测反馈「还是看不到盒子、相机不是看正下方」——probe_scan_pose
只数了橘子像素，盒子可见性是肉眼误判。本脚本不猜颜色：写姿态后取
cam.data.view_matrix/intrinsic_matrices，把橘子窗口 3 点、盒子窗口 4 点
（含盒顶）投影到像素坐标，直接打印 in/out of frame。用法（GPU2）：

    CUDA_VISIBLE_DEVICES=2 python scripts/probe_project_pose.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

WARMUP_STEPS, SETTLE_STEPS = 10, 12

# pan 扫掠：sh/wr/el 固定在 -0.47/1.10/0.09，找让橘子+盒子组居中的 yaw
PANS = [-0.05]
SH_LIST = [-0.35, -0.45, -0.55, -0.65]
ROLLS = [0.0]
WRS = [1.48, 1.60, 1.658]
ELS = [-0.25, 0.09, 0.39]
ROLL, GRIP = -0.8044013083, 0.2555162024919699

ORANGE_PTS = [(0.33, -0.38), (0.36, -0.36), (0.38, -0.34)]
BOX_PTS = [(0.40, -0.46), (0.44, -0.49), (0.47, -0.50), (0.48, -0.44)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projection-based pose sweep.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def project(world_xyz: torch.Tensor, view: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """world (N,3) -> 像素 (N,2)；view (N,3,4) world->cam（z 前 x 右 y 下）。"""
    n = world_xyz.shape[0]
    cam = torch.einsum("nij,nj->ni", view[:, :, :3], world_xyz) + view[:, :, 3]
    z = cam[:, 2].clamp_min(1e-6)
    uv = torch.stack([K[:, 0, 0] * cam[:, 0] / z + K[:, 0, 2],
                      K[:, 1, 1] * cam[:, 1] / z + K[:, 1, 2]], dim=-1)
    return uv


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """xyzw 四元数 (N,4) -> 旋转矩阵 (N,3,3)。"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return r


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple import mdp as v0  # noqa: E402
    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV4PickAppleEnvCfg,
    )

    torch.manual_seed(0)
    poses = [(p, s, r, w, e) for p in PANS for s in SH_LIST for r in ROLLS for w in WRS for e in ELS]
    n = len(poses)
    cfg = CameraDistillV4PickAppleEnvCfg()
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

        for _ in range(WARMUP_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        env.reset()
        for i, (pan, sh, rl, wr, el) in enumerate(poses):
            ids = torch.tensor([i], device=env.device)
            m = torch.tensor([[pan, sh, el, wr, rl]], device=env.device)
            robot.write_joint_position_to_sim(m, env_ids=ids, joint_ids=arm_ids)
            robot.write_joint_velocity_to_sim(torch.zeros_like(m), env_ids=ids, joint_ids=arm_ids)
            g = torch.full((1, len(grip_ids)), GRIP, device=env.device)
            robot.write_joint_position_to_sim(g, env_ids=ids, joint_ids=grip_ids)
            robot.set_joint_position_target(m, env_ids=ids, joint_ids=arm_ids)
            robot.set_joint_position_target(g, env_ids=ids, joint_ids=grip_ids)
        for _ in range(SETTLE_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        W, H = int(cam.cfg.width), int(cam.cfg.height)
        # CameraData 无 view_matrix——用 pos_w + quat_w_ros 自建（ROS 约定：
        # z 前、x 右、y 下，与投影公式一致）
        pos = t(cam.data.pos_w).clone().float()  # (N,3)
        quat = t(cam.data.quat_w_ros).clone().float()  # (N,4) xyzw
        R = quat_to_rot(quat)  # (N,3,3)
        view = torch.cat([R, (-R @ pos.unsqueeze(-1))], dim=-1)  # (N,3,4)
        K = t(cam.data.intrinsic_matrices).clone()
        origins = t(env.scene.env_origins)

        TABLE = ((-0.145, 0.855), (-0.677, -0.227))
        TABLE_Z = 0.041

        def corners_on_table(env_i: int) -> tuple[int, list]:
            # 画面四角视线投到桌面 z=TABLE_Z，判定交点是否在桌内
            cnt, pts = 0, []
            R_i, K_i = R[env_i], K[env_i]
            for (cu, cv) in ((0, 0), (W, 0), (0, H), (W, H)):
                pix = torch.tensor([[cu - K_i[0, 2], cv - K_i[1, 2], 1.0]], device=env.device)
                d_cam = torch.linalg.solve(K_i, pix.T).T  # (1,3)
                d_w = (R_i.T @ d_cam.T).T  # cam->world 方向
                o_w = pos[env_i]
                denom = d_w[:, 2]
                if abs(float(denom)) < 1e-6:
                    pts.append(None); continue
                t = (TABLE_Z - o_w[2]) / denom
                g = o_w + t * d_w  # 桌面交点(世界)
                gl = (g - origins[env_i])[0]  # env 局部
                ok = TABLE[0][0] <= float(gl[0]) <= TABLE[0][1] and TABLE[1][0] <= float(gl[1]) <= TABLE[1][1]
                cnt += int(ok)
                pts.append((round(float(gl[0]), 2), round(float(gl[1]), 2), ok))
            return cnt, pts

        def uv(env_i: int, xy: tuple[float, float], z: float) -> tuple[float, float]:
            w = torch.tensor([[xy[0], xy[1], z]], device=env.device) + origins[env_i]
            p = project(w, view[env_i : env_i + 1], K[env_i : env_i + 1])
            return round(float(p[0, 0]), 1), round(float(p[0, 1]), 1)

        print(f"\n画面 {W}x{H}；判定=橘子 3 点全入框 且 盒子(含盒顶) ≥3 点入框", flush=True)
        for i, (pan, sh, rl, wr, el) in enumerate(poses):
            o_uvs = [uv(i, p, v0.FRUIT_SPAWN_Z) for p in ORANGE_PTS]
            o_in = sum(1 for (u, v) in o_uvs if 0 <= u <= W and 0 <= v <= H)
            b_uvs = [uv(i, p, v0.BOX_SPAWN_Z + 0.06) for p in BOX_PTS]  # 盒顶(5cm 壁)
            b_in = sum(1 for (u, v) in b_uvs if 0 <= u <= W and 0 <= v <= H)
            c_in, corners = corners_on_table(i)
            # 光轴(z 轴)世界方向与 -z 的夹角 + 相机高度（俯视程度）
            axis_w = R[i][:, 2]
            axis_tilt = float(torch.rad2deg(torch.acos((-axis_w[2]).clamp(-1, 1))))
            height = float(pos[i][2] - 0.041)
            flag = "★" if o_in == 3 and b_in >= 4 and c_in == 4 and axis_tilt < 46 else " "
            if axis_tilt >= 60:
                continue  # 非俯视姿态不打印，聚焦
            print(
                f"{flag} sh={sh:+.2f} wr={wr:.2f} el={el:.2f} | 轴倾{axis_tilt:.0f}° 高{height:.2f}m | 橘 {o_in}/3 盒 {b_in}/4 | 角 {c_in}/4",
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
