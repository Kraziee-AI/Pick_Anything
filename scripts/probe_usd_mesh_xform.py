"""sim2sim 视觉网格替换支撑探针：dump USD 可视网格在 Isaac 活舞台的组合世界变换。

背景：onshape MJCF 的腕部网格与 USD 渲染差 ~25%（策略爪位误判根因）。
已用 pxr 从 geometries.usd 提取 USD 网格（/tmp/usd_wrist|follower|jaw.obj），
但 base.usda 层级单位未组合不可直接用。本探针在 Isaac 事实舞台上读
wrist / wrist_roll_follower / moving_jaw 三个 mesh prim 的组合 LocalToWorld，
换算到 USD gripper 系，供 MJCF 摆放（再经两系固定变换进 MJCF gripper 系）。

用法：CUDA_VISIBLE_DEVICES=1 python scripts/probe_usd_mesh_xform.py
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

POSE_O = (-0.05, -0.907, 0.190, 1.580, -0.804)
GRIPPER = 0.2555162024919699


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump composed mesh xforms.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple.camera_env_cfg import (  # noqa: E402
        CameraClosureScanHomeEnvCfg,
    )

    torch.manual_seed(0)
    cfg = CameraClosureScanHomeEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = 1
    cfg.events.reset_staged.params["forced_stage"] = 0
    env = ManagerBasedRLEnv(cfg)

    try:
        robot = env.scene["robot"]
        arm_ids, _ = robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
        grip_ids, _ = robot.find_joints(["gripper"])
        env.reset()
        ids = torch.arange(1, device=env.device)
        m = torch.tensor([POSE_O], device=env.device)
        robot.write_joint_position_to_sim(m, env_ids=ids, joint_ids=arm_ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(m), env_ids=ids, joint_ids=arm_ids)
        g = torch.full((1, len(grip_ids)), GRIPPER, device=env.device)
        robot.write_joint_position_to_sim(g, env_ids=ids, joint_ids=grip_ids)
        robot.set_joint_position_target(m, env_ids=ids, joint_ids=arm_ids)
        robot.set_joint_position_target(g, env_ids=ids, joint_ids=grip_ids)
        for _ in range(4):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        from pxr import Usd, UsdGeom  # noqa: E402

        stage: Usd.Stage = env.sim.stage
        origins = env.scene.env_origins.cpu().numpy()[0]

        grip_body = int(robot.find_bodies(["gripper"])[0][0])
        gp_isaac = robot.data.body_pos_w.cpu().numpy()[0, grip_body] - origins
        gq_isaac = robot.data.body_quat_w.cpu().numpy()[0, grip_body]  # wxyz

        def quat_to_mat(q):
            w, x, y, z = q
            return np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])

        R_usd = quat_to_mat(gq_isaac)
        print(f"\n[USD gripper 世界] pos={gp_isaac.round(6).tolist()} quat(wxyz)={gq_isaac.round(6).tolist()}",
              flush=True)

        for part in ["wrist", "wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"]:
            # 只取 /World/envs/env_0/ 下的实例（场景里有同名 prim 的多个层级——
            # 2026-10-07 全局名字搜索拿错实例，网格飘到碰撞爪 13cm 外）
            prim = None
            for p in stage.Traverse():
                if p.GetName() == part and str(p.GetPath()).startswith("/World/envs/env_0/"):
                    prim = p
                    break
            # 列出全部同名候选供人工核对
            cands = [str(p.GetPath()) for p in stage.Traverse() if p.GetName() == part]
            print(f"\n[{part}] 选中={prim.GetPath() if prim else None}")
            print(f"  候选: {cands[:6]}", flush=True)
            if prim is None:
                continue
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            t = np.array(xf.ExtractTranslation()) - origins
            q = xf.ExtractRotation().GetQuaternion()
            qw, (qx, qy, qz) = q.GetReal(), q.GetImaginary()
            R_part = quat_to_mat((qw, qx, qy, qz))
            rel_t = R_usd.T @ (t - gp_isaac)
            rel_R = R_usd.T @ R_part
            print(f"  世界: t={t.round(6).tolist()} q(wxyz)=({qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f})")
            print(f"  USD gripper 系: t={rel_t.round(6).tolist()}")
            print(f"  旋转矩阵(USD gripper 系):\n{np.round(rel_R, 6)}")
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
