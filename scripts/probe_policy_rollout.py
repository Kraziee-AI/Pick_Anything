"""sim2sim 对照探针：在 Isaac 里跑同一 cam_v12 jit 策略 120 步，dump 参考轨迹。

MuJoCo 侧症状：无摩擦后臂 30 步内 lift 掉 0.37rad、相机俯角 33°→59°→29°
剧烈摆动、橘出画、策略卷到限位。需要分辨：运动学不一致（相机随姿态漂移）
还是动力学响应不一致（臂坠落过快）。本探针 dump Isaac 参考轨迹：
* 每 15 步：实际关节角、相机俯角（MuJoCo 口径）、橘 uv（Isaac 内参）、爪角
* 存 3 帧真实 obs 图（0/60/120 步）

与 /tmp/test_nofric2.py 的 MuJoCo 输出逐行对拍。

用法：
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_policy_rollout.py
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

POSE_O = (-0.05, -0.907, 0.190, 1.580, -0.804)
GRIPPER = 0.2555162024919699
ORANGE_XY = (0.305, -0.36)
BOX_XY = (0.465, -0.425)
N_STEPS = 120
POLICY = (PROJECT_ROOT / "logs" / "rsl_rl" / "pick_apple_staged_so101_camera"
          / "2026-10-07_00-53-39_cam_v12_ft_lowlr" / "exported" / "policy.pt")
C = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def quat_to_mat_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac reference rollout of cam_v12 jit.")
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
    cfg = CameraClosureScanHomeEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.play_mode()  # 关闭观测 DR（corruption）——与确定性评估 92.19% 同口径
    cfg.scene.num_envs = 1
    cfg.events.reset_staged.params["forced_stage"] = 0
    env = ManagerBasedRLEnv(cfg)

    try:
        jit = torch.jit.load(str(POLICY), map_location="cpu")
        jit.eval()
        cam = env.scene.sensors["wrist_camera"]
        robot = env.scene["robot"]
        arm_ids, arm_names = robot.find_joints(
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
        quat0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
        origins = t(env.scene.env_origins)
        pos_o = torch.tensor([[ORANGE_XY[0], ORANGE_XY[1], v0.FRUIT_SPAWN_Z]], device=env.device)
        env.scene["orange"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_o + origins, quat0], dim=-1), env_ids=ids)
        pos_b = torch.tensor([[BOX_XY[0], BOX_XY[1], v0.BOX_SPAWN_Z]], device=env.device)
        env.scene["cardboard_box"].write_root_pose_to_sim_index(
            root_pose=torch.cat([pos_b + origins, quat0], dim=-1), env_ids=ids)

        from PIL import Image
        out_dir = PROJECT_ROOT / "photos" / "rollout_probe"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n===== Isaac 参考轨迹（橘 (0.305,-0.36)，与 MuJoCo /tmp/test_nofric2.py 对拍）=====")
        print("t | 橘uv | 俯角 | lift_j | 爪开 | 关节(实际)")
        env.reset()
        for step in range(N_STEPS):
            if step % 15 == 0:
                pos_w = cam.data.pos_w.cpu().numpy()[0] - origins.cpu().numpy()[0]
                quat_w = cam.data.quat_w_world.cpu().numpy()[0]
                R_ros = quat_to_mat_xyzw(quat_w)
                R_mj = R_ros @ C.T
                tilt = math.degrees(math.acos(np.clip(-(-R_mj[:, 2])[2], -1, 1)))
                K = cam.data.intrinsic_matrices.cpu().numpy()[0]
                img_h, img_w = cam.image_shape[0], cam.image_shape[1]
                o_state = env.scene["orange"].data.root_pos_w.cpu().numpy()[0] - origins.cpu().numpy()[0]
                c = R_ros.T @ (o_state - pos_w)
                if c[0] > 1e-6:
                    u = (K[0, 2] - K[0, 0] * c[1] / c[0]) / img_w
                    v = (K[1, 2] - K[1, 1] * c[2] / c[0]) / img_h
                else:
                    u = v = float("nan")
                jp = robot.data.joint_pos.cpu().numpy()[0]
                arm = jp[arm_ids]
                print(f"{step:4d} | u={u:5.2f} v={v:5.2f} | {tilt:4.1f}° | {arm[1]:+.2f} | {jp[grip_ids][0]:.2f}"
                      f" | {np.round(arm, 2).tolist()}", flush=True)
                if step in (0, 60, 120):
                    rgb = t(cam.data.output["rgb"])[0, :, :, :3].cpu().numpy()
                    Image.fromarray(rgb).save(out_dir / f"isaac_t{step:03d}.png")

            # obs 组包（与 MuJoCo policy_wrapper 同契约）→ jit 确定性动作
            # 注意：observation_manager 的 wrist_rgb 已是 CameraImage 预处理过的
            # CHW float[0,1]（camera_mdp.CameraImage），直接用、不要再 permute。
            obs = env.observation_manager.compute()
            if step < 15:
                pro = obs["proprioception"].cpu().numpy()
                img_np = obs["wrist_rgb"].cpu().numpy()
                print(f"OBS t={step}: pro={np.round(pro, 3).tolist()} "
                      f"img_mean={img_np.mean(axis=(0, 2, 3)).round(3).tolist()}", flush=True)
            with torch.inference_mode():
                a = jit(obs["proprioception"].cpu(), [obs["wrist_rgb"].cpu()]).squeeze(0).cpu().numpy()
            if step < 15:
                print(f"  → a={np.round(a, 2).tolist()}", flush=True)
            a = np.clip(a, -1.0, 1.0)
            env.step(torch.tensor(a, dtype=torch.float32, device=env.device).unsqueeze(0))
        print("[save]", out_dir)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
