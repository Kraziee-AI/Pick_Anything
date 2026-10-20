"""sim2sim 动力学对拍：固定动作序列的关节阶跃响应（无策略、无 obs 反馈）。

MuJoCo 侧症状：同策略同时序下 MuJoCo 肘 15 步 +0.99 rad（Isaac +0.34），
怀疑 Newton 与 MuJoCo-CPU 的关节动力学不一致（friction/damping 语义差）。
本探针在 Isaac 里发固定动作 [0,0,+1,0,0,+1] × 45 步，每 5 步 dump 关节角，
与 /tmp/ab_mujoco_step.py 逐行对拍（纯物理，排除感知耦合）。

用法：
  CUDA_VISIBLE_DEVICES=1 python scripts/probe_joint_response.py
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
N_STEPS = 45
ACTION = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)  # elbow_flex +1, gripper 开


def t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-action joint step response probe.")
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
        for _ in range(10):  # 让写入生效
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

        a = torch.tensor([ACTION], dtype=torch.float32, device=env.device)
        print("\n===== Isaac 阶跃响应（动作 [pan,lift,elbow,wf,roll,grip]=[0,0,1,0,0,1]）=====")
        print("t | pan | lift | elbow | wf | roll | grip")
        for step in range(N_STEPS):
            if step % 5 == 0:
                jp = robot.data.joint_pos.cpu().numpy()[0]
                print(f"{step:3d} | " + " | ".join(f"{v:+.3f}" for v in jp[[*arm_ids, *grip_ids]]),
                      flush=True)
            env.step(a)
        jp = robot.data.joint_pos.cpu().numpy()[0]
        print(f"{N_STEPS:3d} | " + " | ".join(f"{v:+.3f}" for v in jp[[*arm_ids, *grip_ids]]))
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
