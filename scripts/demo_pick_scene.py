"""Demo: table + cube + SO101 arm in Isaac Sim.

Run (GUI):
    conda activate isaaclab_61
    cd ~/IsaacLab_6.1/Pick_Anything
    python scripts/demo_pick_scene.py --visualizer kit

Run (headless smoke test):
    python scripts/demo_pick_scene.py --steps 900
"""

import argparse
import os
import sys
from pathlib import Path

# Make the project package importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# First Kit launch requires EULA acceptance.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick_Anything demo scene.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--steps", type=int, default=0, help="Exit after N steps (0 = run until window closes).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for fruit placement (default: random each run).")
    return parser.parse_args()


def as_tensor(buffer):
    """IsaacLab 3.0 data buffers may be lazy ProxyArray; normalize to torch.Tensor."""
    return buffer.torch if hasattr(buffer, "torch") else buffer


def main() -> None:
    args_cli = parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics import PhysxCfg

    from pick_anything.env_cfg import BOX_SPAWN_Z, FRUIT_SPAWN_Z, PickSceneCfg, sample_object_positions

    sim_cfg = SimulationCfg(dt=1 / 120, physics=PhysxCfg(), device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(PickSceneCfg(num_envs=1))
    sim.set_camera_view(eye=[1.5, -1.8, 1.2], target=[0.35, -0.4, 0.1])

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    objects = {
        "apple": scene["apple"],
        "orange": scene["orange"],
        "box": scene["cardboard_box"],
    }
    spawn_z = {"apple": FRUIT_SPAWN_Z, "orange": FRUIT_SPAWN_Z, "box": BOX_SPAWN_Z}

    # Randomize placement within arm reach (annulus sector clipped to table).
    import random

    import torch

    rng = random.Random(args_cli.seed)
    placements = sample_object_positions(rng)
    for name, obj in objects.items():
        x, y = placements[name]
        quat = as_tensor(obj.data.root_quat_w).clone()  # keep spawn orientation
        pos = torch.tensor([[x, y, spawn_z[name]]], device=obj.device, dtype=quat.dtype)
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        print(f"{name} spawn: ({x:.3f}, {y:.3f})")

    # Resolve SO101 joint indices by name (order is not guaranteed).
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    arm_ids, _ = robot.find_joints(arm_names)
    gripper_ids, _ = robot.find_joints(["gripper"])
    default_joint_pos = as_tensor(robot.data.default_joint_pos)
    default_arm = default_joint_pos[:, arm_ids]
    default_gripper = default_joint_pos[:, gripper_ids]

    print(f"robot joints: {robot.joint_names}")
    print(f"robot base pos: {as_tensor(robot.data.root_pos_w)[0].cpu().numpy()}")
    for name, obj in objects.items():
        print(f"{name} init pos: {as_tensor(obj.data.root_pos_w)[0].cpu().numpy()}")

    # Slow sinusoidal wave to verify the actuators are alive.
    tick = 0
    while simulation_app.is_running():
        if args_cli.steps and tick >= args_cli.steps:
            break

        t = tick * sim_cfg.dt
        wave = 0.3 * torch.sin(torch.tensor(2.0 * t, device=robot.device))
        targets_arm = default_arm.clone()
        targets_arm[:, 1] += wave          # shoulder_lift
        targets_arm[:, 2] -= 0.5 * wave    # elbow_flex
        robot.set_joint_position_target(targets_arm, joint_ids=arm_ids)
        robot.set_joint_position_target(default_gripper, joint_ids=gripper_ids)
        robot.write_data_to_sim()

        sim.step()
        scene.update(sim_cfg.dt)

        if tick % 300 == 0:
            joint_pos = as_tensor(robot.data.joint_pos)[0].cpu().numpy().round(3)
            obj_pos = {
                n: as_tensor(o.data.root_pos_w)[0].cpu().numpy().round(3).tolist()
                for n, o in objects.items()
            }
            print(f"t={t:6.2f}s  joints={list(joint_pos)}  objects={obj_pos}")
        tick += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
