"""Sanity probe for the Pick-Apple RL env: wiring of state machine, rewards and terminations.

Headless, small env count. Checks, in order:

1. env builds, resets, steps with zero actions; obs/reward shapes are sane;
2. ``sample_placements`` respects rect bounds and separation constraints;
3. displacing the box/orange lights the disturbance penalties;
4. teleporting a "grasped" orange into the box triggers the success
   termination and the place reward;
5. teleporting the orange off the table triggers the failure termination.

Run:
    conda activate isaaclab_61
    cd ~/IsaacLab_6.1/Pick_Anything
    python scripts/probe_pick_env.py --num_envs 4
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick-Apple env sanity probe.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    return parser.parse_args()


def as_tensor(buf):
    return buf.torch if hasattr(buf, "torch") else buf


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

    from pick_anything.tasks.pick_apple import mdp  # noqa: E402
    from pick_anything.tasks.pick_apple.pick_env_cfg import PickAppleEnvCfg  # noqa: E402

    torch.manual_seed(7)

    print("== [1] build env ==")
    cfg = PickAppleEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"   obs group {key}: shape {tuple(value.shape)}")
    else:
        print(f"   obs shape: {tuple(obs.shape)}")

    actions = torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        env.step(actions)
    state0 = mdp.PickState.get(env)
    print(f"   stepped {args.steps} zero actions")
    print(f"   env_origins[0]: {env.scene.env_origins[0].cpu().numpy().round(3)}")
    print(f"   state orange_pos[0] (local): {state0.orange_pos[0].cpu().numpy().round(3)}")
    print(f"   state spawn_box_xy[0]: {state0.spawn_box_xy[0].cpu().numpy().round(3)}")
    print(f"   state box_pos[0] (local): {state0.box_pos[0].cpu().numpy().round(3)}")
    print(f"   episode sums so far:")
    for name, sums in env.reward_manager._episode_sums.items():
        print(f"     {name:>18}: {sums.cpu().numpy().round(3)}")

    print("== [1b] robot contact baselines + claw geometry (post-reset, zero actions) ==")
    contact_sensor = env.scene.sensors["robot_contact"]
    body_names = [str(nm) for nm in contact_sensor.body_names]
    forces_now = as_tensor(contact_sensor.data.net_forces_w).reshape(args.num_envs, -1, 3).norm(dim=-1)
    print(f"   sensor bodies ({len(body_names)}): {body_names}")
    for i, nm in enumerate(body_names):
        print(f"     {nm:>10}: max {forces_now[:, i].max().item():6.3f} N   mean {forces_now[:, i].mean().item():6.3f} N")
    print(f"   gripper joint pos [rad]: {state0.gripper_pos.cpu().numpy().round(3)}")
    print(f"   pocket-jaw dist [m]: {(state0.pocket_pos - state0.jaw_pos).norm(dim=-1).cpu().numpy().round(3)}")
    print(f"   orange-pocket dist [m]: {state0.dist_orange_pocket.cpu().numpy().round(3)}")
    print(f"   pocket z [m]: {state0.pocket_pos[:, 2].cpu().numpy().round(3)} (orange-down gate needs < {mdp.GRASP_LIFT_Z})")
    print(f"   arm_force max {state0.arm_force.max().item():.3f} N | ee_force max {state0.ee_force.max().item():.3f} N")
    # who is the gripper body pressing? force vector + claw heights
    vecs = as_tensor(contact_sensor.data.net_forces_w).reshape(args.num_envs, -1, 3)
    if "gripper" in body_names:
        gv = vecs[:, body_names.index("gripper")]
        print(f"   gripper body force vectors [N]: {gv.cpu().numpy().round(2)}")
    robot_body_names = [str(nm) for nm in env.scene["robot"].data.body_names]
    origins = as_tensor(env.scene.env_origins)
    body_pos = as_tensor(env.scene["robot"].data.body_pos_w)
    for nm in ("gripper", "jaw"):
        if nm in robot_body_names:
            z = (body_pos[:, robot_body_names.index(nm)] - origins)[:, 2]
            print(f"   {nm} body pos z [m]: {z.cpu().numpy().round(3)}")
    print(f"   dist_jaw_orange [m]: {state0.dist_jaw_orange.cpu().numpy().round(3)}")
    print(f"   orange_force [N]: {state0.orange_force.cpu().numpy().round(3)}")

    # gripper-body phantom force hunt: does it depend on the jaw angle?
    robot = env.scene["robot"]
    gripper_col = body_names.index("gripper")
    for cmd in (-0.17, 0.5, 1.3):
        pos = as_tensor(robot.data.joint_pos)[:, state0.gripper_id].clone()
        pos[:] = cmd
        robot.write_joint_position_to_sim_index(position=pos.unsqueeze(-1), joint_ids=[state0.gripper_id])
        for _ in range(10):
            env.step(actions)
        vecs2 = as_tensor(contact_sensor.data.net_forces_w).reshape(args.num_envs, -1, 3)
        mag = vecs2.norm(dim=-1)
        state0._last_step = -1
        print(
            f"   claw cmd {cmd:+.2f} rad -> gripper {mag[:, gripper_col].mean().item():6.2f} N | "
            f"jaw {mag[:, body_names.index('jaw')].mean().item():6.2f} N | "
            f"wrist {mag[:, body_names.index('wrist')].mean().item():6.2f} N"
        )

    # decisive: tilt the shoulder strongly either way; a claw-vs-table resting
    # force must clear when the arm rises (half the envs up, half down)
    lift_id = int(robot.find_joints(["shoulder_lift"])[0][0])
    tilted = as_tensor(robot.data.joint_pos).clone()
    half = max(1, args.num_envs // 2)
    tilted[:half, lift_id] += 0.9
    tilted[half:, lift_id] -= 0.9
    robot.write_joint_position_to_sim_index(position=tilted)
    for _ in range(10):
        env.step(actions)
    vecs3 = as_tensor(contact_sensor.data.net_forces_w).reshape(args.num_envs, -1, 3)
    print(f"   shoulder_lift +0.9 (first {half} envs) / -0.9 (rest) gripper force [N]:")
    print(f"     per-env: {vecs3[:, gripper_col].norm(dim=-1).cpu().numpy().round(2)}")
    state0._last_step = -1

    print("== [2] sample_placements constraints (difficulty 1.0, 20k draws) ==")
    draws = mdp.sample_placements(20000, 1.0, env.device)
    (x0, x1), (y0, y1) = mdp._RECTS["fruit"]
    (bx0, bx1), (by0, by1) = mdp._RECTS["box"]
    bad_box = ((draws["box"][:, 0] < bx0) | (draws["box"][:, 0] > bx1) | (draws["box"][:, 1] < by0) | (draws["box"][:, 1] > by1)).sum()
    bad_fruit_xy = ((draws["orange"][:, 0] < x0) | (draws["orange"][:, 0] > x1) | (draws["orange"][:, 1] < y0) | (draws["orange"][:, 1] > y1)).sum()
    bad_sep = ((draws["orange"] - draws["box"]).norm(dim=-1) < mdp.BOX_FRUIT_CLEAR_M).sum()
    print(f"   box-outside-rect={int(bad_box)} fruit-outside-rect={int(bad_fruit_xy)} separation-violations={int(bad_sep)}")
    for name in ("box", "orange"):
        xy = draws[name].cpu().numpy()
        print(f"   {name:>6}: x [{xy[:,0].min():.3f},{xy[:,0].max():.3f}] y [{xy[:,1].min():.3f},{xy[:,1].max():.3f}]")

    print("== [3] disturbance penalties ==")
    state = mdp.PickState.get(env)
    box = env.scene["cardboard_box"]
    orange = env.scene["orange"]

    def teleport(obj, offset_xy, z=None):
        pos = as_tensor(obj.data.root_pos_w).clone()
        pos[:, :2] += offset_xy
        if z is not None:
            pos[:, 2] = z
        quat = as_tensor(obj.data.root_quat_w).clone()
        pose = torch.cat([pos, quat], dim=-1)
        vel = torch.zeros(args.num_envs, 6, device=env.device)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(vel)
        state._last_step = -1  # force cache refresh

    teleport(box, torch.tensor([0.10, 0.0], device=env.device))
    env.step(actions)
    pre_box_pen = mdp.box_disturbed(env)
    print(f"   box moved +0.10 m: penalty term = {pre_box_pen.cpu().numpy().round(3)} (state disp {state.box_disp.cpu().numpy().round(3)})")

    teleport(orange, torch.tensor([0.10, 0.0], device=env.device))
    env.step(actions)
    print(f"   orange moved +0.10 m: penalty term = {mdp.orange_disturbed(env).cpu().numpy().round(3)}")
    print(f"   after penalties, box local pos: {state.box_pos.cpu().numpy().round(3)}")

    print("== [3b] stage-A pregrasp: orange teleported into the capture pocket, claw forced open ==")
    env.reset()
    state = mdp.PickState.get(env)
    state.update(env)
    robot = env.scene["robot"]
    orange = env.scene["orange"]
    origins = as_tensor(env.scene.env_origins)
    # the pocket lives on the gripper body, which the gripper joint does not
    # move (it swings the jaw), so teleport order is free
    pocket_w = state.pocket_pos + origins
    quat = as_tensor(orange.data.root_quat_w).clone()
    orange.write_root_pose_to_sim(torch.cat([pocket_w, quat], dim=-1))
    orange.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=env.device))
    joint_pos = as_tensor(robot.data.joint_pos)[:, state.gripper_id].clone()
    joint_pos[:] = 1.60  # fully open (0 rad = closed on this USD)
    robot.write_joint_position_to_sim_index(
        position=joint_pos.unsqueeze(-1), joint_ids=[state.gripper_id]
    )
    state._last_step = -1
    env.step(actions)
    state.update(env)
    print(f"   orange-pocket dist after step: {state.dist_orange_pocket.cpu().numpy().round(3)} (gate < {mdp.PREGRASP_POCKET_M})")
    print(f"   gripper pos after step: {state.gripper_pos.cpu().numpy().round(3)} (gate >= {mdp.PREGRASP_MIN_GRIPPER_RAD})")
    print(f"   orange z: {state.orange_pos[:, 2].cpu().numpy().round(3)} (gate < {mdp.GRASP_LIFT_Z})")
    print(f"   pregrasp_now: {state.pregrasp_now.cpu().numpy()}  has_pregrasp: {state.has_pregrasp.cpu().numpy()}")
    print(f"   'pregrasp' episode sum: {env.reward_manager._episode_sums['pregrasp'].cpu().numpy().round(1)} (expect +8 where fired)")

    print("== [3c] collision penalties: box teleported against the claw ==")
    env.reset()
    state = mdp.PickState.get(env)
    state.update(env)
    contact_sensor = env.scene.sensors["robot_contact"]
    body_names = [str(nm) for nm in contact_sensor.body_names]
    jaw_col = body_names.index("jaw")
    jaw_w = state.jaw_pos + origins
    box_quat = as_tensor(box.data.root_quat_w).clone()
    box.write_root_pose_to_sim(torch.cat([jaw_w, box_quat], dim=-1))
    box.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=env.device))
    state._last_step = -1
    worst_arm = 0.0
    worst_jaw = 0.0
    for _ in range(5):
        env.step(actions)
        state.update(env)
        worst_arm = max(worst_arm, float(state.arm_force.max()))
        jaw_mag = as_tensor(contact_sensor.data.net_forces_w).reshape(args.num_envs, -1, 3)[:, jaw_col].norm(dim=-1)
        worst_jaw = max(worst_jaw, float(jaw_mag.max()))
        state._last_step = -1
    print(f"   arm_force peak {worst_arm:.2f} N, jaw column force peak {worst_jaw:.2f} N")
    print(f"   arm_collision term: {mdp.arm_collision(env).cpu().numpy().round(1)}")
    print(f"   ee_bad_contact term: {mdp.ee_bad_contact(env).cpu().numpy().round(1)}")

    env.reset()  # clean slate for the success-path test

    print("== [4] success path: grasped orange teleported into the box ==")
    state.has_grasped[:] = True  # pretend the orange was carried there
    orange = env.scene["orange"]
    box_pos = as_tensor(box.data.root_pos_w).clone()
    in_box_pos = box_pos + torch.tensor([0.0, 0.0, 0.03], device=env.device)
    quat = as_tensor(orange.data.root_quat_w).clone()
    orange.write_root_pose_to_sim(torch.cat([in_box_pos, quat], dim=-1))
    orange.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=env.device))
    state._last_step = -1
    rew_sum = 0.0
    done_hit = False
    for _ in range(10):
        _, rew, _, _, _ = env.step(actions)
        rew_sum += float(rew[0])
        term = env.termination_manager.get_term("orange_in_box")
        state._last_step = -1  # keep caches fresh for get_term reads
        if term.any():
            done_hit = True
            print(f"   orange_in_box termination fired for envs {term.cpu().numpy().nonzero()[0].tolist()}")
            print(f"   place reward this episode: {env.reward_manager._episode_sums['place'].cpu().numpy().round(1)}")
            break
    if not done_hit:
        print(f"   !! success termination did NOT fire; in_box={state.orange_in_box.cpu().numpy()} "
              f"dist_jaw={state.dist_jaw_orange.cpu().numpy().round(3)} speed={state.orange_vel.norm(dim=-1).cpu().numpy().round(3)}")
    print(f"   (reward accumulated during test: {rew_sum:.1f})")

    print("== [5] failure path: orange off the table ==")
    obs, _ = env.reset()
    teleport(orange, torch.tensor([1.5, 0.0], device=env.device), z=0.5)
    fired = False
    for _ in range(10):
        env.step(actions)
        if env.termination_manager.get_term("orange_left_table").any():
            fired = True
            break
    print(f"   orange_left_table termination fired: {fired}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
