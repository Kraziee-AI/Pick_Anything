"""Target-relative training state, resets and grasp/lift rewards.

Privileged geometry belongs to the critic, training masks and task scoring.
The actor receives only RGB, its target mask and 24-D proprioception.
"""

import torch

from isaaclab.utils.math import quat_apply
from pick_anything.grounded_pick.observations import tensor
from pick_anything.tabletop.catalog import OBJECTS, sample_layout
from pick_anything.tasks.pick_apple.mdp import POCKET_OFFSET_B
from pick_anything.tasks.pick_apple.staged_mdp import CLAW_BODY_NAMES


class TargetState:
    def __init__(self, env):
        n, device = env.num_envs, env.device
        self.target_index = torch.zeros(n, dtype=torch.long, device=device)
        self.spawn_center_z = torch.zeros(n, len(env.cfg.object_keys), device=device)
        self.confirmed_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.grasped = torch.zeros(n, dtype=torch.bool, device=device)
        self.wrong_lift_ever = torch.zeros(n, dtype=torch.bool, device=device)
        self.max_rise = torch.zeros(n, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.just_grasped = self.grasped.clone()
        self.just_success = self.success.clone()
        self.aligned = torch.zeros(n, dtype=torch.bool, device=device)
        self.grasp_run = torch.zeros(n, dtype=torch.long, device=device)
        robot = env.scene["robot"]
        self.gripper_body = int(robot.find_bodies(["gripper"])[0][0])
        self.gripper_joint = int(robot.find_joints(["gripper"])[0][0])
        names = list(env.scene.sensors["robot_contact"].body_names)
        self.claw_columns = [i for i, name in enumerate(names) if name in CLAW_BODY_NAMES]
        if not self.claw_columns:
            raise RuntimeError("robot_contact 缺少夹爪刚体。")
        self.previous_distance = torch.zeros(n, device=device)
        self.previous_rise = torch.zeros(n, device=device)
        self.has_previous = torch.zeros(n, dtype=torch.bool, device=device)
        self._step = -1

    @classmethod
    def get(cls, env):
        if not hasattr(env, "open_vocab_state"):
            env.open_vocab_state = cls(env)
        return env.open_vocab_state

    def update(self, env):
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        centers, velocities, forces = [], [], []
        for key in env.cfg.object_keys:
            asset = env.scene[key]
            obj = OBJECTS[key]
            quat = tensor(asset.data.root_quat_w)
            offset = quat.new_tensor([0, 0, obj.height / 2 - obj.origin_to_bottom]).expand(env.num_envs, 3)
            centers.append(tensor(asset.data.root_pos_w) + quat_apply(quat, offset))
            velocities.append(tensor(asset.data.root_lin_vel_w))
            force = tensor(env.scene.sensors[f"{key}_contact"].data.net_forces_w)
            forces.append(force.norm(dim=-1).sum(-1))
        self.centers = torch.stack(centers, 1)
        rows = torch.arange(env.num_envs, device=env.device)
        self.center = self.centers[rows, self.target_index]
        self.velocity = torch.stack(velocities, 1)[rows, self.target_index]
        self.target_force = torch.stack(forces, 1)[rows, self.target_index]
        robot = env.scene["robot"]
        gripper_quat = tensor(robot.data.body_quat_w)[:, self.gripper_body]
        offset = gripper_quat.new_tensor(POCKET_OFFSET_B).expand(env.num_envs, 3)
        self.pocket = tensor(robot.data.body_pos_w)[:, self.gripper_body] + quat_apply(gripper_quat, offset)
        self.gripper_pos = tensor(robot.data.joint_pos)[:, self.gripper_joint]
        gripper_vel = tensor(robot.data.body_lin_vel_w)[:, self.gripper_body]
        contact = tensor(env.scene.sensors["robot_contact"].data.net_forces_w)
        self.claw_force = contact[:, self.claw_columns].norm(dim=-1).amax(-1)
        self.distance = (self.center - self.pocket).norm(dim=-1)
        self.rise = self.center[:, 2] - self.spawn_center_z[rows, self.target_index]
        self.rel_speed = (self.velocity - gripper_vel).norm(dim=-1)
        self.held_now = ((self.distance < 0.045) & (self.claw_force > 2.2)
                         & (self.target_force > 0.1) & (self.gripper_pos > 0.05)
                         & (self.gripper_pos < 1.6) & (self.rel_speed < 0.15))
        # Alignment while open must precede grasp confirmation; early table
        # contacts alone do not disable the approach shaping reward.
        self.aligned |= (self.distance < 0.04) & (self.gripper_pos > 1.3)
        self.grasp_run = torch.where(self.held_now & self.aligned, self.grasp_run + 1, 0)
        self.just_grasped = ~self.grasped & (self.grasp_run >= 3)
        self.grasped |= self.just_grasped
        lifted = self.held_now & self.grasped & (self.rise > env.cfg.lift_height)
        self.confirmed_steps = torch.where(lifted, self.confirmed_steps + 1, 0)
        self.just_success = ~self.success & (self.confirmed_steps >= env.cfg.lift_confirm_steps)
        self.success |= self.just_success
        all_rises = self.centers[..., 2] - self.spawn_center_z
        selected = torch.nn.functional.one_hot(self.target_index, len(env.cfg.object_keys)).bool()
        self.wrong_lift = ((all_rises > 0.04) & ~selected).any(-1)
        self.wrong_lift_ever |= self.wrong_lift
        self.max_rise = torch.maximum(self.max_rise, self.rise)
        finite = torch.isfinite(tensor(robot.data.joint_pos)).all(-1)
        finite &= torch.isfinite(self.centers).all(dim=(1, 2))
        self.invalid = ~finite

    def reset(self, env, ids):
        for name in ("confirmed_steps", "grasped", "success", "just_grasped", "just_success", "has_previous"):
            getattr(self, name)[ids] = 0
        self.aligned[ids], self.grasp_run[ids] = False, 0
        self.wrong_lift_ever[ids], self.max_rise[ids] = False, 0
        self.previous_distance[ids], self.previous_rise[ids] = 0, 0
        if env.cfg.target_key == "random":
            self.target_index[ids] = torch.randint(len(env.cfg.object_keys), (len(ids),), device=env.device)
        else:
            self.target_index[ids] = env.cfg.object_keys.index(env.cfg.target_key)
        self._step = -1


def reset_scene(env, env_ids):
    state = TargetState.get(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state.reset(env, ids)
    origins = tensor(env.scene.env_origins)[ids]
    layouts = [sample_layout(env.cfg.object_keys, int(seed)) for seed in
               torch.randint(2**31 - 1, (len(ids),), device="cpu").tolist()]
    for key in ("cardboard_box", *env.cfg.object_keys):
        pose = torch.tensor([(*layout[key]["position"], *layout[key]["orientation_xyzw"])
                             for layout in layouts], device=env.device)
        pose[:, :3] += origins
        asset = env.scene[key]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        if key != "cardboard_box":
            obj = OBJECTS[key]
            col = env.cfg.object_keys.index(key)
            # Compare lift with the resting centre height, not the extra 4 mm
            # clearance used at spawn or a moving per-step reference.
            state.spawn_center_z[ids, col] = origins[:, 2] + obj.spawn_z - 0.004 + obj.height / 2 - obj.origin_to_bottom
    robot = env.scene["robot"]
    positions = tensor(robot.data.default_joint_pos)[ids].clone()
    positions += torch.empty_like(positions).uniform_(-0.03, 0.03)
    positions[:, state.gripper_joint] = 1.745
    robot.write_joint_position_to_sim_index(position=positions, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions), env_ids=ids)
    robot.set_joint_position_target_index(target=positions, env_ids=ids)


def critic_features(env):
    state = TargetState.get(env)
    state.update(env)
    selected = torch.nn.functional.one_hot(state.target_index, len(env.cfg.object_keys)).float()
    features = torch.cat([
        state.center - state.pocket, state.velocity,
        (state.centers - state.pocket[:, None]).flatten(1), selected,
        state.gripper_pos[:, None], state.claw_force[:, None], state.target_force[:, None],
        state.rise[:, None], state.grasped.float()[:, None], state.aligned.float()[:, None],
        state.grasp_run.float()[:, None], state.confirmed_steps.float()[:, None],
    ], -1)
    return torch.nan_to_num(features, nan=0, posinf=5, neginf=-5).clamp(-20, 20)


def approach_progress(env):
    state = TargetState.get(env)
    state.update(env)
    progress = torch.where(state.has_previous & ~state.grasped,
                           (state.previous_distance - state.distance) / 0.01, 0.0)
    state.previous_distance.copy_(state.distance)
    state.has_previous[:] = True
    return torch.nan_to_num(progress).clamp(-2, 2)


def grasp_reward(env):
    state = TargetState.get(env)
    state.update(env)
    near = (1 - torch.tanh(state.distance / 0.04)) * state.aligned
    close = (1.745 - state.gripper_pos).clamp(0, 1.745) / 1.745
    return torch.nan_to_num(near * close * ~state.grasped)


def lift_progress(env):
    state = TargetState.get(env)
    state.update(env)
    rise = state.rise.clamp(0, env.cfg.lift_height + 0.02)
    result = (rise - state.previous_rise) / 0.01 * state.held_now * state.grasped
    state.previous_rise.copy_(rise)
    return torch.nan_to_num(result).clamp(-2, 2)


def milestone_reward(env):
    state = TargetState.get(env)
    state.update(env)
    return state.just_grasped.float() * 3 + state.just_success.float() * 10


def wrong_object(env):
    state = TargetState.get(env)
    state.update(env)
    return state.wrong_lift.float()


def success(env):
    state = TargetState.get(env)
    state.update(env)
    return state.success


def dropped_or_invalid(env):
    state = TargetState.get(env)
    state.update(env)
    return state.invalid | (state.center[:, 2] < tensor(env.scene.env_origins)[:, 2] + 0.02)
