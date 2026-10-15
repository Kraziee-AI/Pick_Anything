"""GT-mask PPO shaping for the observed premature-closing failure."""

import torch
from isaaclab.managers import ManagerTermBase
from pick_anything.tasks.pick_apple.staged_mdp import StagedPickState, PREGRASP_MIN_GRIPPER_RAD
from pick_anything.tasks.pick_apple.stages import PickStage
from pick_anything.tasks.pick_apple.camera_mdp import CameraApproachProgress


class GroundedApproachProgress(CameraApproachProgress):
    """Retain approach/recovery gradients until the object is physically lifted.

    A sustained table contact is not a pickup. The previous grasp-latch gate
    disabled recovery even when the claw left the orange on the table.
    """

    def __call__(self, env, scale: float = 0.01, d_max: float = 0.45):
        state = StagedPickState.get(env)
        state.update(env)
        dist = (state.orange_pos - state.pocket_pos).norm(dim=-1).clamp(max=d_max)
        pay = torch.where(self._has_prev & ~state.progress.lifted,
                          (self._prev_dist - dist) / scale, 0.0)
        self._prev_dist.copy_(dist)
        self._has_prev.fill_(True)
        return torch.nan_to_num(pay, nan=0.0, posinf=0.0, neginf=0.0)


def empty_gripper_before_lift(env):
    """Penalty for closing to the empty-claw range before a physical pickup."""
    state = StagedPickState.get(env)
    state.update(env)
    return ((state.progress.stage <= int(PickStage.LIFT)) & ~state.progress.lifted
            & (state.gripper_pos <= env.cfg.grasp_min_gripper_rad)).float()


class PregraspOpeningProgress(ManagerTermBase):
    """Pay bounded opening progress before GRASP, with symmetric closing cost.

    The approach potential alone cannot distinguish an open from a closed claw.
    This adds a gripper-angle potential capped at the existing open-alignment
    threshold. Holding still pays zero; opening and reclosing in the same stage
    has zero net undiscounted shaping reward. No reward is paid on reset or in
    GRASP and later stages, where closing must remain possible.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        self._initialized[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, angle_scale: float = 0.2):
        state = StagedPickState.get(env)
        state.update(env)
        opening = state.gripper_pos.clamp(0.0, PREGRASP_MIN_GRIPPER_RAD)
        active = (state.progress.stage < int(PickStage.GRASP)) & ~state.progress.grasped
        reward = torch.where(self._initialized & active, (opening - self._previous) / angle_scale, 0.0)
        self._previous.copy_(opening)
        self._initialized.fill_(True)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
