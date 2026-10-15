"""Grounded-only grasp milestones; shared training task code stays unchanged."""

from __future__ import annotations

from pick_anything.tasks.pick_apple.camera_progress import CameraStageProgress
from pick_anything.tasks.pick_apple.stages import PREGRASP_POCKET_M


class GroundedStageProgress(CameraStageProgress):
    """Use sustained-contact handoff and once-per-control-step confirmation.

    An already sustained physical grasp may pass PREGRASP without reopening.
    GRASP confirmation still runs before grasped is latched. CameraStageProgress
    supplies the same tested transition logic and per-environment step dedup;
    Fully closed empty-claw angles cannot confirm a new grasp. The final
    physical placement-success conditions and shared camera task stay intact.
    """

    grasp_semantics = "confirmed_nonempty_angle_GRASP_completion_or_reset_bank_seed"

    def __init__(self, num_envs, device, physical_state=None, min_gripper_rad=0.0,
                 grasp_confirm_proximity_m=PREGRASP_POCKET_M):
        super().__init__(num_envs, device)
        self.physical_state = physical_state
        self.min_gripper_rad = min_gripper_rad
        self.grasp_confirm_proximity_m = grasp_confirm_proximity_m

    def update(self, step, cond, confirm):
        if self.physical_state is not None:
            cond = dict(cond)
            distance = (self.physical_state.orange_pos - self.physical_state.pocket_pos).norm(dim=-1)
            cond["GRASP"] = (cond["GRASP"] & (self.physical_state.gripper_pos > self.min_gripper_rad)
                             & (distance < self.grasp_confirm_proximity_m))
        return super().update(step, cond, confirm)
