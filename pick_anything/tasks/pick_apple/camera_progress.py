"""Camera stage transitions accept an already sustained physical grasp."""

import torch

from .stages import ConfirmedGraspStageProgress, GRASP_CONFIRM_STEPS, PickStage


class CameraStageProgress(ConfirmedGraspStageProgress):
    """Keep open-gripper alignment, or recognize an already held object.

    Reopening a sustained grasp just to pass PREGRASP would lose the object.
    Three consecutive existing GRASP conditions can unlock that transition;
    the regular GRASP confirmation still runs before publishing grasped.
    No distance, angle, contact-force, lift or final-success threshold changes.
    """

    grasp_semantics = "confirmed_GRASP_completion_or_reset_bank_seed_with_sustained_pregrasp_contact"

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.pregrasp_contact_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_update_step = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None, *, stage_seed=None):
        super().reset(env_ids, stage_seed=stage_seed)
        self.pregrasp_contact_count[slice(None) if env_ids is None else env_ids] = 0
        self.last_update_step[slice(None) if env_ids is None else env_ids] = -1

    def update(self, step, cond, confirm):
        fresh = step != self.last_update_step
        active = (self.stage == int(PickStage.PREGRASP)) & ~self.first_step & fresh
        raw = cond.get("GRASP", torch.zeros_like(self.grasped))
        self.pregrasp_contact_count.copy_(torch.where(fresh, torch.where(
            active & raw, self.pregrasp_contact_count + 1, torch.zeros_like(self.pregrasp_contact_count),
        ), self.pregrasp_contact_count))
        held = active & (self.pregrasp_contact_count >= confirm.get("GRASP", GRASP_CONFIRM_STEPS))
        conditions = dict(cond)
        conditions["PREGRASP"] = cond.get("PREGRASP", torch.zeros_like(self.grasped)) | held
        result = super().update(step, conditions, confirm, update_mask=fresh)
        self.last_update_step.copy_(step)
        return result
