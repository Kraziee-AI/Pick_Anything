"""Checkpoint-compatible pure camera PPO smoothing experiments (no distillation)."""
from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg

from .camera_env_cfg import CameraClosureFinetuneEnvCfg
from .agents.camera_ppo_cfg import CameraPPOBaseRunnerCfg, CAMERA_ALGO_CFG


class FilteredRelativeJointPositionAction(RelativeJointPositionAction):
    """Filter once per control step; retain the original relative physics-step targets.

    Raw actions remain the PPO samples for on-policy likelihoods, last_action and
    action_rate. joint_target observes the target actually sent to the actuator.
    The gripper keeps its original action term to preserve closure timing.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError("Action smoothing alpha must be in (0, 1].")
        self._alpha = cfg.alpha
        self._filtered = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions):
        super().process_actions(actions)
        self._filtered.lerp_(self._processed_actions, self._alpha)
        self._processed_actions.copy_(self._filtered)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._filtered.zero_()
        else:
            self._filtered[env_ids] = 0.0


@configclass
class FilteredRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type = FilteredRelativeJointPositionAction
    alpha: float = 0.7


class SmoothCameraPPO(PPO):
    """Warm-start actor/critic but reset optimizer for the changed reward objective.

    Base PPO.load otherwise restores the checkpoint optimizer AND learning rate,
    silently overriding the fine-tuning rate specified in the new config.
    Explicit load_cfg requests are respected.
    """

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": False,
                        "iteration": True, "rnd": False}
        result = super().load(loaded_dict, load_cfg, strict)
        print(f"[smooth-ppo] learning_rate={self.learning_rate:g}; "
              f"checkpoint_optimizer_loaded={bool(load_cfg.get('optimizer'))}", flush=True)
        return result


@configclass
class CameraFilteredEnvCfg(CameraClosureFinetuneEnvCfg):
    """Optional arm EMA shared by training and deployment; gripper unchanged."""

    def __post_init__(self):
        super().__post_init__()
        # Preserve the configuration used for the successful filtered playback.
        self.rewards.action_rate.weight = -0.1
        arm = self.actions.arm_action
        self.actions.arm_action = FilteredRelativeJointPositionActionCfg(
            asset_name=arm.asset_name, joint_names=arm.joint_names,
            scale=arm.scale, offset=arm.offset, clip=arm.clip,
            use_zero_offset=arm.use_zero_offset, alpha=0.7,
        )


@configclass
class CameraSmoothPPOBaseRunnerCfg(CameraPPOBaseRunnerCfg):
    max_iterations = 1000
    save_interval = 100
    experiment_name = "pick_apple_staged_so101_camera_smooth"
    algorithm = CAMERA_ALGO_CFG.replace(
        class_name="pick_anything.tasks.pick_apple.smooth_camera_cfg:SmoothCameraPPO",
        learning_rate=3.0e-5,
    )


@configclass
class CameraSmoothPPORunnerCfg(PresetCfg):
    default = CameraSmoothPPOBaseRunnerCfg()
