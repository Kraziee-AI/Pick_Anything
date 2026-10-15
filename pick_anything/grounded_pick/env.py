"""Grounded task with optional first-HOME-episode evaluation before autoreset."""

from __future__ import annotations

import json

import torch

from isaaclab.envs import ManagerBasedRLEnv
from pick_anything.tasks.pick_apple.staged_mdp import StagedPickState

from .progress import GroundedStageProgress
from .observations import tensor


class GroundedPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._evaluation_diagnostics = None
        self._evaluation_seen = None
        self._evaluation_counts = dict.fromkeys(
            ["episodes", "success", "time_out", "staged_orange_dropped", "staged_box_tipped",
             "staged_abnormal_robot", "grasped", "lifted", "over_box_latched", "in_box_latched"], 0,
        )
        super().__init__(cfg, **kwargs)

    def load_managers(self):
        # Scene/physics are initialized here, before observation terms first
        # access StagedPickState.get(self). The attachment is per environment;
        # neither the shared class nor other tasks are monkey-patched.
        state = StagedPickState.get(self)
        state.progress = GroundedStageProgress(self.num_envs, self.device, state, self.cfg.grasp_min_gripper_rad,
                                              self.cfg.grasp_confirm_proximity_m)
        super().load_managers()
        print(f"[GROUNDED] grasp milestone: {state.progress.grasp_semantics}", flush=True)

    def _get_evaluation_diagnostics(self):
        directory = getattr(self.cfg, "evaluation_diagnostics_dir", "")
        if not self.cfg.evaluate_first_episode or not directory:
            return None
        if self._evaluation_diagnostics is None:
            from .diagnostics import EpisodeDiagnostics

            self._evaluation_diagnostics = EpisodeDiagnostics(
                directory, self.num_envs, self.step_dt, self.cfg.evaluation_trace_every,
            )
        return self._evaluation_diagnostics

    def step(self, action):
        # Record ordinary steps after observation computation; terminal steps are
        # captured in _reset_idx before physics/milestones are reset.
        result = super().step(action)
        diagnostic = self._get_evaluation_diagnostics()
        if diagnostic is not None:
            diagnostic.capture(self)
        return result

    def _reset_idx(self, env_ids):
        # At this point terminal state and causes still belong to the completed
        # episode. Counting after super() would instead inspect the reset state.
        if self.cfg.evaluate_first_episode:
            if self.cfg.events.reset_staged.params["forced_stage"] != 0:
                raise ValueError("完整任务评估必须从 HOME 出生；请勿使用 --train_env_cfg。")
            if self._evaluation_seen is None:
                self._evaluation_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = ids[(self.episode_length_buf[ids] > 0) & ~self._evaluation_seen[ids]]
            if ids.numel():
                diagnostic = self._get_evaluation_diagnostics()
                if diagnostic is not None:
                    diagnostic.capture(self, ids, terminal=True)
                    diagnostic.end_episodes(self, ids)
                self._evaluation_seen[ids] = True
                counts = self._evaluation_counts
                counts["episodes"] += int(ids.numel())
                for name in ("success", "time_out", "staged_orange_dropped", "staged_box_tipped", "staged_abnormal_robot"):
                    counts[name] += int(self.termination_manager.get_term(name)[ids].sum())
                state = self.staged_pick_state
                for name in ("grasped", "lifted", "over_box_latched", "in_box_latched"):
                    counts[name] += int(getattr(state.progress, name)[ids].sum())
                if counts["episodes"] == self.num_envs:
                    params = self.cfg.observations.rgb_mask.image.params
                    result = {"episodes": self.num_envs, "birth_policy": "HOME", "mask_backend": params["backend"],
                              "instruction": params["instruction"],
                              "grasp_semantics": getattr(state.progress, "grasp_semantics", "instantaneous_condition_latch"),
                              "rates": {name: value / self.num_envs for name, value in counts.items() if name != "episodes"}}
                    if diagnostic is not None:
                        diagnostic.finish(result, seed=self.cfg.seed)
                    print("GROUNDED_PICK_EVAL_RESULT=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
                    raise SystemExit(0)
        result = super()._reset_idx(env_ids)
        # HOME must not inherit the preceding episode's closed gripper. Restore
        # the existing default reference; reset-bank rows keep their grasp pose.
        state = self.staged_pick_state
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        home_ids = ids[state.birth_stage[ids] == 0]
        if home_ids.numel():
            robot = self.scene["robot"]
            joint_ids = [state.gripper_id]
            position = tensor(robot.data.default_joint_pos)[home_ids][:, joint_ids]
            robot.write_joint_position_to_sim_index(position=position, env_ids=home_ids, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(position), env_ids=home_ids, joint_ids=joint_ids,
            )
            robot.set_joint_position_target_index(target=position, env_ids=home_ids, joint_ids=joint_ids)
        diagnostic = self._get_evaluation_diagnostics()
        if diagnostic is not None:
            diagnostic.start_episodes(self, env_ids)
        return result
