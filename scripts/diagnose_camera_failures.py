"""Opt-in first-HOME-episode trajectories for camera PPO, including pre-reset state.

Use eval_pick_home.py with --external_callback scripts.diagnose_camera_failures.install.
PICK_DIAG_ENVS=16; PICK_DIAG_DIR=outputs/camera_ppo/failure_diagnosis/seed42.
Measurements never modify actions, physical thresholds or task decisions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from pick_anything.grounded_pick.diagnostics import EpisodeDiagnostics
from pick_anything.tasks.pick_apple import mdp as v0
from pick_anything.tasks.pick_apple.stages import (
    PLACE_INSIDE_XY_M, PLACE_INSIDE_Z_MIN_M, PLACE_INSIDE_Z_MAX_M,
    RELEASE_SETTLED_SPEED_MPS, STAGE_NAMES,
)


class CameraFailureDiagnostics(EpisodeDiagnostics):
    def capture(self, env, env_ids=None, terminal=False):
        super().capture(env, env_ids, terminal)
        state = env.staged_pick_state
        rows = torch.cat([
            env.episode_length_buf.float()[:, None], state.gripper_pos[:, None],
            state.orange_local, state.orange_vel.norm(dim=-1)[:, None],
            state.progress.stable_count.float()[:, None], state.box_pos,
        ], dim=-1).detach().cpu().tolist()
        ids = range(env.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
        frames = []
        for i in ids:
            if i in self.done or i not in self.episodes:
                continue
            ep = self.episodes[i]
            row = rows[i]
            step = int(row[0])
            if step <= 0 or step == ep.get("_extra_last_step", -1):
                continue
            ep["_extra_last_step"] = step
            record = ep["terminal_snapshot"]
            inside = (abs(row[2]) < PLACE_INSIDE_XY_M and abs(row[3]) < PLACE_INSIDE_XY_M
                      and PLACE_INSIDE_Z_MIN_M < row[4] < PLACE_INSIDE_Z_MAX_M)
            clear = record["pocket_distance_m"] > state.release_clearance_m
            settled = row[5] < RELEASE_SETTLED_SPEED_MPS
            record.update(orange_local_m=row[2:5], orange_speed_mps=row[5],
                          stable_count=int(row[6]), box_pos_m=row[7:10],
                          release_inside=inside, release_clear=clear, release_settled=settled)
            ep["max_gripper_rad"] = max(ep.get("max_gripper_rad", -10), row[1])
            ep["min_gripper_rad"] = min(ep.get("min_gripper_rad", 10), row[1])
            for name, flag in (("release_inside_steps", inside), ("release_clear_steps", inside and clear),
                               ("release_ready_steps", inside and clear and settled),
                               ("pregrasp_open_steps", record["pregrasp_open"])):
                ep[name] = ep.get(name, 0) + int(flag)
            ep["max_release_confirmation"] = max(ep.get("max_release_confirmation", 0), int(row[6]))
            if record["pregrasp_near"]:
                ep["max_gripper_when_near_rad"] = max(ep.get("max_gripper_when_near_rad", -10), row[1])
            if terminal or step in (1, 30, 90, 300, 600, 1200, 1800, 2400):
                frames.append((i, step, "terminal_previous_obs" if terminal else "observation"))
        if frames:
            from PIL import Image
            rgb = env.scene.sensors["wrist_camera"].data.output["rgb"][..., :3].detach().cpu().numpy()
            directory = self.directory / "frames"
            directory.mkdir(exist_ok=True)
            for i, step, label in frames:
                Image.fromarray(rgb[i]).save(directory / f"env{i:02d}_step{step:04d}_{label}.png")
        if terminal:
            for i in ids:
                if i not in self.done and i in self.episodes:
                    self.episodes[i]["reward_totals"] = {
                        name: float(values[i]) for name, values in env.reward_manager._episode_sums.items()
                    }
        # Separate extras retain release geometry for sampled frames, without
        # changing the existing shared diagnostic trace schema.
        if not hasattr(self, "extra_trace"):
            self.extra_trace = (self.directory / "release_trace.jsonl").open("w")
        for i in ids:
            if i not in self.done and i in self.episodes:
                ep = self.episodes[i]
                rec = ep.get("terminal_snapshot")
                if rec and (terminal or rec["step"] % self.trace_every == 0):
                    self.extra_trace.write(json.dumps(rec, allow_nan=False) + "\n")

    def finish(self, result, seed):
        if hasattr(self, "extra_trace"):
            self.extra_trace.close()
        super().finish(result, seed)
        path = self.directory / "summary.json"
        report = json.loads(path.read_text())
        categories = {}
        for ep in report["episodes"]:
            record = ep["terminal_snapshot"]
            if "success" in ep["termination"]:
                cause = "success"
            elif "staged_box_tipped" in ep["termination"]:
                cause = "box_tipped"
            elif ep["max_orange_height_m"] <= 0.105:
                cause = "not_lifted"
            elif not record["release_inside"]:
                cause = "lifted_but_not_inside_box_at_timeout"
            elif not record["release_clear"]:
                cause = "inside_box_but_gripper_not_clear"
            elif not record["release_settled"]:
                cause = "inside_box_but_not_settled"
            else:
                cause = "release_confirmation_insufficient"
            ep["failure_category"] = cause
            categories[cause] = categories.get(cause, 0) + 1
        report["failure_categories"] = categories
        report["notes"] = [
            "Camera RGB policy: mask measurements are unused and not visibility evidence.",
            "Per-step statistics use first episodes only; terminal state is captured before reset.",
            "Terminal image is the preceding observation; terminal geometry is the current physics state.",
            "Failure categories describe measured terminal conditions, not proven causal attribution.",
            "Task actions, thresholds, rewards and reset distribution were not changed by diagnostics.",
        ]
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print("CAMERA_FAILURE_DIAGNOSTICS=" + str(path), flush=True)


def install():
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from scripts import eval_pick_home

    eval_pick_home.EVALUATION_EPISODES = int(os.environ.get("PICK_DIAG_ENVS", "16"))
    argv = eval_pick_home.install_episode_counter()
    original_reset, original_step = ManagerBasedRLEnv._reset_idx, RslRlVecEnvWrapper.step
    diagnostic = None

    def reset(env, env_ids):
        nonlocal diagnostic
        if diagnostic is None:
            diagnostic = CameraFailureDiagnostics(
                os.environ.get("PICK_DIAG_DIR", "outputs/camera_ppo/failure_diagnosis/seed42"),
                env.num_envs, env.step_dt, trace_every=10,
            )
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        terminal_ids = ids[env.episode_length_buf[ids] > 0]
        terminal_ids = torch.tensor(
            [i for i in terminal_ids.tolist() if i not in diagnostic.done], device=env.device, dtype=torch.long,
        )
        if terminal_ids.numel():
            diagnostic.capture(env, terminal_ids, terminal=True)
            diagnostic.end_episodes(env, terminal_ids)
        output = original_reset(env, env_ids)
        diagnostic.start_episodes(env, env_ids)
        return output

    def step(wrapper, actions):
        try:
            output = original_step(wrapper, actions)
        except SystemExit:
            path = os.environ.get("PICK_EVAL_RESULT_PATH")
            result = json.loads(Path(path).read_text()) if path else {"episodes": len(diagnostic.done)}
            diagnostic.finish(result, wrapper.unwrapped.cfg.seed)
            raise
        diagnostic.capture(wrapper.unwrapped)
        return output

    ManagerBasedRLEnv._reset_idx, RslRlVecEnvWrapper.step = reset, step
    return argv
