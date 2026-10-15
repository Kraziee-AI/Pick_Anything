"""Read-only, opt-in first-episode measurements; no changes to task decisions."""

from __future__ import annotations

import json
import math

import torch

from .grounding import workspace_output
from .observations import tensor
from pick_anything.tasks.pick_apple import mdp as v0
from pick_anything.tasks.pick_apple.staged_mdp import GRASP_CLAW_FORCE_N, PREGRASP_MIN_GRIPPER_RAD
from pick_anything.tasks.pick_apple.stages import (
    GRASP_MIN_GRIPPER_RAD, GRASP_MAX_GRIPPER_RAD, GRASP_PROXIMITY_M,
    GRASP_CONFIRM_STEPS, LIFT_Z_M, PREGRASP_POCKET_M, STAGE_NAMES,
)


class EpisodeDiagnostics:
    """Statistics at every control step; disk trace is downsampled plus events.

    `grasp_raw` reproduces the instantaneous physical condition, not the
    permanent `grasped` flag (whose semantics are tagged in the result).
    Statistics are recorded before autoreset, and
    only the first episode per environment is measured.
    """

    def __init__(self, directory, num_envs, step_dt, trace_every=5):
        self.directory = workspace_output(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = (self.directory / "trace.jsonl").open("w")
        self.num_envs, self.step_dt, self.trace_every = num_envs, step_dt, trace_every
        self.episodes = {}
        self.done = set()
        self.grasp_min_gripper_rad = GRASP_MIN_GRIPPER_RAD
        self.grasp_confirm_proximity_m = GRASP_PROXIMITY_M

    def start_episodes(self, env, env_ids):
        ids = torch.as_tensor(env_ids).cpu().tolist()
        origins = tensor(env.scene.env_origins)
        positions = (tensor(env.scene["orange"].data.root_pos_w) - origins).cpu().tolist()
        for i in ids:
            if i in self.done or i in self.episodes:
                continue
            self.episodes[i] = {
                "env_id": i, "initial_orange_pos_m": positions[i], "steps": 0,
                "stage_steps": [0] * len(STAGE_NAMES), "max_stage": 0,
                "min_pocket_distance_m": math.inf, "max_orange_height_m": positions[i][2],
                "max_height_gain_m": 0.0, "max_grasp_raw_run_steps": 0,
                "grasp_raw_steps": 0, "near_pregrasp_steps": 0, "pregrasp_ready_steps": 0,
                "near_but_not_open_steps": 0, "height_above_lift_steps": 0,
                "lift_ready_steps": 0, "mask_nonempty_steps": 0,
                "approach_disabled_steps": 0, "first_grasp_latch": None,
                "first_stage_entry_s": {"HOME": 0.0}, "last_step": -1,
                "_grasp_run": 0, "_previous_event": None,
            }

    def capture(self, env, env_ids=None, terminal=False):
        state = env.staged_pick_state
        n = env.num_envs
        progress = state.progress
        self.grasp_min_gripper_rad = max(GRASP_MIN_GRIPPER_RAD, getattr(progress, "min_gripper_rad", GRASP_MIN_GRIPPER_RAD))
        self.grasp_confirm_proximity_m = getattr(progress, "grasp_confirm_proximity_m", GRASP_PROXIMITY_M)
        diagnostic = getattr(env, "grounding_diagnostics", None)
        if diagnostic is None:
            boxes = torch.zeros(n, 4, device=env.device)
            detected = torch.zeros(n, device=env.device)
        else:
            boxes, detected = diagnostic["boxes"], diagnostic["detected"]
        robot = env.scene["robot"]
        # One device-to-host copy per step rather than synchronizing per metric.
        columns = [env.episode_length_buf.float()[:, None], progress.stage.float()[:, None],
                   state.orange_pos, state.pocket_pos, state.orange_vel,
                   state.claw_force[:, None], state.orange_force[:, None], state.gripper_pos[:, None],
                   progress.confirm_count[:, 2:4].float(),
                   torch.stack([progress.grasped, progress.lifted, progress.over_box_latched,
                                progress.in_box_latched], -1).float(),
                   boxes, detected.float()[:, None],
                   tensor(robot.data.joint_pos), tensor(robot.data.joint_pos_target),
                   env.action_manager.action]
        rows = torch.cat(columns, -1).detach().cpu().tolist()
        ids = range(n) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
        camera = env.scene.sensors["wrist_camera"]
        height, width = camera.cfg.height, camera.cfg.width
        for i in ids:
            if i in self.done or i not in self.episodes:
                continue
            row = rows[i]
            ep = self.episodes[i]
            step, stage = int(row[0]), int(row[1])
            if step <= 0 or step == ep["last_step"]:
                continue
            ep["last_step"] = step
            op, pp = row[2:5], row[5:8]
            distance = math.dist(op, pp)
            claw_force, orange_force, grip = row[11:14]
            grasped, lifted, over_box, in_box = map(bool, row[16:20])
            x0, y0, x1, y1 = row[20:24]
            mask_area = max(0, min(width, x1) - max(0, x0)) * max(0, min(height, y1) - max(0, y0))
            mask_nonempty = bool(row[24]) and mask_area > 0
            raw = (claw_force > GRASP_CLAW_FORCE_N and orange_force > 0.15
                   and distance < self.grasp_confirm_proximity_m and self.grasp_min_gripper_rad < grip < GRASP_MAX_GRIPPER_RAD)
            near = distance < PREGRASP_POCKET_M
            open_enough = grip > PREGRASP_MIN_GRIPPER_RAD
            pregrasp = near and open_enough and op[2] < v0.GRASP_LIFT_Z and not grasped
            lift_ready = op[2] > LIFT_Z_M and distance < GRASP_PROXIMITY_M + 0.01 and orange_force > 0.2
            ep["steps"] += 1
            ep["stage_steps"][stage] += 1
            ep["max_stage"] = max(ep["max_stage"], stage)
            ep["min_pocket_distance_m"] = min(ep["min_pocket_distance_m"], distance)
            ep["max_orange_height_m"] = max(ep["max_orange_height_m"], op[2])
            ep["max_height_gain_m"] = max(ep["max_height_gain_m"], op[2] - ep["initial_orange_pos_m"][2])
            ep["_grasp_run"] = ep["_grasp_run"] + 1 if raw else 0
            ep["max_grasp_raw_run_steps"] = max(ep["max_grasp_raw_run_steps"], ep["_grasp_run"])
            for key, value in (("grasp_raw_steps", raw), ("near_pregrasp_steps", near),
                               ("pregrasp_ready_steps", pregrasp), ("near_but_not_open_steps", near and not open_enough),
                               ("height_above_lift_steps", op[2] > LIFT_Z_M), ("lift_ready_steps", lift_ready),
                               ("mask_nonempty_steps", mask_nonempty), ("approach_disabled_steps", grasped)):
                ep[key] += int(value)
            elapsed = step * self.step_dt
            ep["first_stage_entry_s"].setdefault(STAGE_NAMES[stage], elapsed)
            if grasped and ep["first_grasp_latch"] is None:
                ep["first_grasp_latch"] = {"time_s": elapsed, "stage": STAGE_NAMES[stage],
                                           "pocket_distance_m": distance, "gripper_rad": grip,
                                           "claw_force_n": claw_force, "orange_force_n": orange_force,
                                           "orange_height_m": op[2], "raw_run_steps": ep["_grasp_run"]}
            event = (stage, grasped, lifted, over_box, in_box, raw, mask_nonempty)
            record = {
                "env_id": i, "step": step, "time_s": elapsed, "stage": STAGE_NAMES[stage],
                "orange_pos_m": op, "pocket_pos_m": pp, "orange_vel_mps": row[8:11],
                "pocket_distance_m": distance, "gripper_rad": grip,
                "claw_force_n": claw_force, "orange_force_n": orange_force,
                "grasp_raw": raw, "grasp_raw_run_steps": ep["_grasp_run"],
                "grasp_confirm_steps": int(row[14]), "lift_confirm_steps": int(row[15]),
                "grasped_latched": grasped, "lifted_latched": lifted,
                "pregrasp_near": near, "pregrasp_open": open_enough,
                "pregrasp_ready": pregrasp, "lift_ready": lift_ready,
                "mask_nonempty": mask_nonempty, "mask_box_xyxy": row[20:24],
                "mask_timing": "preceding_observation" if terminal else "current_observation",
                "joint_pos_rad": row[25:31], "joint_target_rad": row[31:37], "actions": row[37:43],
            }
            ep["terminal_snapshot"] = record
            if terminal or step % self.trace_every == 0 or event != ep["_previous_event"]:
                self.trace.write(json.dumps(record, allow_nan=False) + "\n")
            ep["_previous_event"] = event

    def end_episodes(self, env, env_ids):
        ids = torch.as_tensor(env_ids).cpu().tolist()
        causes = {name: env.termination_manager.get_term(name).cpu().tolist() for name in
                  ("success", "time_out", "staged_orange_dropped", "staged_box_tipped", "staged_abnormal_robot")}
        for i in ids:
            self.episodes[i]["termination"] = [name for name, values in causes.items() if values[i]]
            self.done.add(i)

    def finish(self, result, seed):
        self.trace.close()
        episodes = []
        for i in sorted(self.done):
            ep = {key: value for key, value in self.episodes[i].items() if not key.startswith("_")}
            ep["stage_seconds"] = {name: count * self.step_dt for name, count in zip(STAGE_NAMES, ep.pop("stage_steps"))}
            ep["max_stage"] = STAGE_NAMES[ep["max_stage"]]
            ep["max_grasp_raw_run_s"] = ep["max_grasp_raw_run_steps"] * self.step_dt
            ep["grasp_continuous_3_steps"] = ep["max_grasp_raw_run_steps"] >= GRASP_CONFIRM_STEPS
            first = ep["first_grasp_latch"]
            ep["grasp_latched_before_grasp_stage"] = first is not None and STAGE_NAMES.index(first["stage"]) < 2
            ep["mask_nonempty_fraction"] = ep["mask_nonempty_steps"] / max(ep["steps"], 1)
            episodes.append(ep)
        report = {
            "evaluation": result, "seed": seed, "control_dt_s": self.step_dt,
            "thresholds": {"pregrasp_distance_m": PREGRASP_POCKET_M, "pregrasp_open_rad": PREGRASP_MIN_GRIPPER_RAD,
                           "grasp_distance_m": self.grasp_confirm_proximity_m, "grasp_gripper_rad": [self.grasp_min_gripper_rad, GRASP_MAX_GRIPPER_RAD],
                           "grasp_claw_force_n": GRASP_CLAW_FORCE_N, "grasp_confirm_steps": GRASP_CONFIRM_STEPS,
                           "lift_height_m": LIFT_Z_M},
            "episodes": episodes,
            "notes": ["All maxima, durations and runs use every control step; trace is sampled plus event frames.",
                      "grasped semantics are given by evaluation.grasp_semantics; raw grasp is always instantaneous.",
                      "GT rectangle covers occluded orange pixels; nonempty mask does not prove visible orange.",
                      "Diagnostics do not change actions, observations, rewards, milestones or reset distribution."],
        }
        path = self.directory / "summary.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"GROUNDED_PICK_DIAGNOSTICS={path}", flush=True)
