"""Measure deterministic PPO action roughness and playback cost without changing training.

Use play_pick.py with --external_callback scripts.diagnose_camera_ppo.install.
PICK_DIAG_STEPS (default 600), PICK_DIAG_OUT (default photos/camera_ppo_diagnosis).
Disables automatic exports so existing exported policies are preserved.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def install():
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    steps = int(os.environ.get("PICK_DIAG_STEPS", "600"))
    output = Path(os.environ.get("PICK_DIAG_OUT", "photos/camera_ppo_diagnosis"))
    records = []
    policy_times = []
    original_policy = OnPolicyRunner.get_inference_policy
    original_step = RslRlVecEnvWrapper.step

    def sync(device):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    def get_policy(self, device=None):
        model = original_policy(self, device)

        class TimedPolicy:
            def __call__(self, obs):
                sync(device)
                start = time.perf_counter()
                result = model(obs)
                sync(device)
                policy_times.append((time.perf_counter() - start) * 1000)
                return result

            def reset(self, dones):
                model.reset(dones)

        return TimedPolicy()

    def step(self, actions):
        env = self.unwrapped
        sync(env.device)
        start = time.perf_counter()
        result = original_step(self, actions)
        sync(env.device)
        elapsed = (time.perf_counter() - start) * 1000
        state = env.staged_pick_state
        records.append({
            "raw": actions.detach().cpu().numpy().copy(),
            "q": tensor(env.scene["robot"].data.joint_pos).cpu().numpy().copy(),
            "v": tensor(env.scene["robot"].data.joint_vel).cpu().numpy().copy(),
            "stage": state.progress.stage.cpu().numpy().copy(),
            "done": result[2].cpu().numpy().copy(),
            "step_ms": elapsed,
        })
        if len(records) < steps:
            return result
        output.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.stack([record[key] for record in records]) for key in records[0]}
        raw = arrays["raw"]
        clipped = np.clip(raw, -1, 1)
        valid = ~arrays["done"][:-1].astype(bool)
        delta = np.diff(clipped, axis=0)[valid]
        flips = ((clipped[1:] * clipped[:-1] < 0) &
                 (np.abs(clipped[1:]) > 0.2) & (np.abs(clipped[:-1]) > 0.2))[valid]

        def timing(values):
            values = np.asarray(values)[20:]
            return {"mean": float(values.mean()), "p50": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)), "max": float(values.max())}

        summary = {
            "steps": steps, "num_envs": self.num_envs,
            "joint_names": env.scene["robot"].joint_names,
            "physics_dt": env.physics_dt, "control_dt": env.step_dt,
            "image_shape": list(env.scene.sensors["wrist_camera"].image_shape),
            "action_rate_weight": env.cfg.rewards.action_rate.weight,
            "policy_ms": timing(policy_times), "environment_ms": timing(arrays["step_ms"]),
            "total_ms": timing(arrays["step_ms"] + np.asarray(policy_times)),
            "saturated_fraction_per_action": (np.abs(raw) >= 1).mean(axis=(0, 1)).tolist(),
            "rms_delta_per_action": np.sqrt((delta ** 2).mean(axis=0)).tolist(),
            "strong_reversal_fraction_per_action": flips.mean(axis=0).tolist(),
            "mean_sum_delta_squared": float((delta ** 2).sum(axis=-1).mean()),
            "completed_episodes": int(arrays["done"].sum()),
        }
        summary["by_stage"] = {}
        for stage in np.unique(arrays["stage"][:-1]):
            mask = valid & (arrays["stage"][:-1] == stage)
            d = np.diff(clipped, axis=0)[mask]
            summary["by_stage"][str(int(stage))] = {
                "samples": int(mask.sum()),
                "mean_sum_delta_squared": float((d ** 2).sum(-1).mean()),
            }
        np.savez_compressed(output / "rollout.npz", **arrays, policy_ms=policy_times)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("CAMERA_PPO_DIAG=" + json.dumps(summary), flush=True)
        raise SystemExit(0)

    OnPolicyRunner.get_inference_policy = get_policy
    OnPolicyRunner.export_policy_to_jit = lambda *args, **kwargs: None
    OnPolicyRunner.export_policy_to_onnx = lambda *args, **kwargs: None
    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]
