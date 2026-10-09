"""Complete-task deterministic evaluation plus command/actuator smoothness.

PICK_SMOOTH_EVAL_OUT selects output JSON, default photos/camera_ppo_diagnosis/eval.json.
Use --num_envs N: exactly one HOME episode per environment is counted.
Standard playback exports are disabled to preserve existing artifacts.
"""
from __future__ import annotations

import json
import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def skip_exports():
    """Playback callback: preserve checkpoint exports (EMA belongs to the environment)."""
    from rsl_rl.runners import OnPolicyRunner

    OnPolicyRunner.export_policy_to_jit = lambda *args, **kwargs: None
    OnPolicyRunner.export_policy_to_onnx = lambda *args, **kwargs: None
    return sys.argv[1:]


def install():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    skip_exports()

    original_step = RslRlVecEnvWrapper.step
    counted = None
    prev_command = prev_applied = prev_velocity = None
    sums = {key: 0.0 for key in ("command_delta_squared", "applied_delta_squared",
                                "velocity_reversals", "transitions")}
    counts = {key: 0 for key in ("success", "time_out", "staged_orange_dropped",
                                "staged_box_tipped", "staged_abnormal_robot")}
    steps = 0

    def step(self, actions):
        nonlocal counted, prev_command, prev_applied, prev_velocity, steps
        env = self.unwrapped
        if counted is None:
            counted = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)
        active = ~counted
        velocity = tensor(env.scene["robot"].data.joint_vel).clone()
        command = actions.clamp(-1, 1).clone()
        result = original_step(self, actions)
        applied = torch.cat([
            env.action_manager.get_term(name).processed_actions / env.cfg.actions.__dict__[name].scale
            for name in ("arm_action", "gripper_action")
        ], dim=-1).clone()
        if prev_command is not None:
            sums["transitions"] += int(active.sum())
            sums["command_delta_squared"] += float(((command - prev_command)[active] ** 2).sum())
            sums["applied_delta_squared"] += float(((applied - prev_applied)[active] ** 2).sum())
            reversal = ((velocity * prev_velocity < 0) & (velocity.abs() > 0.1)
                        & (prev_velocity.abs() > 0.1))
            sums["velocity_reversals"] += float(reversal[active, :5].sum())
        prev_command, prev_applied, prev_velocity = command, applied, velocity
        dones = result[2].bool() & active
        for name in counts:
            counts[name] += int(env.termination_manager.get_term(name)[dones].sum())
        counted |= dones
        steps += 1
        if steps % 300 == 0:
            print(f"[smooth-eval] step={steps}; finished={int(counted.sum())}/{self.num_envs}; "
                  f"success={counts['success']}", flush=True)
        if not bool(counted.all()):
            return result
        transitions = max(1, sums["transitions"])
        summary = {
            "episodes": self.num_envs, "steps": steps,
            "success_rate": counts["success"] / self.num_envs,
            "termination_counts": counts,
            "mean_command_delta_squared": sums["command_delta_squared"] / transitions,
            "mean_applied_delta_squared": sums["applied_delta_squared"] / transitions,
            "arm_velocity_reversal_fraction": sums["velocity_reversals"] / (transitions * 5),
            "arm_alpha": getattr(env.cfg.actions.arm_action, "alpha", 1.0),
            "action_rate_weight": env.cfg.rewards.action_rate.weight,
            "birth_policy": "forced HOME; first episode per environment",
        }
        output = Path(os.environ.get("PICK_SMOOTH_EVAL_OUT", "photos/camera_ppo_diagnosis/eval.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print("CAMERA_SMOOTH_EVAL=" + json.dumps(summary), flush=True)
        raise SystemExit(0)

    RslRlVecEnvWrapper.step = step
    return sys.argv[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str)
    custom, remaining = parser.parse_known_args()
    if custom.output:
        os.environ["PICK_SMOOTH_EVAL_OUT"] = custom.output
    sys.argv = [sys.argv[0]] + remaining
    import pick_anything.tasks  # noqa: F401
    from isaaclab_rl.entrypoints.dispatch import run_play_cli

    sys.argv.extend(["--external_callback", "scripts.eval_camera_smooth.install"])
    raise SystemExit(run_play_cli())
