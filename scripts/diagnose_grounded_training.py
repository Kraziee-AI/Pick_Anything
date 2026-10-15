"""Repeat complete GT-mask HOME evaluations and aggregate first-episode evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from grounded_pick_cli import PROJECT_ROOT, configure_caches


def main():
    configure_caches()
    from pick_anything.grounded_pick.grounding import workspace_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/grounded_pick/recovery_diagnosis")
    args = parser.parse_args()
    if Path(args.label).name != args.label or args.label in {".", ".."}:
        parser.error("label must be a single directory name")
    if args.num_envs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("num-envs must be positive and seeds must be unique")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    from pick_anything.grounded_pick.checkpoints import image_history_from_checkpoint
    image_history = image_history_from_checkpoint(checkpoint)
    root = workspace_output(args.output_root) / args.label
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    files = ["pick_anything/grounded_pick/progress.py", "pick_anything/grounded_pick/env.py",
             "pick_anything/grounded_pick/config.py", "pick_anything/grounded_pick/observations.py",
             "pick_anything/grounded_pick/rewards.py",
             "pick_anything/grounded_pick/checkpoints.py",
             "pick_anything/grounded_pick/diagnostics.py", "scripts/grounded_pick_cli.py",
             "pick_anything/tasks/pick_apple/stages.py", "pick_anything/tasks/pick_apple/camera_progress.py"]
    hashes = {}
    for name in files:
        data = (PROJECT_ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_sha256": hashes, "num_envs": args.num_envs, "seeds": args.seeds,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
                "mask_backend": "gt", "birth_policy": "HOME", "commands": []}
    metadata["image_history"] = image_history
    environment = os.environ.copy()
    environment["ISAACSIM_ASSET_ROOT"] = str(PROJECT_ROOT / "assets/isaac_cache")
    temporary = root / "tmp"
    temporary.mkdir()
    environment["TMPDIR"] = str(temporary.resolve())
    reports = []
    for seed in args.seeds:
        directory = root / f"seed{seed}"
        directory.mkdir()
        command = [sys.executable, str(PROJECT_ROOT / "scripts/eval_grounded_pick.py"),
                   "--mask-backend", "gt", "--num_envs", str(args.num_envs), "--device", args.device,
                   "--seed", str(seed), "--viz", "none", "--checkpoint", str(checkpoint),
                   "--diagnostics-dir", str(directory), "--trace-every", "5",
                   "--dump-dir", str(directory / "frames"), "--dump-every", "90", "--dump-max", "31"]
        metadata["commands"].append(command)
        (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Evaluating seed {seed}: {directory / 'run.log'}", flush=True)
        with (directory / "run.log").open("w") as output:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        if completed.returncode or not (directory / "summary.json").is_file():
            print("\n".join((directory / "run.log").read_text().splitlines()[-50:]), file=sys.stderr)
            raise SystemExit(completed.returncode or 1)
        reports.append(json.loads((directory / "summary.json").read_text()))
        print(json.dumps({"seed": seed, **reports[-1]["evaluation"]}, ensure_ascii=False), flush=True)
    episodes = [ep for report in reports for ep in report["episodes"]]
    causes = Counter(cause for ep in episodes for cause in ep["termination"])
    total = len(episodes)
    aggregate = {**metadata, "episodes": total, "termination_counts": dict(causes),
                 "success_rate": causes["success"] / total,
                 "max_stage_counts": dict(Counter(ep["max_stage"] for ep in episodes)),
                 "raw_grasp_3_steps": sum(ep["grasp_continuous_3_steps"] for ep in episodes),
                 "confirmed_grasp": sum(ep["first_grasp_latch"] is not None for ep in episodes),
                 "no_open_alignment": sum(ep["pregrasp_ready_steps"] == 0 for ep in episodes),
                 "lift_ready": sum(ep["lift_ready_steps"] > 0 for ep in episodes),
                 "mean_mask_nonempty_fraction": sum(ep["mask_nonempty_fraction"] for ep in episodes) / total,
                 "per_seed": [{"seed": report["seed"], **report["evaluation"]} for report in reports]}
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    print("GROUNDED_DIAGNOSIS=" + json.dumps({k: v for k, v in aggregate.items()
          if k not in {"commands", "source_sha256", "per_seed"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
