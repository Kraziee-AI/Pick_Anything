"""Run reproducible GT/DINO comparisons with frozen single- or multi-object actors."""

import argparse
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
    from pick_anything.grounded_pick.checkpoints import image_history_from_checkpoint
    from pick_anything.tabletop.catalog import OBJECTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene", choices=["single", "multi"], default="single")
    parser.add_argument("--targets", nargs="+", choices=list(OBJECTS), default=["orange"])
    parser.add_argument("--backends", nargs="+", choices=["gt", "dino"], default=["gt", "dino"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grounding-device", default="")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.scene == "single" and args.targets != ["orange"]:
        parser.error("single scene supports only orange")
    if args.num_envs < 1 or args.timeout < 1:
        parser.error("num-envs and timeout must be positive")
    for values in (args.targets, args.backends, args.seeds):
        if len(set(values)) != len(values):
            parser.error("duplicate targets, backends or seeds are not allowed")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    history = image_history_from_checkpoint(checkpoint)
    root = workspace_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "image_history": history, "scene": args.scene, "num_envs": args.num_envs,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "runs": []}
    # Preserve exactly the local task code and launcher used for this comparison.
    source_paths = list((PROJECT_ROOT / "pick_anything/grounded_pick").glob("*.py"))
    source_paths += list((PROJECT_ROOT / "pick_anything/open_vocab_pick").glob("*.py"))
    source_paths += list((PROJECT_ROOT / "pick_anything/tabletop").glob("*.py"))
    source_paths += [Path(__file__).resolve(), PROJECT_ROOT / "scripts/grounded_pick_cli.py",
                     PROJECT_ROOT / "scripts/open_vocab_pick_cli.py"]
    metadata["source_sha256"] = {}
    for path in source_paths:
        relative = path.relative_to(PROJECT_ROOT)
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        destination.write_bytes(data)
        metadata["source_sha256"][str(relative)] = hashlib.sha256(data).hexdigest()
    environment = os.environ.copy()
    environment["ISAACSIM_ASSET_ROOT"] = str(PROJECT_ROOT / "assets/isaac_cache")
    (root / "tmp").mkdir()
    environment["TMPDIR"] = str(root / "tmp")
    entry = "eval_grounded_pick.py" if args.scene == "single" else "eval_open_vocab_pick.py"
    marker = "GROUNDED_PICK_EVAL_RESULT=" if args.scene == "single" else "OPEN_VOCAB_PICK_EVAL_RESULT="
    for target in args.targets:
        for seed in args.seeds:
            for backend in args.backends:
                directory = root / f"{target}_{backend}_seed{seed}"
                directory.mkdir()
                command = [sys.executable, str(PROJECT_ROOT / "scripts" / entry),
                           "--checkpoint", str(checkpoint), "--instruction", OBJECTS[target].caption,
                           "--mask-backend", backend, "--num_envs", str(args.num_envs),
                           "--device", args.device, "--grounding-device", args.grounding_device,
                           "--seed", str(seed), "--viz", "none", "--dump-dir", str(directory / "frames"),
                           "--dump-every", "90", "--dump-max", "31"]
                if args.scene == "single":
                    command += ["--diagnostics-dir", str(directory)]
                else:
                    command += ["--target-key", target, "--actor-only"]
                row = {"target": target, "seed": seed, "backend": backend, "command": command, "status": "running"}
                metadata["runs"].append(row)
                (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"Evaluating {directory.name}; log: {directory / 'run.log'}", flush=True)
                try:
                    with (directory / "run.log").open("w") as stream:
                        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                                                   stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                    lines = (directory / "run.log").read_text().splitlines()
                    results = [json.loads(line.split(marker, 1)[1]) for line in lines if line.startswith(marker)]
                    if completed.returncode or len(results) != 1:
                        raise RuntimeError(f"exit={completed.returncode}, evaluation results={len(results)}; see run.log")
                    row.update(status="complete", result=results[0])
                    (directory / "result.json").write_text(json.dumps(results[0], indent=2, ensure_ascii=False) + "\n")
                    print(json.dumps(results[0], ensure_ascii=False), flush=True)
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    row.update(status="failed", error=str(exc))
                    raise
                finally:
                    (root / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (root / "results.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
