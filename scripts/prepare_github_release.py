"""Audit/export the public working tree without staging, deleting, or pushing files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {"assets", "pick_anything", "scripts", "small_grasp", "media"}
PUBLIC_TOP = {"README.md", ".gitignore", ".gitattributes", "requirements.txt"}
SELECTED_WEIGHTS = {
    "models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt",
    "small_grasp/models/graspnet_epoch18.tar",
    "reset_states/pick_orange_bank_v1.pt",
}
SKIP_DIRS = {".git", "__pycache__", "outputs", "output", "logs", "cache"}
LIMIT_BYTES = 95 * 1024 * 1024


def public_path(relative):
    """Explicit publication policy, including files already tracked by Git."""
    p = Path(relative)
    name = p.as_posix()
    if name in PUBLIC_TOP or name in SELECTED_WEIGHTS:
        return True
    if not p.parts or p.parts[0] not in PUBLIC_ROOTS:
        return False
    if any(s in SKIP_DIRS or s.startswith(".") for s in p.parts):
        return False
    if p.suffix.lower() in {".md", ".pyc", ".pyo", ".log", ".orig"}:
        return False
    if ".bak" in p.name or p.name == "imgui.ini":
        return False
    if p.parts[:2] in {("small_grasp", "research"), ("small_grasp", "models")}:
        return False
    if p.parts[0] == "media":
        return p.name in {
            "rl_grasp.mp4", "rl_grasp_preview.jpg",
            "language_grasp.mp4", "language_grasp_preview.jpg", "provenance.json",
        }
    return True


def candidates():
    for relative in PUBLIC_TOP | SELECTED_WEIGHTS:
        p = ROOT / relative
        if p.is_file():
            yield p
    for name in sorted(PUBLIC_ROOTS):
        for folder, dirs, files in os.walk(ROOT / name, followlinks=False):
            here = Path(folder)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                       and not (here == ROOT / "small_grasp" and d in {"research", "models"})]
            for name in sorted(files):
                p = here / name
                if public_path(p.relative_to(ROOT)):
                    yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="New directory inside outputs/; never overwrites")
    parser.add_argument("--report", type=Path, default=Path("outputs/github_release_audit.json"))
    args = parser.parse_args()
    report = (ROOT / args.report).resolve()
    if not report.is_relative_to(ROOT / "outputs"):
        parser.error("report must stay inside this repository's outputs/")
    target = (ROOT / args.export).resolve() if args.export else None
    if target and (not target.is_relative_to(ROOT / "outputs") or target.exists()):
        parser.error("export must be a new directory inside this repository's outputs/")
    entries = {}
    for p in sorted(set(candidates())):
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f"Refusing symlink/outside-workspace file: {p.relative_to(ROOT)}")
        size = p.stat().st_size
        if size > LIMIT_BYTES:
            raise ValueError(f"File exceeds source-upload limit: {p.relative_to(ROOT)}")
        entries[p.relative_to(ROOT).as_posix()] = {
            "bytes": size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    required = PUBLIC_TOP | SELECTED_WEIGHTS | {
        "media/rl_grasp.mp4", "media/rl_grasp_preview.jpg",
        "media/language_grasp.mp4", "media/language_grasp_preview.jpg",
        "small_grasp/vendor/LICENSE",
    }
    missing = sorted(required - entries.keys())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    tracked_excluded = sorted(p for p in tracked if p and not public_path(p))
    result = {
        "files": entries, "file_count": len(entries),
        "total_bytes": sum(x["bytes"] for x in entries.values()),
        "missing_required": missing,
        "tracked_files_excluded_from_snapshot": tracked_excluded,
        "git_index_modified": False, "push_performed": False,
        "note": "Snapshot exclusion does not remove files from existing Git history or index.",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    if target:
        if missing:
            raise SystemExit(f"Public package incomplete; see {report.relative_to(ROOT)}")
        target.mkdir(parents=True)
        for relative in entries:
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dst)
        for relative, record in entries.items():
            if hashlib.sha256((target / relative).read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Export hash mismatch: {relative}")
        result["export_directory"] = str(target.relative_to(ROOT))
        result["export_hashes_verified"] = True
        report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
