"""Initialize the four-channel PPO from a compatible RGB grasping policy."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import torch


def image_history_from_checkpoint(path: str | Path) -> int:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    weight = saved["actor_state_dict"].get("cnns.rgb_mask.0.weight")
    if weight is None or weight.ndim != 4 or weight.shape[1] not in (4, 12):
        raise ValueError("Expected a four-channel or three-frame RGB/mask checkpoint")
    return weight.shape[1] // 4


def rgb_to_rgb_mask_checkpoint(source: dict, image_history: int = 3, learning_rate: float = 2e-5) -> dict:
    """Preserve RGB behavior, add a trainable zero-initialized mask kernel.

    This is initialization, not a continuation of the RGB optimizer/iteration.
    Normalizers and the compatible privileged critic are copied. The optimizer
    parameter groups are retained, with all Adam moments discarded.
    """
    if image_history not in (1, 3) or not 0 < learning_rate < 1:
        raise ValueError("image_history must be 1 or 3 and learning_rate must be in (0,1)")
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
    if not required.issubset(source):
        raise ValueError(f"Missing PPO checkpoint keys: {sorted(required - source.keys())}")
    actor = source["actor_state_dict"]
    key = "cnns.wrist_rgb.0.weight"
    if key not in actor or actor[key].ndim != 4 or actor[key].shape[1] != 3:
        raise ValueError("Expected a three-channel wrist_rgb CNN checkpoint")
    if any(name.startswith("cnns.") and not name.startswith("cnns.wrist_rgb.") for name in actor):
        raise ValueError("Only a single wrist_rgb encoder can be transferred")
    if tuple(actor["obs_normalizer._mean"].shape) != (1, 24):
        raise ValueError("Expected 24-dimensional proprioception")
    if actor["mlp.4.weight"].shape != (6, 128):
        raise ValueError("Expected the compatible six-action PPO actor")
    if tuple(source["critic_state_dict"]["obs_normalizer._mean"].shape) != (1, 49):
        raise ValueError("Expected the compatible 49-dimensional critic")
    result = deepcopy(source)
    result["actor_state_dict"] = {
        name.replace("cnns.wrist_rgb.", "cnns.rgb_mask."): value.clone()
        for name, value in actor.items()
    }
    original = actor[key]
    padded = original.new_zeros((original.shape[0], 4 * image_history, *original.shape[2:]))
    # Histories are oldest -> newest; preserve the source's CURRENT RGB action.
    start = 4 * (image_history - 1)
    padded[:, start:start + 3].copy_(original)
    result["actor_state_dict"]["cnns.rgb_mask.0.weight"] = padded
    result["optimizer_state_dict"]["state"] = {}
    for group in result["optimizer_state_dict"]["param_groups"]:
        group["lr"] = learning_rate
    result["iter"] = 0
    result["infos"] = {"initialization": "rgb_to_rgb_mask_v1", "source_iteration": source["iter"],
                       "image_history": image_history, "learning_rate": learning_rate}
    return result


def initialize_from_rgb(source_path: str | Path, output_dir: str | Path,
                        image_history: int = 3, learning_rate: float = 2e-5) -> Path:
    """Save a versioned initialization within the workspace without overwrites."""
    from .grounding import workspace_output

    source_path = Path(source_path).expanduser().resolve(strict=True)
    output_dir = workspace_output(output_dir)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output = output_dir / f"rgb_to_rgb_mask_v1_h{image_history}_lr{learning_rate:g}_{digest[:16]}.pt"
    if output.exists():
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("infos", {}).get("source_sha256") != digest:
            raise ValueError(f"Initialization metadata mismatch; refusing to overwrite {output}")
        return output
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    result = rgb_to_rgb_mask_checkpoint(source, image_history, learning_rate)
    result["infos"].update(source_checkpoint=str(source_path), source_sha256=digest)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an existing initialization.
    with output.open("xb") as stream:
        torch.save(result, stream)
    return output
