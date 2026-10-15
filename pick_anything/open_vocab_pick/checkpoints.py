"""Strict actor-only transfer for evaluation across different critic contracts."""

import torch


def load_evaluation_actor(runner, path, load_cfg=None, strict=True, map_location=None):
    """Preserve all actor buffers; leave critic, optimizer and iteration untouched."""
    if load_cfg is not None or not strict:
        raise ValueError("Actor-only evaluation requires strict loading and no load_cfg override")
    saved = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    runner.alg.load(saved, {"actor": True}, strict=True)
    print(f"[OPEN-VOCAB] Strict actor-only evaluation: {path}", flush=True)
    return saved.get("infos")
