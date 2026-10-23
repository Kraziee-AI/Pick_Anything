#!/usr/bin/env bash
# View the wrist camera at a sampled staged-reset spawn.
# Usage: bash scripts/view_spawn_camera.sh [--stage 3] [--save]
set -e
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    pick_cuda_runtime="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    if [[ -f "$pick_cuda_runtime" ]]; then
        export LD_PRELOAD="$pick_cuda_runtime${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi
cd "$(dirname "$0")/.."
python scripts/view_spawn_camera.py --visualizer kit --device "${PICK_DEMO_DEVICE:-cuda:0}" "$@"
