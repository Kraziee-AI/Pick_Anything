#!/usr/bin/env bash
# 查看任务场景（demo 演示版：贴图苹果/橘子、纸盒、木桌、SO101）
# 用法: bash scripts/view_scene.sh [seed]
set -e
export LD_PRELOAD=/home/user123/miniconda3/envs/isaaclab_61/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
cd "$(dirname "$0")/.."
SEED="${1:-42}"
python scripts/demo_pick_scene.py --visualizer kit --device cuda:2 --seed "$SEED"
