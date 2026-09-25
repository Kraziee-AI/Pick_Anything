#!/usr/bin/env bash
# 巡览八阶段 reset 状态库（逐行恢复+物理保持，可视化检查每个出生阶段）
# 用法: bash scripts/view_reset_bank.sh [额外参数，如 "--stages 2,3,7"]
set -e
export LD_PRELOAD=/home/user123/miniconda3/envs/isaaclab_61/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
cd "$(dirname "$0")/.."
python scripts/view_reset_bank.py --visualizer kit --device cuda:2 "$@"
