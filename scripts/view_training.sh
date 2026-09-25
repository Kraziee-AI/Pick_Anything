#!/usr/bin/env bash
# 查看 RL 训练环境实况（60mm 方块苹果 + 简化桌 + 随机策略运行的机械臂）
# 用法: bash scripts/view_training.sh
set -e
export LD_PRELOAD=/home/user123/miniconda3/envs/isaaclab_61/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
cd "$(dirname "$0")/.."
python scripts/train_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 --num_envs 4 --device cuda:2 --max_iterations 100 --seed 7 --viz kit
