#!/usr/bin/env bash
# Pure PPO model_13000 + tested arm smoothing, one environment, real-time playback.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_PRELOAD="/home/user123/miniconda3/envs/isaaclab_61/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
exec /home/user123/miniconda3/envs/isaaclab_61/bin/python scripts/play_pick.py \
  --rl_library rsl_rl --task Pick-Apple-SO101-Staged-Camera-Filtered-v0 \
  --num_envs 1 --device cuda:0 --seed 42 --viz kit --real-time \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt \
  --external_callback scripts.eval_camera_smooth.skip_exports "$@"
