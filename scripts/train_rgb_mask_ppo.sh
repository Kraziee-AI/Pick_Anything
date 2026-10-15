#!/usr/bin/env bash
# RGB(3) + GT target mask(1) + proprioception(24) -> PPO -> joint actions(6).
# New training uses three RGB/mask frames initialized from the proven RGB actor.
# An explicit PICK_MASK_CHECKPOINT resumes the matching single/three-frame model.
set -euo pipefail

PICK_MASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/user123/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_61
cd "$PICK_MASK_ROOT"

export ISAACSIM_ASSET_ROOT="$PICK_MASK_ROOT/assets/isaac_cache"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${PICK_MASK_GPU:-0}"
export TMPDIR="$PICK_MASK_ROOT/outputs/grounded_pick/cache/tmp"
mkdir -p "$TMPDIR"

PICK_MASK_START_ARGS=()
if [[ -n "${PICK_MASK_CHECKPOINT:-}" ]]; then
  PICK_MASK_START_ARGS=(--checkpoint "$PICK_MASK_CHECKPOINT")
else
  PICK_MASK_START_ARGS=(--init-from-rgb "${PICK_MASK_RGB_CHECKPOINT:-$PICK_MASK_ROOT/models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt}")
fi
if [[ -n "${PICK_MASK_HISTORY:-}" ]]; then
  PICK_MASK_START_ARGS+=(--image-history "$PICK_MASK_HISTORY")
fi

exec python scripts/train_grounded_pick.py \
  --mask-backend gt \
  --num_envs "${PICK_MASK_ENVS:-128}" --device cuda:0 --seed 42 \
  --max_iterations "${PICK_MASK_ITERATIONS:-1500}" --viz none \
  --run_name rgb_mask_temporal_transfer_v3 \
  --training-recipe "${PICK_MASK_RECIPE:-transfer}" \
  "${PICK_MASK_START_ARGS[@]}" \
  "$@"
