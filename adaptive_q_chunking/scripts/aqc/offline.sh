#!/usr/bin/env bash
# Adaptive Q-Chunking (ACSAC) — offline-only training on OGBench.
# 1M offline gradient steps, evaluating every 100K steps (paper offline protocol).
# Run from the `qc/` directory.
#
# Usage:   bash scripts/aqc/offline.sh <env_name> [seed] [sparse]
set -euo pipefail

# Activate the conda env that has jax/flax/ogbench/mujoco (override with CONDA_ENV=...).
CONDA_ENV="${CONDA_ENV:-deas_real}"
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# JAX in deas_real bundles its own CUDA wheels; the default-loaded system cuda
# module conflicts and breaks the plugin (cuSPARSE not found -> CPU fallback). Unset it.
unset LD_LIBRARY_PATH

ENV="${1:-cube-double-play-singletask-task1-v0}"
SEED="${2:-0}"
SPARSE="${3:-False}"

MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --wandb_entity="${WANDB_ENTITY:-gwanwoo-yonsei-university}" \
  --wandb_project="${WANDB_PROJECT:-AQC}" \
  --run_group=aqc_offline \
  --env_name="${ENV}" \
  --seed="${SEED}" \
  --sparse="${SPARSE}" \
  --horizon_length=5 \
  --agent.num_action_samples=4 \
  --offline_steps=1000000 \
  --online_steps=0 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --video_episodes=1
