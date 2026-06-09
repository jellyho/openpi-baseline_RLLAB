#!/usr/bin/env bash
# Adaptive Q-Chunking (ACSAC) — cube-quadruple with the streamed 100M-transition dataset.
# Per Table 3, cube-quadruple uses N=8 (vs N=4 elsewhere). The 100M dataset is too large
# to hold in memory, so it is streamed in shards via --ogbench_dataset_dir and rotated
# every --dataset_replace_interval steps (same mechanism QC uses). Run from `qc/`.
#
# Usage: bash scripts/aqc/cube_quadruple_100m.sh <dataset_dir> [env_name] [seed]
set -euo pipefail

# Activate the conda env that has jax/flax/ogbench/mujoco (override with CONDA_ENV=...).
CONDA_ENV="${CONDA_ENV:-deas_real}"
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# JAX in deas_real bundles its own CUDA wheels; the default-loaded system cuda
# module conflicts and breaks the plugin (cuSPARSE not found -> CPU fallback). Unset it.
unset LD_LIBRARY_PATH

DATASET_DIR="${1:?path to directory of cube-quadruple .npz shards required}"
ENV="${2:-cube-quadruple-play-singletask-task1-v0}"
SEED="${3:-0}"

MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --wandb_entity="${WANDB_ENTITY:-gwanwoo-yonsei-university}" \
  --wandb_project="${WANDB_PROJECT:-AQC}" \
  --run_group=aqc \
  --env_name="${ENV}" \
  --seed="${SEED}" \
  --horizon_length=5 \
  --agent.num_action_samples=8 \
  --ogbench_dataset_dir="${DATASET_DIR}" \
  --dataset_replace_interval=1000 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --video_episodes=1
