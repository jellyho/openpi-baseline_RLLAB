#!/usr/bin/env bash
# Adaptive Q-Chunking (ACSAC) — quick smoke test (a few thousand offline steps).
# Verifies the full pipeline runs end-to-end on a real OGBench env. Run from `qc/`.
set -euo pipefail

# Activate the conda env that has jax/flax/ogbench/mujoco (override with CONDA_ENV=...).
CONDA_ENV="${CONDA_ENV:-deas_real}"
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# JAX in deas_real bundles its own CUDA wheels; the default-loaded system cuda
# module conflicts and breaks the plugin (cuSPARSE not found -> CPU fallback). Unset it.
unset LD_LIBRARY_PATH

ENV="${1:-cube-double-play-singletask-task1-v0}"

MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --wandb_entity="${WANDB_ENTITY:-gwanwoo-yonsei-university}" \
  --wandb_project="${WANDB_PROJECT:-AQC}" \
  --run_group=aqc_smoke \
  --env_name="${ENV}" \
  --seed=0 \
  --horizon_length=5 \
  --agent.num_action_samples=4 \
  --offline_steps=5000 \
  --online_steps=2000 \
  --start_training=1000 \
  --eval_interval=5000 \
  --eval_episodes=5 \
  --video_episodes=1 \
  --log_interval=1000
