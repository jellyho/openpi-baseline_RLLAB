#!/usr/bin/env bash
# Adaptive Q-Chunking (ACSAC) — offline pretraining + offline-to-online fine-tuning.
#
# Reproduces the paper protocol: 1M offline gradient steps followed by 1M online
# environment steps with the SAME objective (online transitions are simply added to the
# dataset). Run from the `qc/` directory.
#
# Per-task hyperparameters (Table 3): all listed domains use H=5; N=4 except
# cube-quadruple which uses N=8. The scene / puzzle domains use sparse rewards.
#
# Usage:   bash scripts/aqc/offline_to_online.sh <domain> [seed]
#   domain in: cube-double | cube-triple | scene | puzzle
set -euo pipefail

# Activate the conda env that has jax/flax/ogbench/mujoco (override with CONDA_ENV=...).
CONDA_ENV="${CONDA_ENV:-deas_real}"
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# JAX in deas_real bundles its own CUDA wheels; the default-loaded system cuda
# module conflicts and breaks the plugin (cuSPARSE not found -> CPU fallback). Unset it.
unset LD_LIBRARY_PATH

DOMAIN="${1:-cube-double}"
SEED="${2:-0}"
N=4
SPARSE=False

case "$DOMAIN" in
  cube-double) ENV=cube-double-play-singletask-task1-v0 ;;
  cube-triple) ENV=cube-triple-play-singletask-task2-v0 ;;
  scene)       ENV=scene-play-singletask-task1-v0;        SPARSE=True ;;
  puzzle)      ENV=puzzle-3x3-play-singletask-task1-v0;   SPARSE=True ;;
  *) echo "unknown domain: $DOMAIN"; exit 1 ;;
esac

MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --wandb_entity="${WANDB_ENTITY:-gwanwoo-yonsei-university}" \
  --wandb_project="${WANDB_PROJECT:-AQC}" \
  --run_group=aqc \
  --env_name="${ENV}" \
  --seed="${SEED}" \
  --sparse="${SPARSE}" \
  --horizon_length=5 \
  --agent.num_action_samples="${N}" \
  --agent.adaptive_chunking=True \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --video_episodes=1
