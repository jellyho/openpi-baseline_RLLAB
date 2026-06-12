#!/bin/bash
# ============================================================
#  Stage 4 — train the RLT / AQC prefix critic on the precomputed
#  rl_token / base_action / mc_return columns (NO VLA forward).
#
#  Wraps scripts/train_rlt_critic.py, which uses its OWN config registry
#  (VLAAQCConfig in src/openpi/rlt_critic/config.py) — this is NOT an openpi
#  TrainConfig; the critic is a small flax.linen transformer with a bespoke
#  TD/MC-warmup loop, so it lives outside scripts/train.py.
#
#  Usage:
#    ./stage4_train_critic.sh [CONFIG] [BATCH] [GPU]
#      CONFIG  VLAAQCConfig preset (default vla_aqc_warmup; see rlt_critic/config.py)
#      BATCH   global batch size (default 64)
#      GPU     CUDA device index (default 0)
#
#  Env:
#    CKPT_DIR      run output base (run dir = <CKPT_DIR>/<name>/<exp>).
#                  Default: $RLT_CRITIC_CKPT_DIR from setup_env.sh.
#    MEM_FRACTION  XLA GPU memory cap (default 0.9; PREALLOCATE=false so it only grows).
#    EXTRA         extra args to train_rlt_critic.py
#                  (e.g. EXTRA="--task seal-water-bottle-cap --data_root /path").
#
#  Detached / auto-resuming variants live in scripts/train_rlt_critic{,_supervised}.sh.
# ============================================================
set -e
source setup_env.sh

CONFIG="${1:-vla_aqc_warmup}"
BATCH="${2:-64}"
GPU="${3:-0}"
CKPT_DIR="${CKPT_DIR:-$RLT_CRITIC_CKPT_DIR}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
EXTRA="${EXTRA:-}"

mkdir -p logs
LOG="logs/rlt_critic_${CONFIG}_$(date +%Y%m%d-%H%M%S).log"
echo "config=$CONFIG  batch=$BATCH  gpu=$GPU"
echo "ckpt_dir=$CKPT_DIR   (run dir = $CKPT_DIR/$CONFIG/<exp>)"
echo "log -> $LOG"

CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
uv run scripts/train_rlt_critic.py \
    --config "$CONFIG" --batch_size "$BATCH" --loader_processes 4 \
    --checkpoint_base_dir "$CKPT_DIR" --resume $EXTRA 2>&1 | tee "$LOG"
