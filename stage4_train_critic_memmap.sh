#!/bin/bash
# ============================================================
#  Stage 4 (FAST) — train the RLT/AQC critic over a frame-indexed MEMMAP.
#
#  Builds the memmap ONCE (scripts/preprocess_memmap.py), then runs DDP training
#  that reads it by index. The memmap is page-cached in RAM and shared read-only
#  across the DDP loader workers (one physical copy), so the loader is no longer
#  the bottleneck: ~3.4x single-process and near-linear multi-process vs parquet
#  (4w 232/s, 8w 431/s @ B=256). Drop-in alternative to stage4_train_critic.sh.
#
#  Usage:
#    ./stage4_train_critic_memmap.sh <DATASET_ROOT> [CONFIG] [GPUS] [BATCH]
#      DATASET_ROOT  annotated dataset (has rl_token/base_action/reward/mc_return)
#      CONFIG        VLAAQCConfig preset (default vla_aqc_warmup)
#      GPUS          comma list for DDP (default 0,1,2,3); BATCH must be divisible
#      BATCH         global batch size (default 256)
#
#  Env:
#    MEMMAP_DIR    build/read the memmap here (default <DATASET>_memmap). Put on a
#                  fast disk, or /dev/shm if it fits in RAM (instant share).
#    WORKERS       loader worker processes (default 8; they share the page cache).
#    SUBSET        bootstrap_subset candidates (default 0 = all 32; 8 = 4x less
#                  candidate read + host->device transfer, REDQ-style).
#    MEM_FRACTION  XLA GPU mem cap (default 0.9; PREALLOCATE=false so it only grows
#                  to need — LOWER to ~0.04 when sharing busy GPUs).
#    CKPT_DIR      run output base (default $RLT_CRITIC_CKPT_DIR from setup_env.sh).
#    EXTRA         extra args to train_rlt_critic.py (e.g. "--resume").
#
#  Background it for overnight runs:
#    nohup setsid ./stage4_train_critic_memmap.sh <DATASET> ... >/dev/null 2>&1 </dev/null &
# ============================================================
set -e
source setup_env.sh

DATASET="${1:?usage: ./stage4_train_critic_memmap.sh <DATASET_ROOT> [CONFIG] [GPUS] [BATCH]}"
CONFIG="${2:-vla_aqc_warmup}"
GPUS="${3:-0,1,2,3}"
BATCH="${4:-1024}"
MEMMAP_DIR="${MEMMAP_DIR:-${DATASET%/}_memmap}"
WORKERS="${WORKERS:-32}"
SUBSET="${SUBSET:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
CKPT_DIR="${CKPT_DIR:-$RLT_CRITIC_CKPT_DIR}"
EXTRA="${EXTRA:-}"

# train_rlt_critic.py auto-builds the memmap from $DATASET on first run (one-time), then
# reuses it -> a single command handles preprocessing. (To pre-build explicitly instead:
#   uv run scripts/preprocess_memmap.py --input "$DATASET" --out "$MEMMAP_DIR" --workers 4)

# DDP train over the memmap (--data_root tells the auto-builder where to read from).
mkdir -p logs
LOG="logs/rlt_critic_${CONFIG}_memmap_$(date +%Y%m%d-%H%M%S).log"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
echo "config=$CONFIG  gpus=$GPUS (DDP x$NGPU)  batch=$BATCH (=$((BATCH / NGPU))/dev)  workers=$WORKERS  subset=$SUBSET"
echo "memmap=$MEMMAP_DIR   ckpt=$CKPT_DIR"
echo "log -> $LOG"

CUDA_VISIBLE_DEVICES="$GPUS" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
uv run scripts/train_rlt_critic.py \
    --config "$CONFIG" --data_root "$DATASET" --memmap_dir "$MEMMAP_DIR" \
    --batch_size "$BATCH" --loader_processes "$WORKERS" --bootstrap_subset "$SUBSET" \
    --checkpoint_base_dir "$CKPT_DIR" $EXTRA \
  2>&1 | tee "$LOG"
