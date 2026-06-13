#!/bin/bash
# ============================================================
#  Stage 4 (FAST) — train the RLT/AQC critic over a frame-indexed MEMMAP.
#
#  CONFIG-FIRST: the preset's own data_root (data_root_override, or TASKS[task]) IS the
#  dataset -- you don't repeat the path. train_rlt_critic.py auto-builds the memmap from it
#  on first run (one-time) and reuses it after; the memmap is page-cached in RAM and shared
#  read-only across the DDP loader workers, so the loader is no longer the bottleneck.
#
#  Usage:
#    ./stage4_train_critic_memmap.sh [CONFIG] [GPUS] [BATCH]
#      CONFIG  VLAAQCConfig preset (default vla_aqc_warmup); its data_root is the dataset
#      GPUS    comma list for DDP (default 0,1,2,3); BATCH must be divisible
#      BATCH   global batch size (default 1024)
#
#  Env:
#    DATA_ROOT     override the config's dataset path (default: the config's data_root)
#    MEMMAP_DIR    memmap location (default "auto" = <data_root>_memmap; e.g. set to
#                  /dev/shm/<name> for a RAM-tmpfs copy)
#    WORKERS       loader worker processes (default 32; they share the page cache)
#    SUBSET        bootstrap_subset candidates (default 0 = all 32; 8/16 = less candidate I/O)
#    MEM_FRACTION  XLA GPU mem cap (default 0.9; PREALLOCATE=false so it only grows; LOWER to
#                  ~0.04 when sharing busy GPUs)
#    CKPT_DIR      run output base (default $RLT_CRITIC_CKPT_DIR from setup_env.sh)
#    EXTRA         extra args to train_rlt_critic.py (e.g. "--resume")
#
#  Examples:
#    ./stage4_train_critic_memmap.sh vla_aqc_insert-mouse-battery          # uses the config's dataset
#    DATA_ROOT=/path/to/other_ds ./stage4_train_critic_memmap.sh vla_aqc_warmup 0,1 512
#    MEMMAP_DIR=/dev/shm/mb_mm SUBSET=16 ./stage4_train_critic_memmap.sh vla_aqc_insert-mouse-battery
#  Background (overnight):
#    nohup setsid ./stage4_train_critic_memmap.sh vla_aqc_insert-mouse-battery >/dev/null 2>&1 </dev/null &
# ============================================================
set -e
source setup_env.sh

CONFIG="${1:-vla_aqc_warmup}"
GPUS="${2:-0,1,2,3}"
BATCH="${3:-1024}"
MEMMAP_DIR="${MEMMAP_DIR:-auto}"          # "auto" -> <data_root>_memmap (derived in train)
WORKERS="${WORKERS:-0}"
SUBSET="${SUBSET:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
CKPT_DIR="${CKPT_DIR:-$RLT_CRITIC_CKPT_DIR}"
EXTRA="${EXTRA:-}"

mkdir -p logs
LOG="logs/rlt_critic_${CONFIG}_memmap_$(date +%Y%m%d-%H%M%S).log"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
echo "config=$CONFIG  gpus=$GPUS (DDP x$NGPU)  batch=$BATCH (=$((BATCH / NGPU))/dev)  workers=$WORKERS  subset=$SUBSET"
echo "memmap=$MEMMAP_DIR   data_root=${DATA_ROOT:-<config default>}   ckpt=$CKPT_DIR"
echo "log -> $LOG"

# --data_root is passed ONLY when DATA_ROOT is set (else the config's data_root_override drives it).
CUDA_VISIBLE_DEVICES="$GPUS" \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
uv run scripts/train_rlt_critic.py \
    --config "$CONFIG" --memmap_dir "$MEMMAP_DIR" \
    ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
    --batch_size "$BATCH" --loader_processes "$WORKERS" --bootstrap_subset "$SUBSET" \
    --checkpoint_base_dir "$CKPT_DIR" $EXTRA \
  2>&1 | tee "$LOG"
