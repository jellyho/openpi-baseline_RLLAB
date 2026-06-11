#!/usr/bin/env bash
# Launch RLT/AQC critic training (src/openpi/rlt_critic) detached, so it survives the shell.
#
# Usage:
#   scripts/train_rlt_critic.sh [CONFIG] [BATCH] [GPU]
#     CONFIG  registry name (default: vla_aqc_warmup; see src/openpi/rlt_critic/config.py)
#     BATCH   global batch size (default: 64 — fits in ~7GB; raise when a GPU is free)
#     GPU     CUDA device index (default: 3)
#
# Env overrides:
#   MEM_FRACTION  XLA GPU memory cap (default 0.038 ≈ 7GB; raise toward 0.3+ on a free GPU)
#   WANDB         wandb mode (default offline; set "online" once logged in, or "disabled")
#   EXTRA         extra args passed through to train_rlt_critic.py (e.g. "--resume --seed 1")
#
# The critic is tiny (~10M) and trains on precomputed rl_token/base_action latents, so it
# fits alongside other jobs (PREALLOCATE=false: it only grows to the cap, and a cap-OOM
# fails THIS job, never the co-tenants). Checkpoints + metrics.csv land under the run dir
# (config.checkpoint_base_dir). Resume with EXTRA="--resume".
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-vla_aqc_warmup}"
BATCH="${2:-64}"
GPU="${3:-3}"
MEM_FRACTION="${MEM_FRACTION:-0.038}"
WANDB="${WANDB:-offline}"
EXTRA="${EXTRA:-}"

mkdir -p logs/rlt_critic
LOG="logs/rlt_critic/${CONFIG}_b${BATCH}_gpu${GPU}.log"
PIDFILE="logs/rlt_critic/${CONFIG}_b${BATCH}_gpu${GPU}.pid"

echo "launching: config=$CONFIG batch=$BATCH gpu=$GPU mem_frac=$MEM_FRACTION wandb=$WANDB"
echo "log -> $LOG"

nohup setsid env CUDA_VISIBLE_DEVICES="$GPU" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
  WANDB_MODE="$WANDB" \
  .venv/bin/python scripts/train_rlt_critic.py \
    --config "$CONFIG" --batch_size "$BATCH" --loader_processes 4 $EXTRA \
  > "$LOG" 2>&1 < /dev/null &

# $! is the setsid wrapper; the real python pid appears under it — record both for convenience.
sleep 2
REAL_PID="$(pgrep -f "train_rlt_critic.py --config $CONFIG --batch_size $BATCH" | head -1 || true)"
echo "${REAL_PID:-unknown}" > "$PIDFILE"
echo "python pid: ${REAL_PID:-unknown}  (pidfile: $PIDFILE)"
echo "tail -f $LOG   # to watch"
