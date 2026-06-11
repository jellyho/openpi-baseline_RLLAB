#!/usr/bin/env bash
# Auto-resuming supervisor for overnight UNATTENDED critic training.
# Restarts the run (with --resume, from the last 25k-step checkpoint) on any non-zero exit,
# with backoff and a retry cap. Exits cleanly when training reaches num_train_steps.
#
# Usage:   scripts/train_rlt_critic_supervised.sh [CONFIG] [BATCH] [GPU]
# Env:     MEM_FRACTION (0.038) | WANDB (offline) | MAX_RETRIES (50) | BACKOFF (60)
# Launch detached:
#   nohup setsid scripts/train_rlt_critic_supervised.sh vla_aqc_warmup 64 3 >/dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-vla_aqc_warmup}"; BATCH="${2:-64}"; GPU="${3:-3}"
MEM_FRACTION="${MEM_FRACTION:-0.038}"; WANDB="${WANDB:-offline}"
MAX_RETRIES="${MAX_RETRIES:-50}"; BACKOFF="${BACKOFF:-60}"

mkdir -p logs/rlt_critic
LOG="logs/rlt_critic/${CONFIG}_b${BATCH}_gpu${GPU}.log"
echo "$$" > "logs/rlt_critic/${CONFIG}_b${BATCH}_gpu${GPU}.supervisor.pid"

n=0
while :; do
  echo "[supervisor $(date '+%F %T')] launch (attempt $((n+1)), --resume) cfg=$CONFIG b=$BATCH gpu=$GPU" >> "$LOG"
  # --resume is a no-op when no checkpoint exists yet (starts fresh from step 0).
  CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
    WANDB_MODE="$WANDB" \
    .venv/bin/python scripts/train_rlt_critic.py \
      --config "$CONFIG" --batch_size "$BATCH" --loader_processes 4 --resume \
    >> "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "[supervisor $(date '+%F %T')] clean exit (training complete)" >> "$LOG"; break
  fi
  n=$((n+1))
  if [ $n -ge $MAX_RETRIES ]; then
    echo "[supervisor $(date '+%F %T')] exit code=$code; hit MAX_RETRIES=$MAX_RETRIES, giving up" >> "$LOG"; break
  fi
  echo "[supervisor $(date '+%F %T')] exit code=$code; resuming in ${BACKOFF}s (retry $n/$MAX_RETRIES)" >> "$LOG"
  sleep "$BACKOFF"
done
