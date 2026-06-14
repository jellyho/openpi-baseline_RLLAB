#!/bin/bash
# ============================================================
#  Stage 4 (FASTEST) — train the RLT/AQC critic with TRUE multi-process DDP:
#  ONE OS process per GPU (jax.distributed), each loading + feeding only its own B/N shard.
#
#  Why this over stage4_train_critic_memmap.sh (single-process multi-GPU):
#    single-process funnels the WHOLE global batch's gather + H2D through one host loop,
#    so the 4 GPUs sit ~half-idle (util ~50%) waiting to be fed. Here each process drives
#    its own GPU with B/N -> the host pipeline parallelises N-way and GPU util goes ~100%.
#    XLA inserts the cross-process gradient all-reduce (NCCL); math is identical to a single
#    global-B step, so results match exactly.
#
#  Usage:
#    ./stage4_train_critic_ddp.sh [CONFIG] [GPUS] [BATCH]
#      CONFIG  VLAAQCConfig preset (default vla_aqc_warmup); its data_root IS the dataset
#      GPUS    comma list, one process each (default 0,1,2,3); BATCH must divide by #gpus
#      BATCH   GLOBAL batch size (default 1024) -> BATCH/#gpus per process
#
#  Env:
#    DATA_ROOT, MEMMAP_DIR, SUBSET, MEM_FRACTION, CKPT_DIR, EXTRA  -- as in the memmap script
#    PORT          jax.distributed coordinator port (default 29500)
#
#  Examples:
#    ./stage4_train_critic_ddp.sh vla_aqc_insert-mouse-battery 0,1,2,3 1024
#    EXTRA=--resume ./stage4_train_critic_ddp.sh vla_aqc_insert-mouse-battery
#  Background (overnight):
#    nohup setsid ./stage4_train_critic_ddp.sh vla_aqc_insert-mouse-battery >/dev/null 2>&1 </dev/null &
# ============================================================
set -e
source setup_env.sh

CONFIG="${1:-vla_aqc_warmup}"
GPUS="${2:-0,1,2,3,4,5,6,7}"
BATCH="${3:-4096}"
MEMMAP_DIR="${MEMMAP_DIR:-auto}"
SUBSET="${SUBSET:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.9}"
CKPT_DIR="${CKPT_DIR:-$RLT_CRITIC_CKPT_DIR}"
PORT="${PORT:-29500}"
EXTRA="${EXTRA:-}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
if (( BATCH % NGPU != 0 )); then
  echo "ERROR: global BATCH=$BATCH not divisible by #gpus=$NGPU"; exit 1
fi
mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/rlt_critic_${CONFIG}_ddp_${STAMP}.log"
echo "config=$CONFIG  DDP(multi-process) x$NGPU  gpus=$GPUS  global_batch=$BATCH (=$((BATCH/NGPU))/proc)  subset=$SUBSET"
echo "memmap=$MEMMAP_DIR  data_root=${DATA_ROOT:-<config default>}  ckpt=$CKPT_DIR  port=$PORT"
echo "chief log -> $LOG  (non-chief procs -> /dev/null)"

# 1) Pre-build the memmap ONCE (single process, no DDP) so the workers never race on it.
#    Instant no-op if it already exists.
echo "=== pre-build memmap (one-time, no-op if present) ==="
CUDA_VISIBLE_DEVICES="" \
uv run scripts/train_rlt_critic.py --config "$CONFIG" --memmap_dir "$MEMMAP_DIR" \
    ${DATA_ROOT:+--data_root "$DATA_ROOT"} --build_memmap_only \
  2>&1 | tee -a "$LOG"

# 2) Spawn one process per GPU. Each sees a single GPU (CUDA_VISIBLE_DEVICES) and its DDP rank
#    via RLT_* env. Chief (rank 0) -> tee to the log; others -> /dev/null.
echo "=== launch $NGPU DDP processes ==="
PIDS=()
for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  COMMON=( scripts/train_rlt_critic.py
    --config "$CONFIG" --memmap_dir "$MEMMAP_DIR"
    ${DATA_ROOT:+--data_root "$DATA_ROOT"}
    --batch_size "$BATCH" --loader_processes 0 --bootstrap_subset "$SUBSET"
    --checkpoint_base_dir "$CKPT_DIR" $EXTRA )
  if [ "$i" -eq 0 ]; then
    CUDA_VISIBLE_DEVICES="$GPU" RLT_NUM_PROCESSES="$NGPU" RLT_PROCESS_ID="$i" RLT_COORDINATOR="127.0.0.1:$PORT" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
    uv run "${COMMON[@]}" 2>&1 | tee -a "$LOG" &
  else
    CUDA_VISIBLE_DEVICES="$GPU" RLT_NUM_PROCESSES="$NGPU" RLT_PROCESS_ID="$i" RLT_COORDINATOR="127.0.0.1:$PORT" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
    uv run "${COMMON[@]}" >/dev/null 2>&1 &
  fi
  PIDS+=($!)
done

# Kill the whole group if any process dies or on Ctrl-C, so we never leak half a DDP job.
trap 'echo "[ddp] terminating"; kill ${PIDS[@]} 2>/dev/null' INT TERM
wait
