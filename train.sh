#!/bin/bash
# ============================================================
#  Training launcher
#
#  Usage:
#    ./train.sh <config> <num_gpus> <batch_per_gpu> <num_steps>
#
#  Example:
#    ./train.sh pi05_alphaflow_critic_tabletop 8 16 30000
#      → 8 GPUs, 16 samples/GPU (global batch 128), 30k steps
# ============================================================

source setup_env.sh

CONFIG=$1
NUM_GPUS=${2:-4}
BATCH_PER_GPU=${3:-32}
NUM_STEPS=${4:-7500}
NUM_WORKERS=${5:-64}

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config> <num_gpus> <batch_per_gpu> <num_steps>"
    exit 1
fi

GLOBAL_BATCH=$((NUM_GPUS * BATCH_PER_GPU))
# Under SLURM (srun --gres=gpu:N) the scheduler already sets CUDA_VISIBLE_DEVICES;
# only set it ourselves when running outside SLURM.
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

mkdir -p logs
LOG_FILE="logs/${CONFIG}_$(date +%Y%m%d-%H%M%S).log"

echo "config=$CONFIG  gpus=$NUM_GPUS  batch=$GLOBAL_BATCH ($BATCH_PER_GPU/gpu)  steps=$NUM_STEPS"

uv run scripts/train.py "$CONFIG" \
    --exp-name="$CONFIG" \
    --fsdp-devices="$NUM_GPUS" \
    --batch-size="$GLOBAL_BATCH" \
    --num-train-steps="$NUM_STEPS" \
    --num-workers="$NUM_WORKERS" \
    --overwrite 2>&1 | tee -a "$LOG_FILE"
