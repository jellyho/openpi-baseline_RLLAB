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

if [ -z "$CONFIG" ]; then
    echo "Usage: $0"
    exit 1
fi

# Under SLURM (srun --gres=gpu:N) the scheduler already sets CUDA_VISIBLE_DEVICES;
# only set it ourselves when running outside SLURM.
# if [ -z "$SLURM_JOB_ID" ]; then
#     export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
# fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

mkdir -p logs
LOG_FILE="logs/${CONFIG}_$(date +%Y%m%d-%H%M%S).log"

uv run scripts/train.py "$CONFIG" \
    --exp-name="$CONFIG" \
    --fsdp-devices=1 \
    --overwrite
