#!/bin/bash
# ============================================================
#  Training launcher
#
#  Usage:
#    ./stage1_2_train.sh <config>
#
#  Examples:
#    ./stage1_2_train.sh pi05_insert-mouse-battery_bc_ft    # stage 1 (BC)
#    ./stage1_2_train.sh pi05_generalist_rlt_joint          # stage 2 (RLT)
#
#  Checkpoints go to $PI_CKPT_DIR/<config>/<config>/<step>.  $PI_CKPT_DIR is the
#  global default from setup_env.sh; override per run with
#    PI_CKPT_DIR=/path/to/checkpoints ./stage1_2_train.sh <config>
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
    --checkpoint-base-dir="$PI_CKPT_DIR" \
    --fsdp-devices=1 \
    --resume
