#!/bin/bash
# ============================================================
#  Critic visualization launcher
#
#  Renders an mp4 of one LeRobot trajectory with the camera view on the
#  left and the critic's per-step E[V] (vs ground-truth MC return) on the right.
#
#  Usage:
#    ./visualize_critic.sh <config> <checkpoint_dir> <repo_id> [episode] [output]
#
#  Example:
#    ./visualize_critic.sh \
#        pi05_alphaflow_critic_tabletop \
#        checkpoints/pi05_alphaflow_critic_tabletop/pi05_alphaflow_critic_tabletop/29999 \
#        jellyho/aloha_handover_box_joint_pos_rl_mc \
#        0
# ============================================================

source setup_env.sh

CONFIG=$1
CHECKPOINT=$2
REPO_ID=$3
EPISODE=${4:-0}
OUTPUT=${5:-data/critic_vis/${CONFIG}_ep${EPISODE}.mp4}

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$REPO_ID" ]; then
    echo "Usage: $0 <config> <checkpoint_dir> <repo_id> [episode] [output]"
    exit 1
fi

# Single GPU is plenty for inference.
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "config=$CONFIG  ckpt=$CHECKPOINT  repo=$REPO_ID  episode=$EPISODE  out=$OUTPUT"

uv run scripts/visualize_critic.py \
    --config="$CONFIG" \
    --checkpoint="$CHECKPOINT" \
    --repo-id="$REPO_ID" \
    --episode="$EPISODE" \
    --output="$OUTPUT"
