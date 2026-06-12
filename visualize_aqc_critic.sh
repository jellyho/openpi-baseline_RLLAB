#!/bin/bash
# ============================================================
#  AQC prefix-critic visualization launcher (new critic).
#
#  Renders an mp4 of ONE annotated-dataset trajectory: camera on the left, and on
#  the right the critic's value curves (V_demo / V_adaptive vs GT MC return) plus
#  the adaptive commit length h*.  Reads the PRECOMPUTED rl_token + base_action
#  columns, so it needs only the small AQC critic (no VLA forward).
#
#  Usage:
#    ./visualize_aqc_critic.sh <critic_run_dir> <dataset_repo_id> [episode] [output] [critic_step]
#
#  Example:
#    ./visualize_aqc_critic.sh \
#        /data5/gwanwoo/rss_pft/phase1/runs/insert-mouse-battery_a201_sup-fixed_emb384x4L_N32_P5_b256_g0.9999_mcfloor_s0 \
#        jellyho/insert-mouse-battery_annotated \
#        0
# ============================================================
source setup_env.sh
CRITIC_RUN=$1
DATASET=$2
EPISODE=${3:-0}
OUTPUT=${4:-data/critic_vis/aqc_ep${EPISODE}.mp4}
CRITIC_STEP=${5:-latest}
CAM_KEY=${CAM_KEY:-observation.images.cam_high}
LOCAL_ROOT=${LOCAL_ROOT:-}

if [ -z "$CRITIC_RUN" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <critic_run_dir> <dataset_repo_id> [episode] [output] [critic_step]"
    exit 1
fi

if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

echo "critic_run=$CRITIC_RUN  dataset=$DATASET  episode=$EPISODE  step=$CRITIC_STEP  out=$OUTPUT"
EXTRA=()
[ -n "$LOCAL_ROOT" ] && EXTRA+=(--local-root "$LOCAL_ROOT")

uv run scripts/visualize_aqc_critic.py \
    --critic-run-dir "$CRITIC_RUN" \
    --dataset "$DATASET" \
    --episode "$EPISODE" \
    --output "$OUTPUT" \
    --critic-step "$CRITIC_STEP" \
    --cam-key "$CAM_KEY" \
    "${EXTRA[@]}"
