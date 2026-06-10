#!/bin/bash
# ============================================================
#  Annotate a LeRobot dataset with rl_token + base_action
#  using a JOINT RLT model (Pi0RLTJoint).
#
#  Difference vs annotate_rlt.sh (vanilla Pi0RLT):
#    The joint model's RL token comes from the image-token hidden
#    states of the SAME full pi_vla forward used for action sampling,
#    so compute_rl_tokens.py runs the 2B backbone ONCE per state
#    (token + base_action together) instead of twice.  That
#    single-forward path is auto-detected at runtime via
#    hasattr(model, "extract_token_and_base_actions") — no extra
#    flag needed; just point --config-name at a *_rlt_joint config.
#
#  DDP-style: one process per GPU over disjoint data files.
#  Bottleneck is GPU sampling (data loading is ~100x faster), so
#  scaling across GPUs + a big batch is what speeds this up.
# ============================================================
set -e
source setup_env.sh

CONFIG=pi05_generalist_rlt_joint
# Trained joint checkpoint STEP dir (must contain params/).  The joint config
# trains for 100k steps (save_interval 20k) -> last step is 99999.  Update if
# you annotate from an earlier checkpoint.
CKPT=/home/yonsei_jell/openpi-baseline_RLLAB/checkpoints/pi05_generalist_rlt_joint/pi05_generalist_rlt_joint/99999
SRC=/home/yonsei_jell/dualyam_combined                                          # v3.0 source (read-only)
OUT=/NHNHOME/WORKSPACE/0526040008_A/annotated/generalist_joint_annotated        # roomy Lustre disk (3.8TB free)

N=32          # base action samples per frame
BATCH=128     # raise as GPU memory allows (B200 = 183GB); bigger = faster
WORKERS=16    # per process (x4 = 64 cores)
GPUS=(0 1 2 3)
NUM_SHARDS=${#GPUS[@]}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

if [ ! -d "$CKPT/params" ]; then
    echo "ERROR: $CKPT/params not found. Train pi05_generalist_rlt_joint first, or fix CKPT." >&2
    exit 1
fi

# One-time copy of the dataset to the roomy disk (source stays untouched).
if [ ! -d "$OUT" ]; then
    echo "Copying $SRC -> $OUT ..."
    mkdir -p "$(dirname "$OUT")"
    cp -r "$SRC" "$OUT"
fi

# Launch one shard per GPU.
pids=()
for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$idx]}
    echo "shard $idx -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run scripts/compute_rl_tokens.py \
        --config-name "$CONFIG" --checkpoint "$CKPT" \
        --dataset-root "$OUT" \
        --num-shards "$NUM_SHARDS" --shard-index "$idx" \
        --num-action-samples "$N" --batch-size "$BATCH" --num-workers "$WORKERS" &
    pids+=($!)
done

# Wait for all shards; abort if any fails.
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "A shard failed — NOT registering features (dataset incomplete)." >&2
    exit 1
fi

# Register rl_token / base_action in meta/info.json once, after all shards finish.
uv run scripts/compute_rl_tokens.py \
    --config-name "$CONFIG" --checkpoint "$CKPT" \
    --dataset-root "$OUT" --num-action-samples "$N" \
    --register-features-only

echo "DONE -> $OUT"
