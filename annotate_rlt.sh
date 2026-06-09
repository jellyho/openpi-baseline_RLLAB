#!/bin/bash
# ============================================================
#  Annotate a LeRobot dataset with rl_token + base_action,
#  DDP-style: one process per GPU over disjoint data files.
#
#  Bottleneck is GPU sampling (data loading is ~100x faster), so
#  scaling across GPUs + a big batch is what speeds this up.
# ============================================================
set -e
source setup_env.sh

CONFIG=pi05_seal-water-bottle-cap_rlt
CKPT=/data5/jellyho/PFR_RSS/openpi-baseline_RLLAB/checkpoints/pi05_seal-water-bottle-cap_rlt/pi05_seal-water-bottle-cap_rlt/99999
SRC=/data5/jellyho/.cache/huggingface/lerobot/jellyho/seal-water-bottle-cap_rl_224   # v3.0 source (read-only; HF copy)
OUT=/data5/jellyho/PFR_RSS/checkpoints/rss_ckpt/annotated/seal-water-bottle-cap_annotated   # roomy Lustre disk (4.5TB)

N=32          # base action samples per frame
BATCH=128     # raise as GPU memory allows (B200 = 183GB); bigger = faster
WORKERS=16     # per process (x4 = 32 cores)
GPUS=(0 1 2 7)
NUM_SHARDS=${#GPUS[@]}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# One-time copy of the dataset to the roomy disk (source stays untouched).
if [ ! -d "$OUT" ]; then
    echo "Copying $SRC -> $OUT ..."
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
