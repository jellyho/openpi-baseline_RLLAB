#!/bin/bash
# ============================================================
#  Stage 4 (srun, FOREGROUND) — FAST-ITERATION critic training on the 30-episode
#  seal_mini subset (10 success + 10 failure + 10 intervention).
#
#  Uses the `vla_aqc_mini` preset (src/openpi/rlt_critic/config.py): data_root +
#  task baked in, short beta schedule (warmup 5k + ramp 10k), frequent eval (every
#  1k steps) so the value-curve behaviour shows up in minutes. 1 GPU + thread loader
#  (loader_processes=0) — fastest for the base_action TD path; multi-GPU is pointless
#  here (the bottleneck is per-step base_action movement, GPU-count-independent).
#
#  Runs through `srun --pty` so the tqdm bar + eval lines stream live to your terminal.
#  (SSH drop kills it -> use tmux for longer runs.)
#
#  Usage:
#    bash stage4_mini_srun.sh                         # vla_aqc_mini, 1 GPU
#    EXP=g99_test bash stage4_mini_srun.sh            # separate run_dir
#    BATCH=512 LR=4e-4 bash stage4_mini_srun.sh
#
#  Env: CONFIG (default vla_aqc_mini) BATCH(256) LR(3e-4) TIME(2:00:00) EXP PART CACHE_DIR
#
#  NOTE: seal_mini is written one-episode-per-row-group, so an in-loader discount
#  sweep (td.mc_gamma) recomputes FULL-EPISODE mc_return correctly. To test gamma=0.99
#  add a preset variant (or --discount/--mc_gamma flags) -- ask and I'll wire it.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh

CONFIG="${1:-${CONFIG:-vla_aqc_mini}}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-4}"   # lowered from 3e-4 for TD stability (override with LR=...)
TIME="${TIME:-2:00:00}"
PART="${PART:-asus_pro6000,gigabyte_pro6000}"
EXTRA="${EXTRA:-}"                 # extra flags, e.g. EXTRA="--target_tau 0 --bootstrap_subset 8 --agg_beta 20"
DATA_ROOT="/lustre/jellyho/seal_mini"

[ -f "$DATA_ROOT/data/chunk-000/file-000.parquet" ] || { echo "ERROR: seal_mini not found at $DATA_ROOT (run .diag/build_mini.py)"; exit 1; }
EXP_ARG=(); [ -n "${EXP:-}" ] && EXP_ARG=(--exp_name "$EXP")

echo "=== stage4 MINI critic (srun --pty, 1 GPU, thread loader) ==="
echo "config=$CONFIG  batch=$BATCH  lr=$LR  data=$DATA_ROOT (30 eps)"
echo "ckpt_base=$RLT_CRITIC_CKPT_DIR"

exec srun \
    --partition="$PART" --qos=pro6000_qos \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=32 --mem=64G --time="$TIME" \
    --job-name=rlt_mini --pty \
    env NCCL_P2P_DISABLE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    uv run scripts/train_rlt_critic.py \
        --config "$CONFIG" \
        --batch_size "$BATCH" \
        --lr "$LR" \
        --loader_processes 0 \
        --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        "${EXP_ARG[@]}" $EXTRA --resume
