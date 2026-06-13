#!/bin/bash
# ============================================================
#  Stage 4 EXP2 (srun, FOREGROUND) — PROGRESS-reward critic on seal_mini_progress.
#
#  Uses the `vla_aqc_mini_progress` preset (src/openpi/rlt_critic/config.py): paper-style
#  task-progress target mc_return = gamma^(T-t)*I(success), support [0,1], explicit `done`
#  column. Universal schedule: pure-MC warmup (5k, no base_action -> ~44 it/s) then a HARD
#  switch to TD bootstrap (ramp=0) with the MC floor + EMA target net (tau=0.005).
#
#  preload=True (config default): the whole seal_mini_progress (~9 GB w/ fp16 base_action) is
#  decoded into RAM once at startup -> zero disk I/O / parquet re-decode. Single-process
#  thread loader (loader_processes=0) — fastest for the base_action TD path; preload requires
#  it (multiprocess workers would each duplicate the RAM cache).
#
#  Usage:
#    bash stage4_exp2_progress_srun.sh                 # vla_aqc_mini_progress, 1 GPU
#    EXP=run2 bash stage4_exp2_progress_srun.sh        # separate run_dir
#    BATCH=512 LR=1e-4 bash stage4_exp2_progress_srun.sh
#
#  Env: CONFIG(vla_aqc_mini_progress) BATCH(256) LR(1e-4) TIME(2:00:00) EXP PART CACHE_DIR EXTRA
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh

CONFIG="${1:-${CONFIG:-vla_aqc_mini_progress}}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-4}"
TIME="${TIME:-2:00:00}"
PART="${PART:-asus_pro6000,gigabyte_pro6000}"
EXTRA="${EXTRA:-}"                 # e.g. EXTRA="--preload False" or "--bootstrap_subset 8"
DATA_ROOT="/lustre/jellyho/seal_mini_progress"

[ -f "$DATA_ROOT/data/chunk-000/file-000.parquet" ] || { echo "ERROR: seal_mini_progress not found at $DATA_ROOT (run .diag/build_mini_progress.py)"; exit 1; }
EXP_ARG=(); [ -n "${EXP:-}" ] && EXP_ARG=(--exp_name "$EXP")

echo "=== stage4 EXP2 PROGRESS critic (srun --pty, 1 GPU, thread loader + preload) ==="
echo "config=$CONFIG  batch=$BATCH  lr=$LR  data=$DATA_ROOT"
echo "ckpt_base=$RLT_CRITIC_CKPT_DIR"

exec srun \
    --partition="$PART" --qos=pro6000_qos \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=32 --mem=96G --time="$TIME" \
    --job-name=rlt_exp2 --pty \
    env NCCL_P2P_DISABLE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    uv run scripts/train_rlt_critic.py \
        --config "$CONFIG" \
        --batch_size "$BATCH" \
        --lr "$LR" \
        --loader_processes 0 \
        --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        "${EXP_ARG[@]}" $EXTRA --resume
