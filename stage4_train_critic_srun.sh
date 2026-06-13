#!/bin/bash
# ============================================================
#  Stage 4 (srun, FOREGROUND) — watch RLT/AQC critic training live.
#
#  Same training as stage4_train_critic_slurm.sh, but launched with `srun --pty`
#  so stdout (incl. the live tqdm progress bar) streams to YOUR terminal in real
#  time — no log-file tailing. Allocates on the pro6000 partition and runs in the
#  foreground.
#
#  ⚠️ Foreground = if your SSH session drops, the job DIES. For anything long, run
#     it inside tmux/screen:   tmux new -s rlt   ->   bash stage4_train_critic_srun.sh
#  ⚠️ Same config => SAME run_dir / W&B run as the sbatch job. If an rlt_seal* job is
#     already running, scancel it first OR set EXP=<name> here, or the two runs will
#     clobber each other's checkpoints. (Conversely: same config + --resume means
#     this srun run CONTINUES the sbatch run's checkpoints — handy after scancel.)
#
#  For a real 3-day unattended run, prefer sbatch (stage4_train_critic_slurm.sh) +
#  `tail -f logs/slurm_rlt_critic_<jobid>.out`  and the live W&B run page.
#
#  Usage:
#    bash stage4_train_critic_srun.sh                       # 2 GPU, seal v3, vla_aqc_warmup
#    GPUS=1 BATCH=256 LR=3e-4 bash stage4_train_critic_srun.sh
#    EXP=seal_srun_test TIME=2:00:00 bash stage4_train_critic_srun.sh
#
#  Env overrides:
#    GPUS   #GPUs on one node (default 2; BATCH must be divisible by it)
#    CONFIG TASK DATA_ROOT BATCH LR LOADER   (same meaning as the sbatch version)
#    TIME   srun walltime  (default 8:00:00; assoc max = 3-00:00:00)
#    EXP    exp_name override -> separate run_dir (avoid clobbering another run)
#    PART   partition list (default asus_pro6000,gigabyte_pro6000)
#    CACHE_DIR  cache + checkpoint base (default /lustre/jellyho)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# Caches + checkpoints on lustre; srun --export=ALL carries these to the compute node.
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh

GPUS="${GPUS:-1}"
CONFIG="${1:-${CONFIG:-vla_aqc_warmup}}"
TASK="${TASK:-seal-water-bottle-cap}"
DATA_ROOT="${DATA_ROOT:-/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3}"
BATCH="${BATCH:-256}"
LR="${LR:-3e-4}"
LOADER="${LOADER:-0}"   # 0 = in-process thread loader: fastest for base_action TD (avoids the
                        # 1.2GB/batch worker->main IPC). lp=0 -> 2.9 it/s vs lp=8 -> 1.9 it/s.
TIME="${TIME:-3-00:00:00}"
PART="${PART:-asus_pro6000,gigabyte_pro6000}"
CPUS="${CPUS:-$((GPUS * 8))}"
MEM_FRAC="${MEM_FRAC:-0.9}"

[ -d "$DATA_ROOT/data" ] || { echo "ERROR: no data/ under DATA_ROOT=$DATA_ROOT"; exit 1; }
[ $((BATCH % GPUS)) -eq 0 ] || { echo "ERROR: BATCH=$BATCH not divisible by GPUS=$GPUS"; exit 1; }

# Collision guard: warn if an rlt_seal* job is already queued/running (same run_dir risk).
OTHER=$(squeue -u "$USER" -h -o "%i %j %t" 2>/dev/null | grep -E "rlt_seal" || true)
if [ -n "$OTHER" ]; then
    echo "⚠️  rlt_seal job(s) already in queue/running:"
    echo "$OTHER" | sed 's/^/      /'
    echo "    Same CONFIG => SAME run_dir/W&B. scancel them, or set EXP=<name>, else they collide."
    echo "    Continuing in 8s — Ctrl-C to abort."
    sleep 8
fi

EXP_ARG=(); [ -n "${EXP:-}" ] && EXP_ARG=(--exp_name "$EXP")

echo "=== stage4 RLT/AQC critic (srun --pty, foreground) ==="
echo "gpus=$GPUS  cpus=$CPUS  time=$TIME  part=$PART"
echo "config=$CONFIG  task=$TASK  batch=$BATCH (-> $((BATCH/GPUS))/device)  lr=$LR  loader=$LOADER"
echo "ckpt_base=$RLT_CRITIC_CKPT_DIR   (run dir = <base>/$CONFIG/<run_name>)"
echo "(Ctrl-C stops the run; checkpoints every 25k steps survive for --resume.)"

exec srun \
    --partition="$PART" --qos=pro6000_qos \
    --nodes=1 --ntasks=1 --gres=gpu:"$GPUS" \
    --cpus-per-task="$CPUS" --mem=128G --time="$TIME" \
    --job-name=rlt_seal_srun --pty \
    env NCCL_P2P_DISABLE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
    uv run scripts/train_rlt_critic.py \
        --config "$CONFIG" \
        --task "$TASK" \
        --data_root "$DATA_ROOT" \
        --batch_size "$BATCH" \
        --lr "$LR" \
        --loader_processes "$LOADER" \
        --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        "${EXP_ARG[@]}" --resume
