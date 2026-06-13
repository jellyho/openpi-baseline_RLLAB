#!/bin/bash
# ============================================================
#  Stage 4 (SLURM, 4x RTXPRO6000) — train the RLT / AQC prefix critic.
#
#  This cluster's login node has no GPU, so stage 4 runs through SLURM. It wraps
#  scripts/train_rlt_critic.py exactly like stage4_train_critic.sh, but submits to
#  the pro6000 partition with 4 GPUs (single-process JAX data-parallel: the global
#  batch is split across all 4 local devices) and writes checkpoints/caches to
#  /lustre (home / /data5 are not writable here).
#
#  Default wiring: the seal-water-bottle-cap v3-annotated dataset + the
#  `vla_aqc_warmup` preset = the ReLU-blend MC-warmup ("beta scheduling") mode
#  (beta=0 for 20k steps, then a cosine ramp to 1 over 30k; see TDConfig in
#  src/openpi/rlt_critic/config.py). The v3 dataset already matches this config's
#  defaults (gamma=0.9999, value support [-1, 0]).
#
#  4-GPU sizing: global BATCH=1024 -> 256/device (= the 1-GPU per-device batch, so
#  4x throughput at the same step count). LR is bumped to 6e-4 (sqrt scaling for the
#  4x batch); set LR=3e-4 to keep the single-GPU LR. The model is tiny and the
#  pipeline is host-bound, so loader_processes is raised to 8 to feed 4 GPUs.
#
#  Usage:
#    sbatch stage4_train_critic_slurm.sh                         # seal v3 + vla_aqc_warmup, 4 GPU
#    sbatch stage4_train_critic_slurm.sh vla_aqc_warmup_softmax  # other preset
#    BATCH=512 LR=4e-4 sbatch stage4_train_critic_slurm.sh       # smaller global batch
#
#  Env overrides (export before sbatch, or pass via sbatch --export=ALL,VAR=...):
#    CONFIG     VLAAQCConfig preset       (default vla_aqc_warmup)
#    TASK       challenge task name       (default seal-water-bottle-cap)
#    DATA_ROOT  annotated dataset dir     (default the seal v3 path below)
#    BATCH      global batch size         (default 1024; must be divisible by 4)
#    LR         learning rate             (default 6e-4; 3e-4 = single-GPU default)
#    LOADER     loader_processes          (default 8)
#    CACHE_DIR  cache + checkpoint base   (default /lustre/jellyho)
#    MEM_FRAC   XLA GPU memory cap        (default 0.9; PREALLOCATE=false, grows only)
#
#  Interactive alternative (grab 4 GPUs on one node, then use the plain launcher):
#    srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=24 -p asus_pro6000 --pty bash
#    CACHE_DIR=/lustre/jellyho EXTRA="--task seal-water-bottle-cap --lr 6e-4 \
#      --data_root /lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3" \
#      bash stage4_train_critic.sh vla_aqc_warmup 1024
# ============================================================
#SBATCH --job-name=rlt_seal
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_rlt_critic_%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Keep caches + checkpoints on lustre (home / /data5 unwritable on this box).
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh

CONFIG="${1:-${CONFIG:-vla_aqc_warmup}}"
TASK="${TASK:-seal-water-bottle-cap}"
DATA_ROOT="${DATA_ROOT:-/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3}"
BATCH="${BATCH:-512}"
LR="${LR:-3e-4}"
LOADER="${LOADER:-0}"   # 0 = in-process THREAD loader. For base_action TD the 1.2GB/batch is the
                        # cost, and worker->main IPC dominates, so spawn workers are SLOWER:
                        # measured lp=0 -> 2.9 it/s vs lp=8 -> 1.9 it/s. (Raise only for the no-base
                        # MC path, where batches are tiny and multiprocess scales.)
MEM_FRAC="${MEM_FRAC:-0.9}"

[ -d "$DATA_ROOT/data" ] || { echo "ERROR: no data/ under DATA_ROOT=$DATA_ROOT"; exit 1; }
[ $((BATCH % 4)) -eq 0 ] || { echo "ERROR: BATCH=$BATCH not divisible by 4 GPUs"; exit 1; }

mkdir -p logs
echo "=== stage4 RLT/AQC critic (SLURM, 2x pro6000) ==="
echo "node      : $(hostname)   gpus: ${CUDA_VISIBLE_DEVICES:-?}"
echo "config    : $CONFIG   task: $TASK"
echo "batch     : $BATCH global (-> $((BATCH/4))/device)   lr: $LR   loader_procs: $LOADER"
echo "data_root : $DATA_ROOT"
echo "ckpt_base : $RLT_CRITIC_CKPT_DIR   (run dir = <base>/$CONFIG/<run_name>)"

# NCCL_P2P_DISABLE=1: these pro6000 GPUs have NO NVLink (topo=NODE, PCIe), so P2P-over-PCIe
# is the flaky path that deadlocks the 2-GPU clique rendezvous on some nodes. Disabling it
# loses nothing (no NVLink) and inits ~4x faster (SHM/host path). Required for reliable DDP.
NCCL_P2P_DISABLE=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
uv run scripts/train_rlt_critic.py \
    --config "$CONFIG" \
    --task "$TASK" \
    --data_root "$DATA_ROOT" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --loader_processes "$LOADER" \
    --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
    --resume
