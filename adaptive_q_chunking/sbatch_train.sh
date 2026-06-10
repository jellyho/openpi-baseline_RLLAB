#!/bin/bash
# ============================================================
#  VLA AQC critic training (GPU).
#  Usage:
#    sbatch sbatch_train.sh                              # default config
#    sbatch sbatch_train.sh vla_aqc_td_macro             # named config
#    sbatch sbatch_train.sh vla_aqc_td_macro --timing_steps 100   # throughput probe
#    sbatch sbatch_train.sh vla_aqc_td_macro --resume
# ============================================================
#SBATCH --job-name=aqc_vla
#SBATCH --gres=gpu:1
#SBATCH --partition=big_suma_rtx3090,suma_rtx4090,suma_a6000,base_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=node16,node19,node08,node31,node04,node05,node18
#SBATCH --output=/home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/adaptive_q_chunking/logs/train_%j.out
#SBATCH --requeue

set -euo pipefail

CONFIG="${1:-vla_aqc_td_macro}"
shift || true
EXTRA="$@"                                  # e.g. --timing_steps 100 / --resume / --seed 1

# Auto-resume across requeues (preemption / node failure): the train code resumes from the
# last checkpoint if --resume is passed and checkpoints exist (else fresh start).
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    EXTRA="${EXTRA} --resume"
    echo "(requeue #${SLURM_RESTART_COUNT} -> adding --resume)"
fi

DIR=/home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/adaptive_q_chunking

echo "=== aqc_vla train ==="
echo "node   : $(hostname)"
echo "start  : $(date)"
echo "config : ${CONFIG}   extra: ${EXTRA}"

source /home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/.venv/bin/activate

# Confirm JAX sees the GPU (CPU fallback would be ~100x slower — fail loudly instead).
python - <<'PY'
import jax
devs = jax.devices()
print("jax devices:", devs)
assert any(d.platform == "gpu" for d in devs), "NO GPU visible to JAX — aborting (would run on CPU)"
PY

cd "${DIR}"
python vla_train.py --config "${CONFIG}" ${EXTRA}

echo "end    : $(date)"
