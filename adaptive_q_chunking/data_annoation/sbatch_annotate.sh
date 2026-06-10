#!/bin/bash
# ============================================================
#  Reward relabeling + mc_return recomputation (CPU-only I/O job)
#
#  - Updates reward: living -0.0001 -> -0.0004
#  - Recomputes mc_return: gamma 0.995 -> 0.999
#  - 4 files processed in parallel (1 worker per file)
#  - GPU not needed; bottleneck is lustre read+write bandwidth
#
#  Estimated time: ~20 min (4 workers on 4 files simultaneously)
#  In-place: overwrites reward/mc_return in the original dataset (atomic).
# ============================================================
#SBATCH --job-name=reward_annotate
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/adaptive_q_chunking/data_annoation/logs/annotate_%j.out
#SBATCH --requeue

set -euo pipefail

# Absolute path (SLURM copies the batch script to a spool dir, so BASH_SOURCE
# would resolve there — hardcode the real location instead).
SCRIPT_DIR="/home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/adaptive_q_chunking/data_annoation"

echo "=== reward_annotate job ==="
echo "node    : $(hostname)"
echo "start   : $(date)"
echo "workers : 4 (one per parquet file)"
echo ""

# activate the venv used by this project
source /home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB/.venv/bin/activate

INPUT="/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/insert-mouse-battery_annotated"

# In-place: overwrites reward/mc_return in the original files (atomic temp+rename).
# Only the two scalar columns change; rl_token/base_action are preserved as-is.
python "${SCRIPT_DIR}/reward_annotate.py" \
    --input   "${INPUT}" \
    --inplace \
    --workers 4

echo ""
echo "done  : $(date)"
echo "output: ${INPUT}  (updated in place)"
