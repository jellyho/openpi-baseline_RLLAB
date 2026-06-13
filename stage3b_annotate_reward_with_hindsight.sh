#!/bin/bash
#SBATCH --job-name=reward_annotate
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/annotate_%x_%j.out
#SBATCH --error=logs/annotate_%x_%j.err
# ---------------------------------------------------------------------------
# Re-annotate reward/mc_return on the MERGED (original + augmented) dataset,
# one task per GPU node.  (No GPU is actually used; this is CPU/IO bound.)
#
#   TASK = insert-mouse-battery | seal-water-bottle-cap | tower-of-hanoi-game
#
# Usage (run once per node, one task each):
#   ./sbatch_annotate.sh insert-mouse-battery
#   ./sbatch_annotate.sh seal-water-bottle-cap
#   ./sbatch_annotate.sh tower-of-hanoi-game
# or under SLURM:
#   sbatch ./sbatch_annotate.sh insert-mouse-battery
#
# What it does (per task):
#   1. builds  <task>_merged_v3input/  = symlinks to the 4 original annotated
#      parquet files + the 1 augmented (intervention_failures) file.  Instant,
#      no data is copied; the originals are never modified.
#   2. runs reward_annotate.py over that merged set -> fresh JOINT v3
#      normalization (living=-1, fail=-C_fail, gamma=0.9999, then /Z global),
#      writing a full annotated copy to  <task>_annotated_v3_augmented/.
# ---------------------------------------------------------------------------
set -euo pipefail

TASK="${1:-${TASK:-insert-mouse-battery}}"
if [[ -z "${TASK}" ]]; then
  echo "ERROR: pass a task name, e.g. ./sbatch_annotate.sh insert-mouse-battery" >&2
  exit 1
fi

BASE_DIR="/NHNHOME/WORKSPACE/0526040008_A/jellyho"
ORIG="${BASE_DIR}/${TASK}_annotated"
AUG="${BASE_DIR}/${TASK}_annotated_v3"
MERGED="${BASE_DIR}/${TASK}_merged_v3input"
OUTPUT="${BASE_DIR}/${TASK}_annotated_v3_augmented"

PY="python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS=5    # 5 files in the merged set -> 5 parallel writers

echo "=================================================================="
echo " TASK    : ${TASK}"
echo " orig    : ${ORIG}"
echo " aug     : ${AUG}"
echo " merged  : ${MERGED}   (symlinks)"
echo " output  : ${OUTPUT}"
echo " started : $(date)"
echo "=================================================================="

# ---- sanity ----
for d in "${ORIG}/data/chunk-000" "${AUG}/data/chunk-000"; do
  [[ -d "${d}" ]] || { echo "ERROR: missing ${d}" >&2; exit 1; }
done

# ---- step 1: build the merged symlink dir (idempotent, instant) ----
M="${MERGED}/data/chunk-000"
mkdir -p "${M}"
ln -sf "${ORIG}/data/chunk-000/file-000.parquet" "${M}/file-000.parquet"
ln -sf "${ORIG}/data/chunk-000/file-001.parquet" "${M}/file-001.parquet"
ln -sf "${ORIG}/data/chunk-000/file-002.parquet" "${M}/file-002.parquet"
ln -sf "${ORIG}/data/chunk-000/file-003.parquet" "${M}/file-003.parquet"
ln -sf "${AUG}/data/chunk-000/file-000.parquet"  "${M}/file-004.parquet"
echo "[merge] ${M}:"
ls -l "${M}"

# ---- step 2: annotate (fresh JOINT v3 normalization over orig + augmented) ----
# --force: the augmented part is already normalized to mc_min=-1, which would
#          otherwise trip the 'already v3' skip guard; we want a fresh joint pass.
"${PY}" "${HERE}/reward_annotate.py" \
  --input  "${MERGED}" \
  --output "${OUTPUT}" \
  --workers "${WORKERS}" \
  --force

echo "=================================================================="
echo " DONE ${TASK}: ${OUTPUT}    finished: $(date)"
echo "=================================================================="
