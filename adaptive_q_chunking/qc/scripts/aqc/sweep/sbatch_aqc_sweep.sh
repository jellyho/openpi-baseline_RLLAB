#!/bin/bash
#SBATCH --partition=big_suma_rtx3090,suma_rtx4090,base_suma_rtx3090,suma_a6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=node16,node19,node08,node31,node04,node05,node18
#
# AQC sweep array worker. Reads its arg-line from the manifest produced by
# submit_aqc_sweep.py (line number = SLURM_ARRAY_TASK_ID + 1) and runs main.py.

JOBS_FILE="$1"
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
PY_ARGS=$(sed -n "${LINE_NO}p" "$JOBS_FILE")

echo "=========================================================="
echo "Array task : ${SLURM_ARRAY_TASK_ID}  (line ${LINE_NO})"
echo "Node       : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "Manifest   : ${JOBS_FILE}"
echo "Args       : ${PY_ARGS}"
echo "=========================================================="

if [ -z "${PY_ARGS}" ]; then
  echo "ERROR: empty args for task ${SLURM_ARRAY_TASK_ID}"; exit 1
fi

cd /home/gwanwoo13/projects/rss_ptf/adaptive_q_chunking/qc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deas_real
unset LD_LIBRARY_PATH          # use the env's bundled CUDA wheels (fixes cuSPARSE-not-found)
export MUJOCO_GL=egl

# Absolute interpreter path (robust if `conda activate`'s PATH change doesn't stick on a node).
/home/gwanwoo13/miniconda3/envs/deas_real/bin/python main.py ${PY_ARGS}
