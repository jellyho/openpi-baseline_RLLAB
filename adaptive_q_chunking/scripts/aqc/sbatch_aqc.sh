#!/bin/bash
#SBATCH --partition=big_suma_rtx3090,suma_rtx4090,base_suma_rtx3090,suma_a6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --exclude=node16,node19,node08,node31,node04,node05,node18
#
# Adaptive Q-Chunking (ACSAC) SLURM launcher — mirrors the working sbatch_acfql.sh setup
# (deas_real env, big_qos, node excludes, unset LD_LIBRARY_PATH, MUJOCO_GL=egl) for the AQC
# agent in adaptive_q_chunking/qc.
#
# Array-job mapping: task_id -> (env, seed), exactly like sbatch_acfql.sh.
#
# Usage (array job, one task per env x seed):
#   N_ENVS=...; N_SEEDS=...; total=$((N_ENVS * N_SEEDS))
#   sbatch --array=0-$((total-1)) scripts/aqc/sbatch_aqc.sh \
#       aqc <run_group> "<env1> <env2> ..." "<seed1> <seed2> ..." <N> <sparse>
# Example:
#   sbatch --array=0-4 scripts/aqc/sbatch_aqc.sh aqc aqc_cube_double \
#       "cube-double-play-singletask-task1-v0 cube-double-play-singletask-task2-v0" "0 1" 4 False

# 1. args
AGENT_NAME="$1"          # must be "aqc"
RUN_GROUP="$2"
IFS=' ' read -r -a ENVS_LIST <<< "$3"
IFS=' ' read -r -a SEEDS <<< "$4"
N_SAMPLES="${5:-4}"      # rejection-sampling size N (paper: 4; cube-quadruple: 8)
SPARSE="${6:-False}"     # True for scene / puzzle domains

if [ "$AGENT_NAME" != "aqc" ]; then
    echo "Error: AGENT_NAME mismatch. Expected aqc, got $AGENT_NAME"
    exit 1
fi

# 2. array task -> (env, seed)
NUM_SEEDS=${#SEEDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_IDX=$((TASK_ID / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))
CURRENT_ENV=${ENVS_LIST[$ENV_IDX]}
CURRENT_SEED=${SEEDS[$SEED_IDX]}

# 3. per-domain dataset streaming (large datasets) + paper's N override
EXTRA_ARGS=()
if [[ "$CURRENT_ENV" == *"cube-quadruple"* ]]; then
    EXTRA_ARGS+=(--ogbench_dataset_dir="/home/gwanwoo13/.ogbench/data/cube-quadruple-play-100m-v0")
    N_SAMPLES=8   # Table 3: cube-quadruple uses N=8
elif [[ "$CURRENT_ENV" == *"cube-triple"* ]]; then
    EXTRA_ARGS+=(--ogbench_dataset_dir="/home/gwanwoo13/.ogbench/data/cube-triple-play-100m-v0")
fi

echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID:-NA}   Array Task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "Env: $CURRENT_ENV   Seed: $CURRENT_SEED   Agent: $AGENT_NAME"
echo "N (num_action_samples): $N_SAMPLES   Sparse: $SPARSE"
echo "=========================================================="

# 4. run
cd /home/gwanwoo13/projects/rss_ptf/adaptive_q_chunking/qc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deas_real
unset LD_LIBRARY_PATH        # use the env's bundled CUDA wheels (fixes cuSPARSE-not-found)
export MUJOCO_GL=egl

PY_ARGS=(
    main.py
    --agent="agents/aqc.py"
    --wandb_entity="${WANDB_ENTITY:-gwanwoo-yonsei-university}"
    --wandb_project="${WANDB_PROJECT:-AQC}"
    --seed=$CURRENT_SEED
    --run_group=$RUN_GROUP
    --env_name=$CURRENT_ENV
    --horizon_length=5
    --agent.num_action_samples=$N_SAMPLES
    --agent.adaptive_chunking=True
    --sparse=$SPARSE
    --offline_steps=1000000
    --online_steps=1000000
    --eval_interval=100000
    --eval_episodes=50
    --video_episodes=1
    --save_interval=100000
)

python "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}"
