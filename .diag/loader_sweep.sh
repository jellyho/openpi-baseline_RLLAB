#!/bin/bash
# Measure steady-state throughput (it/s) vs loader_processes to find where it plateaus.
# timing mode => excludes compile; logger off. 2 GPU, NCCL_P2P_DISABLE=1.
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  nproc=$(nproc)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

run() {
  lp=$1
  echo ""; echo "########## LOADER_PROCESSES=$lp  (2 GPU, batch 512, timing 120) ##########"
  timeout 260 env NCCL_P2P_DISABLE=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config vla_aqc_warmup --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes "$lp" --timing_steps 120 \
        --exp_name "_diag_lp$lp" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/ls.$$ 2>&1
  rc=$?
  grep -E "=== timing:|=> 500k" /tmp/ls.$$ || { echo "(no timing; rc=$rc)"; tail -5 /tmp/ls.$$; }
}

run 8
run 16
run 32
echo ""; echo "=== LOADER SWEEP DONE ==="
