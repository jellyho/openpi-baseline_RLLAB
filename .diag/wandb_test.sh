#!/bin/bash
# Test whether WANDB online mode (its background service process started after CUDA init)
# is what deadlocks the 2-GPU NCCL clique. Compare online vs offline, loader=8, 2 GPU.
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

run() {
  name=$1; mode=$2
  echo ""; echo "########## WANDB-TEST: $name  (WANDB_MODE=$mode, loader=8, 2 GPU) ##########"
  timeout 200 env WANDB_MODE="$mode" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config vla_aqc_warmup --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes 8 --timing_steps 8 \
        --exp_name "_diag_wandb_$name" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/wbt.$$ 2>&1
  rc=$?
  tail -16 /tmp/wbt.$$
  if grep -q "=== timing:" /tmp/wbt.$$; then echo ">>> [$name] PASS"; else echo ">>> [$name] FAIL/HANG rc=$rc"; fi
}

run online  online
run offline offline
echo ""; echo "=== WANDB-TEST DONE ==="
