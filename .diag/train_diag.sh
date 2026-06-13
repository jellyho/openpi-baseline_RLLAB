#!/bin/bash
# Isolate the 2-GPU training hang: thread loader (loader_processes=0, no spawn workers)
# vs spawn loader (8). timing_steps => runs a few steps then exits. WANDB disabled.
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

run() {
  name=$1; lp=$2
  echo ""; echo "########## TRAIN-DIAG: $name  (loader_processes=$lp, 2 GPU, timing) ##########"
  timeout 280 env WANDB_MODE=disabled \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config vla_aqc_warmup --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes "$lp" --timing_steps 8 \
        --exp_name "_diag_$name" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/trd.$$ 2>&1
  rc=$?
  tail -20 /tmp/trd.$$
  if grep -q "=== timing:" /tmp/trd.$$; then echo ">>> [$name] PASS (steps ran)"; else echo ">>> [$name] FAIL/HANG rc=$rc"; fi
}

run thread 0
run spawn  8
echo ""; echo "=== TRAIN-DIAG DONE ==="
