#!/bin/bash
# Pin the ~2 it/s bottleneck: base_action (TD vs MC) and multi-GPU sharding (1 vs 2 GPU).
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  nproc=$(nproc)  alloc CUDA=$CUDA_VISIBLE_DEVICES ==="

run() {
  name=$1; cfg=$2; cvd=$3
  echo ""; echo "########## $name  (config=$cfg, GPUs=$cvd, batch 512, timing 120) ##########"
  timeout 220 env CUDA_VISIBLE_DEVICES="$cvd" NCCL_P2P_DISABLE=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config "$cfg" --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes 8 --timing_steps 120 \
        --exp_name "_diag_bn_$name" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/bn.$$ 2>&1
  rc=$?
  grep -E "=== timing:|=> 500k|devices:" /tmp/bn.$$ || { echo "(no timing; rc=$rc)"; tail -5 /tmp/bn.$$; }
}

run td_2gpu  vla_aqc_warmup 0,1     # reference (base_action TD, 2 GPU)  -> ~2 it/s expected
run td_1gpu  vla_aqc_warmup 0       # isolates multi-GPU sharding cost
run mc_2gpu  vla_mc           0,1   # NO base_action / NO bootstrap -> isolates base_action cost
echo ""; echo "=== BOTTLENECK DONE ==="
