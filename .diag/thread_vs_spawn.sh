#!/bin/bash
# Decisive: is the base_action cost DATA-MOVEMENT (worker->main IPC) or GPU COMPUTE?
# loader_processes=0 = in-process thread loader (NO IPC copy). If much faster than spawn=8,
# the 1.2GB/batch IPC was the bottleneck (free fix). If ~same, it's bootstrap compute (-> #3).
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  nproc=$(nproc) ==="

run() {
  name=$1; lp=$2; cvd=$3
  echo ""; echo "########## $name  (loader_processes=$lp, GPUs=$cvd, timing 120) ##########"
  timeout 240 env CUDA_VISIBLE_DEVICES="$cvd" NCCL_P2P_DISABLE=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config vla_aqc_warmup --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes "$lp" --timing_steps 120 \
        --exp_name "_diag_tvs_$name" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/tvs.$$ 2>&1
  rc=$?
  grep -E "=== timing:|=> 500k" /tmp/tvs.$$ || { echo "(no timing; rc=$rc)"; tail -5 /tmp/tvs.$$; }
}

run thread0_1gpu 0 0      # thread loader, 1 GPU (your setup)
run spawn4_1gpu  4 0      # few spawn workers, 1 GPU
run spawn8_1gpu  8 0      # ref (= ~2.0 it/s)
echo ""; echo "=== THREAD-VS-SPAWN DONE ==="
