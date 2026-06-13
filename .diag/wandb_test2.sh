#!/bin/bash
# CORRECTED: no --timing_steps, so RunLogger (wandb) IS created (timing mode disables it).
# Tests whether wandb ONLINE (service process started after CUDA init) deadlocks the 2-GPU
# NCCL clique vs OFFLINE. Hang shows up as the XLA 'rendezvous ... stuck' message + no steps.
export CACHE_DIR="${CACHE_DIR:-/lustre/jellyho}"
source setup_env.sh
DATA=/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3
echo "=== node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

run() {
  name=$1; mode=$2
  echo ""; echo "########## WB2: $name  (WANDB_MODE=$mode, NO timing -> RunLogger active, loader=8, 2GPU) ##########"
  timeout 200 env WANDB_MODE="$mode" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      uv run scripts/train_rlt_critic.py \
        --config vla_aqc_warmup --task seal-water-bottle-cap --data_root "$DATA" \
        --batch_size 512 --loader_processes 8 \
        --exp_name "_diag_wb2_$name" --checkpoint_base_dir "$RLT_CRITIC_CKPT_DIR" \
        > /tmp/wb2.$$ 2>&1
  rc=$?
  echo "--- last 14 lines ---"; tail -14 /tmp/wb2.$$
  if grep -qE "step +[0-9]+/|[0-9.]+ it/s" /tmp/wb2.$$; then echo ">>> [$name] PROGRESS (trained past step 0)"
  elif grep -qiE "rendezvous|stuck|deadlock" /tmp/wb2.$$; then echo ">>> [$name] HANG (rendezvous stuck), rc=$rc"
  else echo ">>> [$name] NO-PROGRESS rc=$rc (likely hang at step 0)"; fi
}

run online  online
run offline offline
echo ""; echo "=== WB2 DONE ==="
