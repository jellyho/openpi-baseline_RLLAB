#!/bin/bash
# Point HF / LeRobot / openpi caches at a real directory.
# NOTE: setting these to an empty string ("") does NOT unset them — it forces
# the libraries to write caches into the current working directory (the repo).
# Use the standard user cache location instead.

# export CACHE_DIR="${CACHE_DIR:-/NHNHOME/WORKSPACE/0526040008_A/jellyho}"
export CACHE_DIR="${CACHE_DIR:-/data5/jellyho}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR/.cache/openpi}"

# Fast HuggingFace transfers (parallel, Rust-based). ~23x faster uploads/downloads
# on this machine (~1.3 MB/s single-stream -> ~30 MB/s). Requires the `hf_transfer`
# package (installed in the openpi / .venv / posco envs).
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Persistent XLA compilation cache — avoids recompiling the (slow) train step on
# every run.  Keyed by (graph, shapes, flags, jax/xla version); change batch/model
# → new entry compiled once, then reused.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$CACHE_DIR/.cache/jax_compile}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

# Checkpoint output bases (global defaults; override per run by exporting these or
# passing the matching flag). Keep checkpoints off the small/quota'd home dir.
#   PI_CKPT_DIR         pi/VLA training (scripts/train.py, stages 1-2) — dir =
#                       <PI_CKPT_DIR>/<config>/<exp>/<step>. Passed by stage1_2_train.sh
#                       as --checkpoint-base-dir.  Default "./checkpoints" = openpi default.
#   RLT_CRITIC_CKPT_DIR RLT/AQC critic runs (stage4_train_critic.sh) — dir =
#                       <base>/<name>/<exp>.  Passed as --checkpoint_base_dir.
export PI_CKPT_DIR="${PI_CKPT_DIR:-./checkpoints}"
export RLT_CRITIC_CKPT_DIR="${RLT_CRITIC_CKPT_DIR:-$CACHE_DIR/PFR_RSS/checkpoints/rlt_critic_runs}"

# Machine-specific roots read by config.py / rlt_critic/config.py so the per-config
# paths don't have to be hand-edited when moving boxes. Override here per machine.
#   PFR_DATA       raw/merged/combined LeRobot datasets (local_files_path bases)
#   PFR_CKPT       pretrained checkpoints (the rss_ckpt/ pi05 bases the BC configs load)
#   RLT_DATA_BASE  the AQC critic's annotated datasets (per-task <task>_annotated)
export PFR_DATA="${PFR_DATA:-$CACHE_DIR/PFR_RSS/dataset}"
export PFR_CKPT="${PFR_CKPT:-$CACHE_DIR/PFR_RSS/checkpoints}"
export RLT_DATA_BASE="${RLT_DATA_BASE:-$PFR_DATA/phase1_annotated}"

echo "Environment variables set:"
echo "  HF_HOME:          $HF_HOME"
echo "  HF_LEROBOT_HOME:  $HF_LEROBOT_HOME"
echo "  OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "  JAX_COMPILATION_CACHE_DIR: $JAX_COMPILATION_CACHE_DIR"
echo "  HF_HUB_ENABLE_HF_TRANSFER: $HF_HUB_ENABLE_HF_TRANSFER"
echo "  PI_CKPT_DIR: $PI_CKPT_DIR"
echo "  RLT_CRITIC_CKPT_DIR: $RLT_CRITIC_CKPT_DIR"
echo "  PFR_DATA: $PFR_DATA"
echo "  PFR_CKPT: $PFR_CKPT"
echo "  RLT_DATA_BASE: $RLT_DATA_BASE"
