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

echo "Environment variables set:"
echo "  HF_HOME:          $HF_HOME"
echo "  HF_LEROBOT_HOME:  $HF_LEROBOT_HOME"
echo "  OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "  JAX_COMPILATION_CACHE_DIR: $JAX_COMPILATION_CACHE_DIR"
echo "  HF_HUB_ENABLE_HF_TRANSFER: $HF_HUB_ENABLE_HF_TRANSFER"
