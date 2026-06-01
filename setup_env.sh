#!/bin/bash
# Point HF / LeRobot / openpi caches at a real directory.
# NOTE: setting these to an empty string ("") does NOT unset them — it forces
# the libraries to write caches into the current working directory (the repo).
# Use the standard user cache location instead.

export CACHE_DIR="${CACHE_DIR:-/data5/jellyho}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_DIR/.cache/openpi}"

echo "Environment variables set:"
echo "  HF_HOME:          $HF_HOME"
echo "  HF_LEROBOT_HOME:  $HF_LEROBOT_HOME"
echo "  OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
