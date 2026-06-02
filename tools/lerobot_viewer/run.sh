#!/bin/bash
# Launch the LeRobot dataset web viewer.
#
# Usage:
#   ./run.sh [DATASET_ROOT] [PORT]
#
# Defaults: root=/data5/jellyho/PFR_RSS/dataset/phase1  port=7800
#
# If running on a remote server, forward the port from your laptop:
#   ssh -N -L 7800:localhost:7800 <user>@<host>
# then open http://localhost:7800
set -e

ROOT="${1:-/data5/jellyho/PFR_RSS/dataset/phase1}"
PORT="${2:-7800}"
PY=/data5/jellyho/miniconda3/envs/openpi/bin/python

cd "$(dirname "$0")"
exec "$PY" app.py --root "$ROOT" --port "$PORT"
