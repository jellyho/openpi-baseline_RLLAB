#!/bin/bash
# ============================================================
#  Stage 3b — reward v3 + Monte-Carlo return annotation (CPU).
#
#  Adds the reward / mc_return / unnormalized_reward columns the AQC critic reads:
#  living -1/step, success terminal 0, failure terminal -0.4*T_max; gamma=0.9999
#  return-to-go; globally normalized so mc_return in [-1, 0].  Idempotent: skips a
#  dataset already annotated as v3.  Independent of the RLT pass (stage 3a) — run
#  either order on the same dataset root.
#
#  Usage:
#    ./stage3b_stage3b_annotate_reward.sh <DATASET_ROOT> [WORKERS]
#      WORKERS  parallel episode workers (default 4)
#
#  Dry run (design summary, no writes):  DRY_RUN=1 ./stage3b_stage3b_annotate_reward.sh <DATASET_ROOT>
# ============================================================
set -e
source setup_env.sh

ROOT="${1:?usage: ./stage3b_stage3b_annotate_reward.sh <DATASET_ROOT> [WORKERS]}"
WORKERS="${2:-4}"

EXTRA=()
if [ -n "${DRY_RUN:-}" ]; then
    EXTRA+=(--dry_run)
else
    EXTRA+=(--inplace)
fi

uv run python adaptive_q_chunking/data_annoation/reward_annotate.py \
    --input "$ROOT" --workers "$WORKERS" "${EXTRA[@]}"

[ -z "${DRY_RUN:-}" ] && echo "DONE (reward v3) -> $ROOT"
