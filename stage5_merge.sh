#!/bin/bash
# ============================================================
#  Stage 5 — merge a trained RLT(/Joint) checkpoint + AQC prefix critic into ONE
#  deployable bundle (params/ + critic/ + aqc_manifest.json) for adaptive Q-chunking.
#
#  Works for BOTH flavors with the SAME command — just point RLT_CONFIG/RLT_CKPT at a
#  vanilla *_rlt (2 backbone forwards) or a *_rlt_joint (1 forward) checkpoint; merge
#  only records the config name and inference auto-detects the model class.
#
#  Usage:
#    ./stage5_merge.sh [RLT_CONFIG] [RLT_CKPT_STEPDIR] [CRITIC_RUN_DIR] [OUT_BUNDLE]
#      any omitted positional falls back to the default below (or its env var).
#
#  Env knobs:
#    CRITIC_STEP  critic checkpoint step ('latest' | <int>)   (default latest)
#    N            base-action samples (MATCH the annotation)   (default 32)
#    FLOW_STEPS   denoising steps for sampling                 (default 10)
#    COPY_RLT=1   copy RLT params into the bundle instead of symlinking (portable
#                 artifact to move to the deployment box; default: symlink)
#
#  After it finishes, deploy the bundle with the AQC adapter in policy_deployment_RLLAB:
#    --policy examples.openpi_aqc_policy:AQCPolicy  --policy-kwargs bundle_dir=<OUT>
# ============================================================
set -e
source setup_env.sh

# ---- defaults (edit to your paths, or pass as positionals / env) ----------
RLT_CONFIG="${1:-${RLT_CONFIG:-pi05_insert-mouse-battery_rlt}}"
RLT_CKPT="${2:-${RLT_CKPT:-/data5/jellyho/PFR_RSS/checkpoints/rllab_acrft/rlt/pi05_insert-mouse-battery_rlt}}"
CRITIC_RUN="${3:-${CRITIC_RUN:-/data5/gwanwoo/rss_pft/phase1/runs/insert-mouse-battery_a201_sup-fixed_emb384x4L_N32_P5_b256_g0.9999_mcfloor_s0}}"
OUT="${4:-${OUT:-/data5/jellyho/PFR_RSS/checkpoints/rllab_acrft/acrft/${RLT_CONFIG}_aqc_v1_MC}}"
CRITIC_STEP="${CRITIC_STEP:-latest}"
N="${N:-32}"
FLOW_STEPS="${FLOW_STEPS:-10}"

# Accept either a step dir (has params/) OR a params dir passed directly (no params/
# subfolder — the dir IS the orbax params, identified by manifest.ocdbt / _METADATA).
if [ ! -d "$RLT_CKPT/params" ] && [ ! -e "$RLT_CKPT/manifest.ocdbt" ] && [ ! -e "$RLT_CKPT/_METADATA" ]; then
    echo "ERROR: $RLT_CKPT is neither a step dir (with params/) nor an orbax params dir — train $RLT_CONFIG first, or fix RLT_CKPT." >&2
    exit 1
fi
# Tolerate pointing CRITIC_RUN at the inner checkpoints/ dir: config.json lives in the
# RUN dir (alongside checkpoints/), so step up one level when it's missing here.
if [ ! -f "$CRITIC_RUN/config.json" ] && [ -f "$(dirname "$CRITIC_RUN")/config.json" ]; then
    CRITIC_RUN="$(dirname "$CRITIC_RUN")"
fi
if [ ! -f "$CRITIC_RUN/config.json" ]; then
    echo "ERROR: $CRITIC_RUN/config.json not found — point CRITIC_RUN at the RUN dir (the one containing config.json + checkpoints/), not the inner checkpoints/." >&2
    exit 1
fi

EXTRA=()
[ -n "${COPY_RLT:-}" ] && EXTRA+=(--copy-rlt)

uv run python -m openpi.rlt_critic.merge \
    --rlt-config "$RLT_CONFIG" \
    --rlt-checkpoint "$RLT_CKPT" \
    --critic-run-dir "$CRITIC_RUN" \
    --critic-step "$CRITIC_STEP" \
    --num-action-samples "$N" \
    --num-flow-steps "$FLOW_STEPS" \
    --out "$OUT" \
    --overwrite \
    "${EXTRA[@]}"

echo "DONE -> bundle: $OUT"
echo "deploy: examples.openpi_aqc_policy:AQCPolicy  bundle_dir=$OUT   (policy_deployment_RLLAB)"
