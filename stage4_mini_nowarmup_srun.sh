#!/bin/bash
# ============================================================
#  EXP1 (current reward, config vla_aqc_mini) with NO MC WARMUP:
#  beta = 1 from step 0 -> target = max(MC, TD) (hard Cal-QL floor) immediately, no warmup ramp.
#
#  Thin wrapper over stage4_mini_srun.sh prepending `--mc_warmup_steps 0 --mc_ramp_steps 0`
#  (mc_warmup==mc_ramp==0 => beta jumps to 1 at step 0). run_name gets the `mcfloor` tag instead
#  of `warm5k+10k`, so this lands in its OWN run_dir (no collision with the warmup run).
#
#  Tests whether grounding the value with the MC-warmup phase actually matters, vs trusting the
#  (optimistic Best-of-N) bootstrap immediately. Expect this to be the LESS stable / more
#  overestimation-prone run -- the warmup exists precisely to avoid early bootstrap blow-up.
#
#  Usage:  bash stage4_mini_nowarmup_srun.sh
#          EXTRA="--n_step 10" bash stage4_mini_nowarmup_srun.sh   # combine with other flags
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export EXTRA="--mc_warmup_steps 0 --mc_ramp_steps 0${EXTRA:+ $EXTRA}"
echo "(EXP1 + no warmup, beta=1 from step 0)  config=vla_aqc_mini  EXTRA=$EXTRA"
exec bash stage4_mini_srun.sh vla_aqc_mini
