#!/bin/bash
# ============================================================
#  EXP1 (current reward, config vla_aqc_mini) with N-STEP = 100 return.
#
#  Thin wrapper over stage4_mini_srun.sh that prepends `--n_step 100`:
#    target(prefix h) = (h+100)-step realized return + gamma^(h+100) * V(s_{t+h+100}).
#  N=100 is a strong-conservatism extreme (the critic follows the demo for 100 steps before the
#  Best-of-N bootstrap) -> expect the value to drift toward the behaviour (demo) value and
#  prefix_spread to shrink. run_name auto-gets the `_n100` tag, so this lands in its OWN run_dir
#  (no collision with the N=0 run).
#
#  Usage:  bash stage4_mini_n100_srun.sh
#          BATCH=512 LR=1e-4 bash stage4_mini_n100_srun.sh    # overrides pass through
#          EXTRA="--bootstrap_subset 8" bash stage4_mini_n100_srun.sh   # add more flags
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export EXTRA="--n_step 100${EXTRA:+ $EXTRA}"
echo "(EXP1 + n_step=100)  config=vla_aqc_mini  EXTRA=$EXTRA"
exec bash stage4_mini_srun.sh vla_aqc_mini
