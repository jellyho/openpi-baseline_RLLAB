#!/bin/bash
# Wait for the insert+seal reprocess (srun job "phase1fi*") to finish, verify the
# expected episode counts, then convert ALL 3 phase1 datasets v2.1->v3.0 and push.
set -e
cd /data5/jellyho/PFR_RSS/openpi-baseline_RLLAB

MERGED=/data5/jellyho/PFR_RSS/dataset/phase1_merged
STAGE=/data5/jellyho/PFR_RSS/dataset/v30_stage

echo "[chain] waiting for reprocess (squeue job phase1fi*) ..."
while squeue -u jellyho 2>/dev/null | grep -q "phase1fi"; do sleep 20; done
echo "[chain] reprocess job gone; verifying counts"

ins=$(find "$MERGED/insert-mouse-battery" -name '*.parquet' 2>/dev/null | wc -l)
sea=$(find "$MERGED/seal-water-bottle-cap" -name '*.parquet' 2>/dev/null | wc -l)
tow=$(find "$MERGED/tower-of-hanoi-game" -name '*.parquet' 2>/dev/null | wc -l)
echo "[chain] parquet counts: insert=$ins (exp 1119)  seal=$sea (exp 579)  tower=$tow (exp 1507)"
if [ "$ins" -ne 1119 ] || [ "$sea" -ne 579 ] || [ "$tow" -ne 1507 ]; then
    echo "[chain] ABORT: unexpected counts"; exit 1
fi

# Re-stage symlinks (dirs were recreated) + drop processing sidecars.
mkdir -p "$STAGE/jellyho"
for t in insert-mouse-battery seal-water-bottle-cap tower-of-hanoi-game; do
    rm -rf "$MERGED/$t/.cache_meta"
    ln -sfn "$MERGED/$t" "$STAGE/jellyho/${t}_rl_224"
done

echo "[chain] starting v2.1->v3.0 convert + push (DTS-safe)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export JAX_PLATFORMS=cpu
./.venv/bin/python scripts/convert_push_phase1_v30.py
echo "[chain] ALL DONE"
