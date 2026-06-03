#!/bin/bash
# Convert the v2.1 phase1 merged datasets to v3.0 and push to the hub (with v3.0
# tag), then mark public. Datasets are staged as <STAGE>/jellyho/<task>_rl_224
# (symlinks to phase1_merged/<task>).
set -e
cd /data5/jellyho/PFR_RSS/openpi-baseline_RLLAB

STAGE=/data5/jellyho/PFR_RSS/dataset/v30_stage
PY=./.venv/bin/python
export HF_HUB_ENABLE_HF_TRANSFER=1
export JAX_PLATFORMS=cpu

for t in insert-mouse-battery tower-of-hanoi-game seal-water-bottle-cap; do
    REPO="jellyho/${t}_rl_224"
    echo "==================== convert+push $REPO ===================="
    $PY -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id "$REPO" --root "$STAGE" --push-to-hub true --force-conversion
    echo "---- set public: $REPO ----"
    $PY -c "from huggingface_hub import HfApi; HfApi().update_repo_settings(repo_id='$REPO', repo_type='dataset', private=False); print('PUBLIC', '$REPO')"
    # clean up the local _v30 temp build
    rm -rf "$STAGE/jellyho/${t}_rl_224_v30"
    echo "DONE $REPO -> https://huggingface.co/datasets/$REPO"
done
echo "ALL PHASE1 V30 CONVERT+PUSH DONE"
