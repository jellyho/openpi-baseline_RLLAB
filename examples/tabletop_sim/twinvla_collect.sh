#!/bin/bash

cd /data5/jellyho/PFR_RSS/openpi-baseline_RLLAB
# CUDA_VISIBLE_DEVICES=1
export TWINVLA_PATH=/data5/jellyho/PFR_RSS/TwinVLA
export MUJOCO_GL=egl

python examples/tabletop_sim/collect_rollouts.py \
    --checkpoint jellyho/TwinVLA-aloha_handover_box \
    --tgt_dir /data5/jellyho/tabletop/rl_rollouts \
    --repo_id jellyho/aloha_handover_box_joint_pos_rl_new \
    --num_episodes 100 \
    --push_to_hub false