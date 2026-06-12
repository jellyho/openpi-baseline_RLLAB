"""Full end-to-end smoke for the AQC adaptive pipeline: build a bundle, load the REAL RLT
model (2B backbone) + critic, and run one adaptive sample_actions on a fake observation.

Validates the heavy path that the mock-based smoke skipped: Pi0RLT(/Joint) load from orbax
params -> extract_token_and_base_actions (real backbone forward) -> decode -> critic prefix-Q
-> joint (n*, h*) selection -> both exec modes. Action VALUES are meaningless when the RLT
checkpoint doesn't match the critic's task (e.g. hanoi RLT + mouse-battery critic); this only
proves the pipeline runs without errors. Run on CPU (JAX_PLATFORMS=cpu) to avoid GPU contention.

Usage:
  JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_aqc_full.py \
    --rlt-config pi05_tower-of-hanoi-game_rlt_joint \
    --rlt-checkpoint checkpoints/pi05_tower-of-hanoi-game_rlt_joint/.../80000 \
    --critic-run-dir /.../rlt_critic_runs/vla_aqc_warmup/<run> \
    --num-samples 2 --num-flow-steps 2
"""

import argparse
import pathlib

import jax
import numpy as np

import openpi.training.config as _config
from openpi.rlt_critic import inference, merge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rlt-config", required=True)
    p.add_argument("--rlt-checkpoint", required=True, help="RLT step dir (contains params/)")
    p.add_argument("--critic-run-dir", required=True)
    p.add_argument("--critic-step", default="latest")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--num-flow-steps", type=int, default=2)
    p.add_argument("--bundle", default="/tmp/aqc_full_bundle")
    args = p.parse_args()

    rlt_params = pathlib.Path(args.rlt_checkpoint).resolve() / "params"
    print("=== [1] merge.build_bundle ===")
    bundle = merge.build_bundle(
        args.bundle, rlt_params, args.rlt_config, args.critic_run_dir, args.critic_step,
        num_action_samples=args.num_samples, num_flow_steps=args.num_flow_steps, overwrite=True)

    print("\n=== [2] AQCAdaptive.load (REAL RLT backbone + critic) ===")
    ada = inference.AQCAdaptive.load(bundle)
    print(f"  loaded. joint={ada._joint}  N={ada.num_action_samples} S={ada.num_flow_steps} "
          f"macro_H={ada.macro_H} decode={'real' if ada.decode else 'passthrough'}")

    print("\n=== [3] fake obs -> real RLT forward + critic + selection ===")
    tcfg = _config.get_config(args.rlt_config)
    obs = tcfg.model.fake_obs(batch_size=1)
    rng = jax.random.key(0)
    out = ada.sample_actions(rng, obs, exec_mode="truncate", num_samples=args.num_samples)
    print(f"  truncate:      actions {np.asarray(out['actions']).shape}  h*={out['h_star']}  "
          f"n*={out['n_star']}  q_best={out['q_best']:.4f}")
    print(f"  q_by_h: {np.round(np.asarray(out['q_by_h']), 4)}")
    out2 = ada.sample_actions(rng, obs, exec_mode="absolute_hold", num_samples=args.num_samples)
    print(f"  absolute_hold: actions {np.asarray(out2['actions']).shape}  h*={out2['h_star']}")

    assert np.asarray(out["actions"]).shape[0] == out["h_star"]
    assert np.asarray(out2["actions"]).shape[0] == ada.horizon
    print("\nFULL E2E OK: real 2B RLT forward -> critic prefix-Q -> (n*,h*) -> both exec modes")


if __name__ == "__main__":
    main()
