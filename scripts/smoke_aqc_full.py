"""Full end-to-end smoke for the AQC adaptive pipeline: build a bundle, load the REAL RLT
model (2B backbone) + critic, and run one adaptive sample_actions on a fake observation.

Validates the heavy path that the mock-based smoke skipped, for BOTH deployable flavors:
  * vanilla Pi0RLT     (a *_rlt config)       -> extract_rl_token + sample_base_actions (TWO
                                                  backbone forwards: image-only token + full sample)
  * Pi0RLTJoint        (a *_rlt_joint config) -> extract_token_and_base_actions (ONE forward)
then decode -> critic prefix-Q -> joint (n*, h*) selection -> both exec modes. The flavor is
auto-detected from the RLT config (inference branches on hasattr extract_token_and_base_actions),
so this same smoke proves either path just by pointing --rlt-config at a vanilla or joint config.

Action VALUES are meaningless when the RLT checkpoint doesn't match the critic's task (or with
--mock-critic); this proves the pipeline RUNS end-to-end without errors. Run on CPU
(JAX_PLATFORMS=cpu) to avoid GPU contention.

Usage:
  # real critic run dir (task-correct values):
  JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_aqc_full.py \
    --rlt-config pi05_tower-of-hanoi-game_rlt_joint \
    --rlt-checkpoint checkpoints/pi05_tower-of-hanoi-game_rlt_joint/.../80000 \
    --critic-run-dir /.../rlt_critic_runs/vla_aqc_warmup/<run> \
    --num-samples 2 --num-flow-steps 2

  # no critic available -> fabricate a random-init critic to exercise the pipeline
  # (e.g. to verify the VANILLA two-forward path with only an _rlt checkpoint on hand):
  JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_aqc_full.py \
    --rlt-config pi05_seal-water-bottle-cap_rlt \
    --rlt-checkpoint checkpoints/pi05_seal-water-bottle-cap_rlt/.../20000 \
    --mock-critic --num-samples 2 --num-flow-steps 2
"""

import argparse
import dataclasses
import json
import pathlib

import flax.serialization as fs
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _config
from openpi.rlt_critic import config as _ccfg
from openpi.rlt_critic import inference, merge
from openpi.rlt_critic.transformer import PrefixValue


def fabricate_mock_critic(run_dir: pathlib.Path, critic_config: str = "vla_aqc_warmup") -> pathlib.Path:
    """Write a minimal critic run dir (config.json + one random-init params.msgpack) so
    merge/inference can build a bundle WITHOUT a trained critic. Lets the smoke exercise the
    full adaptive path (incl. critic forward + (n*,h*) selection) for any RLT flavor when no
    real critic run is reachable. Values are random — pipeline check only."""
    cfg = _ccfg.get_config(critic_config)
    a, d, td = cfg.arch, cfg.dist, cfg.td
    net = PrefixValue(
        action_dim=cfg.action_dim, horizon=cfg.horizon,
        num_ensembles=a.num_ensembles, num_layers=a.num_layers,
        num_heads=a.num_heads, head_dim=a.head_dim, mlp_dim=a.mlp_dim,
        layer_norm=a.layer_norm, num_atoms=d.num_atoms,
        per_position_head=a.per_position_head,
        state_encoder_dims=tuple(a.state_encoder_dims), macro_group_size=td.macro_group_size)
    params = net.init(jax.random.key(0),
                      jnp.zeros((1, cfg.latent_dim)), jnp.zeros((1, cfg.horizon * cfg.action_dim)))
    step_dir = run_dir / "checkpoints" / "step_00000001"
    step_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (step_dir / "params.msgpack").write_bytes(fs.to_bytes(params))
    print(f"  fabricated mock critic ({critic_config}) -> {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rlt-config", required=True)
    p.add_argument("--rlt-checkpoint", required=True, help="RLT step dir (contains params/)")
    p.add_argument("--critic-run-dir", default=None,
                   help="trained critic run dir (config.json + checkpoints/). Omit with --mock-critic.")
    p.add_argument("--mock-critic", nargs="?", const="vla_aqc_warmup", default=None,
                   metavar="CRITIC_CONFIG",
                   help="fabricate a random-init critic from this config (default vla_aqc_warmup) "
                        "instead of using --critic-run-dir. Pipeline check only (random values).")
    p.add_argument("--critic-step", default="latest")
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--num-flow-steps", type=int, default=2)
    p.add_argument("--bundle", default="/tmp/aqc_full_bundle")
    args = p.parse_args()

    if args.mock_critic is not None:
        args.critic_run_dir = str(fabricate_mock_critic(
            pathlib.Path(args.bundle + "_mockcritic"), args.mock_critic))
    elif args.critic_run_dir is None:
        p.error("provide --critic-run-dir, or --mock-critic to fabricate one")

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
