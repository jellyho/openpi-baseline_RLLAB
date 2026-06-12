"""Stage 2 — merge a trained RLT/AQC critic with its Pi0RLT(/Joint) checkpoint into ONE
deployable bundle, loaded as a single model by ``openpi.rlt_critic.inference.AQCAdaptive``.

NON-BREAKING: this only READS existing checkpoints and WRITES a new bundle directory. It
does not modify any openpi model, the ``Policy``, or any existing checkpoint — so every
pre-existing model keeps working exactly as before.

Why a bundle (and not one orbax pytree): the VLA/RLT model is ``nnx`` (orbax ``params/``)
and the AQC critic is ``flax.linen`` (msgpack). Co-locating them in one directory with a
manifest is robust and framework-agnostic; the wrapper loads both and exposes a single
``sample_actions`` (adaptive replanning). Splicing the linen critic into the nnx pytree is
possible (``nnx_bridge.ToNNX``) but fragile and unnecessary for deployment.

Bundle layout::

    <bundle>/
      params/                 -> RLT model orbax params (symlink to the RLT ckpt's params/, or a copy)
      critic/params.msgpack      trained critic params (flax linen)
      critic/net.json            critic net hyperparameters (rebuilds PrefixValue + HL-Gauss)
      aqc_manifest.json          rlt_config_name, N samples, flow steps, exec defaults, dims

Usage::

    python -m openpi.rlt_critic.merge \
      --rlt-config pi05_insert-mouse-battery_rlt \
      --rlt-checkpoint /path/to/rlt/.../99999 \
      --critic-run-dir  /path/.../rlt_critic_runs/vla_aqc_warmup/<run> \
      --critic-step latest \
      --out /path/to/bundles/mouse_battery_aqc
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil


def _net_hparams_from_config_dict(cd: dict) -> dict:
    """Pull exactly the fields needed to rebuild the critic net from a saved config.json
    (``dataclasses.asdict(VLAAQCConfig)``)."""
    a, d, td = cd["arch"], cd["dist"], cd["td"]
    return {
        "action_dim": cd["action_dim"],
        "horizon": cd["horizon"],
        "latent_dim": cd["latent_dim"],
        "num_ensembles": a["num_ensembles"],
        "num_layers": a["num_layers"],
        "num_heads": a["num_heads"],
        "head_dim": a["head_dim"],
        "mlp_dim": a["mlp_dim"],
        "layer_norm": a["layer_norm"],
        "per_position_head": a["per_position_head"],
        "state_encoder_dims": list(a["state_encoder_dims"]),
        "macro_group_size": td["macro_group_size"],
        "num_atoms": d["num_atoms"],
        "v_min": d["v_min"],
        "v_max": d["v_max"],
        "hl_gauss_sigma_frac": d["hl_gauss_sigma_frac"],
    }


def _resolve_step(run_dir: pathlib.Path, step: str) -> pathlib.Path:
    ckpts = run_dir / "checkpoints"
    avail = sorted(int(p.name.split("_")[1]) for p in ckpts.glob("step_*") if p.is_dir())
    if not avail:
        raise FileNotFoundError(f"no step_* checkpoints under {ckpts}")
    s = avail[-1] if step in ("latest", "", None) else int(step)
    if s not in avail:
        raise FileNotFoundError(f"step {s} not in {avail}")
    return ckpts / f"step_{s:08d}"


def build_bundle(
    out_dir: str | os.PathLike,
    rlt_params_dir: str | os.PathLike,
    rlt_config_name: str,
    critic_run_dir: str | os.PathLike,
    critic_step: str = "latest",
    *,
    num_action_samples: int = 32,
    num_flow_steps: int = 10,
    copy_rlt: bool = False,
    overwrite: bool = False,
) -> pathlib.Path:
    """Assemble the deployable bundle. Returns the bundle path."""
    out = pathlib.Path(out_dir).resolve()
    rlt_params = pathlib.Path(rlt_params_dir).resolve()
    run_dir = pathlib.Path(critic_run_dir).resolve()

    if not rlt_params.exists():
        raise FileNotFoundError(f"RLT params dir not found: {rlt_params}")
    cfg_json = run_dir / "config.json"
    if not cfg_json.exists():
        raise FileNotFoundError(f"critic run config.json not found: {cfg_json}")
    step_dir = _resolve_step(run_dir, critic_step)
    critic_params = step_dir / "params.msgpack"
    if not critic_params.exists():
        raise FileNotFoundError(f"critic params.msgpack not found: {critic_params}")

    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists. Pass --overwrite.")
        shutil.rmtree(out)
    (out / "critic").mkdir(parents=True)

    # 1) RLT params: symlink (default; cheap) or copy (portable artifact).
    dst_params = out / "params"
    if copy_rlt:
        shutil.copytree(rlt_params, dst_params)
    else:
        os.symlink(rlt_params, dst_params)

    # 2) critic params + net hyperparameters.
    shutil.copy2(critic_params, out / "critic" / "params.msgpack")
    cd = json.loads(cfg_json.read_text())
    net = _net_hparams_from_config_dict(cd)
    (out / "critic" / "net.json").write_text(json.dumps(net, indent=2))

    # 3) manifest.
    manifest = {
        "rlt_config_name": rlt_config_name,
        "rlt_params": str(rlt_params),                 # provenance (even if copied)
        "critic_run": str(run_dir),
        "critic_step": int(step_dir.name.split("_")[1]),
        "num_action_samples": int(num_action_samples),
        "num_flow_steps": int(num_flow_steps),
        "action_dim": net["action_dim"],
        "horizon": net["horizon"],
        "macro_group_size": net["macro_group_size"],
        "default_exec_mode": "truncate",               # 'truncate' | 'absolute_hold'
    }
    (out / "aqc_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"bundle written: {out}")
    print(f"  rlt params : {'copied' if copy_rlt else 'symlink'} -> {rlt_params}")
    print(f"  critic     : step {manifest['critic_step']} ({run_dir.name})")
    print(f"  net        : emb{net['num_heads']*net['head_dim']}x{net['num_layers']}L "
          f"K{net['num_ensembles']} atoms{net['num_atoms']} macro{net['macro_group_size']}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output bundle directory")
    p.add_argument("--rlt-config", required=True, help="RLT TrainConfig name (e.g. pi05_insert-mouse-battery_rlt)")
    p.add_argument("--rlt-checkpoint", required=True, help="RLT checkpoint STEP dir (contains params/)")
    p.add_argument("--critic-run-dir", required=True, help="critic run dir (has config.json + checkpoints/)")
    p.add_argument("--critic-step", default="latest", help="step number or 'latest'")
    p.add_argument("--num-action-samples", type=int, default=32)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--copy-rlt", action="store_true", help="copy RLT params instead of symlink (portable)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    rlt_params = pathlib.Path(args.rlt_checkpoint).resolve() / "params"
    build_bundle(
        args.out, rlt_params, args.rlt_config, args.critic_run_dir, args.critic_step,
        num_action_samples=args.num_action_samples, num_flow_steps=args.num_flow_steps,
        copy_rlt=args.copy_rlt, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
