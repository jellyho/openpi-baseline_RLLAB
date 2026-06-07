"""
Augment a LeRobot dataset (v2.1) with two precomputed quantities from a trained
``Pi0RLT`` model, for the downstream actor-critic RL stage:

  1. **RL token** ``z_rl`` — a compact per-frame state representation (the
     encoder-decoder bottleneck of ``src/openpi/models/pi0_rlt.py``).  Stored as
     a ``rl_token`` column (dim ``rlt_token_dim``) in each episode parquet.

  2. **Base-VLA action chunks** — ``num_action_samples`` action chunks sampled
     per frame from the frozen base policy π_vla, shape ``[N, H, D]`` in
     normalized model space (the reference chunks ã the RLT actor conditions on /
     is regularized toward).  Stored as a per-episode sidecar
     ``base_actions/episode_{ep:06d}.npy`` (float16) — NOT a parquet column, since
     ``N·H·D`` floats per frame would bloat the parquet badly.

Both are computed in a SINGLE ordered pass over the dataset (one video decode).
Frame order is guaranteed (``shuffle=False``, ``drop_last=False``), so item ``i``
is global frame ``i`` == the episode-concatenation order of the parquets; the
script asserts ``sum(episode_lengths) == n_frames`` before writing.

Storage is streamed per episode (bounded memory): buffers fill in frame order and
flush at each episode boundary.

Notes
─────
  * The RL token's backbone forward is image-only (instruction is fixed), so the
    prompt does not affect ``z_rl``.  Base-action sampling uses the FULL π_vla
    prefix (images + language + state).
  * Default writes to a COPY (``--output``); pass ``--in-place`` to edit the
    source dataset directly.

Usage
─────
    uv run scripts/compute_rl_tokens.py \\
        --config-name pi05_seal-water-bottle-cap_rlt \\
        --checkpoint  /data5/.../pi05_seal-water-bottle-cap_rlt/<exp>/<step> \\
        --in-place
"""

import argparse
import json
import pathlib
import shutil

import flax.nnx as nnx
import jax
import numpy as np
import pandas as pd
import tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.data_loader import _collate_fn


def _build_dataset(config: _config.TrainConfig):
    """Transformed dataset (full training pipeline) + its DataConfig."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.local_files_path is None:
        raise ValueError("DataConfig.local_files_path is None — need a local dataset to write back to.")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    return dataset, data_config


def _load_model(config: _config.TrainConfig, checkpoint: pathlib.Path):
    """Load the trained RLT model from ``<checkpoint>/params`` (saved dtypes kept)."""
    params = _model.restore_params(checkpoint / "params")  # frozen backbone bf16, trained rlt_* fp32
    model = config.model.load(params)
    model.eval()
    return model


def _episode_paths(out: pathlib.Path):
    """(info dict, [episode parquet paths in episode-index order])."""
    info = json.loads((out / "meta" / "info.json").read_text())
    tmpl = info["data_path"]            # data/chunk-{episode_chunk}/episode_{episode_index}.parquet
    chunks_size = info["chunks_size"]
    n_episodes = info["total_episodes"]
    paths = [
        out / tmpl.format(episode_chunk=ep // chunks_size, episode_index=ep)
        for ep in range(n_episodes)
    ]
    return info, paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True, help="RLT TrainConfig name (e.g. pi05_seal-water-bottle-cap_rlt).")
    p.add_argument("--checkpoint", required=True, help="Trained RLT checkpoint step dir (contains params/).")
    p.add_argument("--output", default=None, help="Destination dataset root (omit when --in-place).")
    p.add_argument("--in-place", action="store_true", help="Edit the source dataset directly (no copy).")
    # what to compute
    p.add_argument("--rl-tokens", action="store_true", default=True, help="Compute the rl_token column.")
    p.add_argument("--no-rl-tokens", dest="rl_tokens", action="store_false")
    p.add_argument("--base-actions", action="store_true", default=True, help="Sample base-VLA action chunks.")
    p.add_argument("--no-base-actions", dest="base_actions", action="store_false")
    # params
    p.add_argument("--column-name", default="rl_token", help="Parquet column name for the RL token.")
    p.add_argument("--num-action-samples", type=int, default=64, help="Base action chunks sampled per frame (N).")
    p.add_argument("--num-flow-steps", type=int, default=10, help="Flow-matching denoising steps for sampling.")
    p.add_argument("--batch-size", type=int, default=8, help="Frames per batch (effective action batch = B·N).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.rl_tokens and not args.base_actions:
        raise ValueError("Nothing to do: both --no-rl-tokens and --no-base-actions set.")

    config = _config.get_config(args.config_name)
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    if not (checkpoint / "params").exists():
        raise FileNotFoundError(f"{checkpoint}/params not found — pass the checkpoint STEP dir.")

    # ── Dataset + model ───────────────────────────────────────────────────────
    dataset, data_config = _build_dataset(config)
    n_frames = len(dataset)
    print(f"Dataset: {data_config.repo_id} | frames: {n_frames} | root: {data_config.local_files_path}")

    model = _load_model(config, checkpoint)
    N, S = args.num_action_samples, args.num_flow_steps
    extract_token = nnx.jit(lambda m, obs: m.extract_rl_token(obs))
    sample_actions = nnx.jit(lambda m, rng, obs: m.sample_base_actions(rng, obs, num_samples=N, num_steps=S))

    # ── Resolve write target (source parquets live in local_files_path) ───────
    src = pathlib.Path(data_config.local_files_path).resolve()
    if args.in_place:
        out = src
        print(f"In-place: editing {out}")
    else:
        if not args.output:
            raise ValueError("--output is required unless --in-place is set.")
        out = pathlib.Path(args.output).resolve()
        if out.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out} exists. Pass --overwrite.")
            shutil.rmtree(out)
        print(f"Copying full dataset {src} → {out}")
        shutil.copytree(src, out)

    # ── Episode boundaries (frame order == episode concatenation order) ───────
    info, ep_paths = _episode_paths(out)
    lengths = [len(pd.read_parquet(pth, columns=[])) for pth in ep_paths]
    total = int(np.sum(lengths))
    if total != n_frames:
        raise RuntimeError(
            f"Sum of episode lengths ({total}) != dataset frame count ({n_frames}). "
            "Global frame ordering does not match the per-episode parquets; aborting to avoid misalignment."
        )

    base_dir = out / "base_actions"
    if args.base_actions:
        base_dir.mkdir(exist_ok=True)

    # ── Per-episode streaming flush (bounded memory) ──────────────────────────
    token_dim = {"val": None}

    def flush(ep: int, buf_z: list, buf_a: list):
        if args.rl_tokens:
            df = pd.read_parquet(ep_paths[ep])
            z_ep = np.asarray(buf_z, dtype=np.float32)               # [ep_len, D]
            df[args.column_name] = list(z_ep)
            df.to_parquet(ep_paths[ep])
            token_dim["val"] = z_ep.shape[1]
        if args.base_actions:
            a_ep = np.asarray(buf_a, dtype=np.float16)               # [ep_len, N, H, D]
            np.save(base_dir / f"episode_{ep:06d}.npy", a_ep)

    # ── Ordered single pass ───────────────────────────────────────────────────
    import torch  # local import (only needed here)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,       # global frame order == parquet concatenation order
        drop_last=False,     # cover EVERY frame (training loader drops the tail!)
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    rng = jax.random.key(args.seed)
    cur_ep, filled = 0, 0
    buf_z: list = []
    buf_a: list = []
    n_batches = (n_frames + args.batch_size - 1) // args.batch_size
    desc = "rl_token + base_actions" if (args.rl_tokens and args.base_actions) else (
        "rl_token" if args.rl_tokens else "base_actions")
    for step, batch in enumerate(tqdm.tqdm(loader, total=n_batches, desc=desc)):
        obs = _model.Observation.from_dict(batch)
        B = int(obs.state.shape[0])
        z = np.asarray(jax.device_get(extract_token(model, obs)), np.float32) if args.rl_tokens else None
        if args.base_actions:
            batch_rng = jax.random.fold_in(rng, step)
            a = np.asarray(jax.device_get(sample_actions(model, batch_rng, obs)), np.float16)  # [B,N,H,D]
        else:
            a = None
        for i in range(B):
            if args.rl_tokens:
                buf_z.append(z[i])
            if args.base_actions:
                buf_a.append(a[i])
            filled += 1
            if filled == lengths[cur_ep]:
                flush(cur_ep, buf_z, buf_a)
                cur_ep += 1
                filled = 0
                buf_z, buf_a = [], []
    assert cur_ep == len(ep_paths) and filled == 0, f"stream ended mid-episode (ep {cur_ep}, filled {filled})"

    # ── Register metadata in info.json ────────────────────────────────────────
    if args.rl_tokens:
        info["features"][args.column_name] = {"dtype": "float32", "shape": [token_dim["val"]], "names": None}
    if args.base_actions:
        # Sidecar (not a parquet feature): row r of episode_{ep}.npy is frame r's
        # [N, H, D] base-action chunks in normalized model space.
        info["base_actions"] = {
            "path": "base_actions/episode_{episode_index:06d}.npy",
            "num_samples": N,
            "action_horizon": int(config.model.action_horizon),
            "action_dim": int(config.model.action_dim),
            "dtype": "float16",
            "space": "normalized_model",
            "num_flow_steps": S,
        }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    msg = []
    if args.rl_tokens:
        msg.append(f"'{args.column_name}' (dim {token_dim['val']}) → parquets")
    if args.base_actions:
        msg.append(f"base_actions [N={N}, H={config.model.action_horizon}, D={config.model.action_dim}] fp16 → {base_dir}")
    print(f"Done @ {out}: " + " ; ".join(msg) + f" for {len(ep_paths)} episodes.")


if __name__ == "__main__":
    main()
