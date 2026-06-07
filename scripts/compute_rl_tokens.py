"""
Augment a LeRobot dataset (v3.0) with two precomputed quantities from a trained
``Pi0RLT`` model, for the downstream actor-critic RL stage.  BOTH are stored as
standard LeRobot feature columns (registered in ``meta/info.json``), so they load
through the normal pipeline and are accessible as ``item["rl_token"]`` /
``item["base_action"]`` via ``LeRobotDataset``.

  1. **RL token** ``z_rl`` — a compact per-frame state representation (the
     encoder-decoder bottleneck of ``src/openpi/models/pi0_rlt.py``).  Stored as a
     1-D ``rl_token`` column (``Sequence[rlt_token_dim]``, float32).

  2. **Base-VLA action chunks** — ``num_action_samples`` (N) action chunks sampled
     per frame from the frozen base policy π_vla.  Stored as a ``base_action``
     column (``Array3D[N, H, 14]``, float16) in **raw original action space** — the
     SAME space as the dataset's ``action`` column (un-normalized, absolute, yam
     14-dim).  At load time, normalize it with the dataset's *action* stats to put
     it back in model space, exactly like the ``action`` column.

Both columns are written back per DATA FILE (v3.0 packs many episodes per
size-capped parquet), keyed by each file's ACTUAL row count.  Each file covers a
contiguous global frame range, so the work parallelizes cleanly across GPUs.

Multi-GPU (DDP-style data parallel)
───────────────────────────────────
The GPU sampling is the bottleneck (data loading is ~100× faster), so split the
data files across processes.  Pre-copy the dataset to a roomy disk once, then run
one process per GPU on a disjoint shard, and register the features once at the end:

    OUT=/big/disk/insert-mouse-battery_annotated
    cp -r /home/yonsei_jell/insert-mouse-battery "$OUT"     # once
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i uv run scripts/compute_rl_tokens.py \\
        --config-name pi05_insert-mouse-battery_rlt --checkpoint <ckpt> \\
        --dataset-root "$OUT" --num-shards 4 --shard-index $i \\
        --num-action-samples 32 --batch-size 128 --num-workers 8 &
    done; wait
    uv run scripts/compute_rl_tokens.py --config-name pi05_insert-mouse-battery_rlt \\
        --checkpoint <ckpt> --dataset-root "$OUT" --num-action-samples 32 \\
        --register-features-only

Single GPU: omit --num-shards/--shard-index (defaults to the whole dataset) and the
features are registered automatically at the end.

Notes
─────
  * The RL token's backbone forward is image-only (the prompt does not affect z_rl);
    base-action sampling uses the FULL π_vla prefix (images + language + state).
  * Sampled chunks come out in normalized model space; they are pushed back to raw
    action space through the SAME canonical output transform the policy uses
    (Unnormalize → AbsoluteActions → YamOutputs), vectorized over N.
  * Per-file RNG (fold_in by file & step) makes results identical regardless of how
    files are sharded across GPUs.
"""

import argparse
import json
import os
import pathlib
import shutil

import flax.nnx as nnx
import jax
import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from openpi.training.data_loader import _collate_fn


def _make_data_config(config: _config.TrainConfig, root_override: str | None):
    data_config = config.data.create(config.assets_dirs, config.model)
    if root_override is not None:
        import dataclasses
        data_config = dataclasses.replace(data_config, local_files_path=root_override)
    if data_config.local_files_path is None:
        raise ValueError("DataConfig.local_files_path is None — need a local dataset to write back to.")
    return data_config


def _build_dataset(config: _config.TrainConfig, data_config):
    """Transformed dataset (full training pipeline) for the given data_config."""
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    return dataset


def _load_model(config: _config.TrainConfig, checkpoint: pathlib.Path):
    """Load the trained RLT model from ``<checkpoint>/params`` (saved dtypes kept)."""
    params = _model.restore_params(checkpoint / "params")  # frozen backbone bf16, trained rlt_* fp32
    model = config.model.load(params)
    model.eval()
    return model


def _build_action_decoder(data_config):
    """Vectorized normalized-model-space → raw-action-space converter for sampled chunks.

    Reuses the EXACT canonical output transform the policy applies (Unnormalize →
    AbsoluteActions → YamOutputs), so the stored ``base_action`` lives in the same
    space as the dataset's raw ``action`` column.  AbsoluteActions needs the state
    and broadcasts over a [.., H, D] batch; the remaining (dim-reducing) outputs run
    row-wise, so we reshape to 2-D for them.
    """
    if data_config.norm_stats is None:
        raise ValueError("DataConfig.norm_stats is None — cannot un-normalize sampled actions.")
    unnorm = _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    outs = list(data_config.data_transforms.outputs)

    def decode(sampled: np.ndarray, state: np.ndarray) -> np.ndarray:
        # sampled: [B, N, H, Dm] (normalized model space); state: [B, Dm] (normalized).
        b, n, h, dm = sampled.shape
        acts = sampled.reshape(b * n, h, dm).astype(np.float32)
        st = np.repeat(state.astype(np.float32), n, axis=0)            # [B*N, Dm]
        d = unnorm({"state": st, "actions": acts})
        # State-dependent transforms (AbsoluteActions) act on the [.., H, Dm] batch.
        i = 0
        while i < len(outs) and isinstance(outs[i], _transforms.AbsoluteActions):
            d = outs[i](d)
            i += 1
        # Remaining transforms (e.g. YamOutputs: slice to 14 dims, un-adapt) are row-wise.
        flat = d["actions"].reshape(-1, dm)
        for t in outs[i:]:
            flat = t({"actions": flat})["actions"]
        return flat.reshape(b, n, h, -1)                              # [B, N, H, 14]

    return decode


def _data_files(out: pathlib.Path):
    """(ordered data-parquet paths, row counts, global start offsets).

    Files are globally ordered by (chunk_index, file_index), matching global frame
    order; each file covers the contiguous range [offset, offset+rows).
    """
    import pyarrow.parquet as pq

    paths = sorted((out / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquets under {out/'data'}.")
    rows = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    offsets = np.concatenate([[0], np.cumsum(rows)])[:-1].tolist()
    return paths, rows, offsets


def _write_data_file(path: pathlib.Path, rl_col, z_arr, ba_col, ba_arr):
    """Add the rl_token / base_action columns to one data parquet (atomic replace).

    Written via HF ``datasets`` so ``base_action`` is encoded as a real Array3D
    feature that ``Dataset.from_parquet(features=...)`` reads back with shape.
    """
    import datasets

    ds = datasets.Dataset.from_parquet(str(path))
    n = len(ds)
    # Drop any pre-existing copies of the columns we're (re)writing (e.g. --overwrite
    # over an already-annotated file), so the concat below can't collide.
    drop = [c for c, arr in [(rl_col, z_arr), (ba_col, ba_arr)] if arr is not None and c in ds.column_names]
    if drop:
        ds = ds.remove_columns(drop)
    # Build the new columns as a separate Dataset with explicit feature types
    # (from_dict honours Array3D/Sequence; add_column/pyarrow cannot encode >1-D
    # arrays), then glue them on by columns.
    extra, feats = {}, {}
    if z_arr is not None:
        if len(z_arr) != n:
            raise RuntimeError(f"{path}: {len(z_arr)} rl_tokens != {n} rows (alignment).")
        extra[rl_col] = list(z_arr)
        feats[rl_col] = datasets.Sequence(length=z_arr.shape[1], feature=datasets.Value("float32"))
    if ba_arr is not None:
        if len(ba_arr) != n:
            raise RuntimeError(f"{path}: {len(ba_arr)} base_actions != {n} rows (alignment).")
        extra[ba_col] = list(ba_arr)
        feats[ba_col] = datasets.Array3D(shape=ba_arr.shape[1:], dtype="float16")
    new_cols = datasets.Dataset.from_dict(extra, features=datasets.Features(feats))
    merged = datasets.concatenate_datasets([ds, new_cols], axis=1)
    tmp = path.with_name(path.name + ".tmp")
    merged.to_parquet(str(tmp))
    os.replace(tmp, path)


def _register_features(out: pathlib.Path, args, config, data_config):
    """Write the rl_token / base_action feature entries into meta/info.json.

    Dims come from the config (no GPU / parquet read needed), so this can run as a
    cheap final step after all shards finish.
    """
    info = json.loads((out / "meta" / "info.json").read_text())
    if args.rl_tokens:
        token_dim = int(config.model.rlt_token_dim)
        info["features"][args.column_name] = {"dtype": "float32", "shape": [token_dim], "names": None}
    if args.base_actions:
        # raw action dim recovered by running the decoder on a dummy (yam → 14).
        decode = _build_action_decoder(data_config)
        raw_dim = int(decode(np.zeros((1, 1, 1, config.model.action_dim), np.float32),
                             np.zeros((1, config.model.action_dim), np.float32)).shape[-1])
        shape = [int(args.num_action_samples), int(config.model.action_horizon), raw_dim]
        info["features"][args.base_action_column] = {"dtype": "float16", "shape": shape, "names": None}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True, help="RLT TrainConfig name (e.g. pi05_insert-mouse-battery_rlt).")
    p.add_argument("--checkpoint", required=True, help="Trained RLT checkpoint step dir (contains params/).")
    p.add_argument("--output", default=None, help="Destination dataset root to COPY to (single-process).")
    p.add_argument("--in-place", action="store_true", help="Edit the source (local_files_path) directly.")
    p.add_argument("--dataset-root", default=None,
                   help="Operate in-place on this dataset dir (read + write). Use for sharded multi-GPU "
                        "runs on a pre-copied dataset; overrides --output/--in-place.")
    # sharding (DDP-style data parallel: one process per GPU over disjoint data files)
    p.add_argument("--num-shards", type=int, default=1, help="Total number of parallel processes.")
    p.add_argument("--shard-index", type=int, default=0, help="This process's shard id in [0, num-shards).")
    p.add_argument("--register-features-only", action="store_true",
                   help="Just register rl_token/base_action in meta/info.json and exit (run once after all shards).")
    # what to compute
    p.add_argument("--rl-tokens", action="store_true", default=True, help="Compute the rl_token column.")
    p.add_argument("--no-rl-tokens", dest="rl_tokens", action="store_false")
    p.add_argument("--base-actions", action="store_true", default=True, help="Sample base-VLA action chunks.")
    p.add_argument("--no-base-actions", dest="base_actions", action="store_false")
    # params
    p.add_argument("--column-name", default="rl_token", help="Parquet column name for the RL token.")
    p.add_argument("--base-action-column", default="base_action", help="Parquet column name for base action chunks.")
    p.add_argument("--num-action-samples", type=int, default=32, help="Base action chunks sampled per frame (N).")
    p.add_argument("--num-flow-steps", type=int, default=10, help="Flow-matching denoising steps for sampling.")
    p.add_argument("--batch-size", type=int, default=8, help="Frames per batch (effective action batch = B·N).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.rl_tokens and not args.base_actions:
        raise ValueError("Nothing to do: both --no-rl-tokens and --no-base-actions set.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}.")

    config = _config.get_config(args.config_name)
    data_config = _make_data_config(config, args.dataset_root)

    # ── register-only: cheap metadata step (no model / GPU) ───────────────────
    if args.register_features_only:
        out = pathlib.Path(args.dataset_root).resolve() if args.dataset_root else pathlib.Path(
            data_config.local_files_path).resolve()
        _register_features(out, args, config, data_config)
        print(f"Registered features in {out}/meta/info.json")
        return

    checkpoint = pathlib.Path(args.checkpoint).resolve()
    if not (checkpoint / "params").exists():
        raise FileNotFoundError(f"{checkpoint}/params not found — pass the checkpoint STEP dir.")

    # ── Resolve read/write root ───────────────────────────────────────────────
    if args.dataset_root:
        out = pathlib.Path(args.dataset_root).resolve()   # read + write in-place here
        print(f"Shard {args.shard_index}/{args.num_shards}: in-place on {out}")
    elif args.in_place:
        out = pathlib.Path(data_config.local_files_path).resolve()
        print(f"In-place: editing {out}")
    else:
        if not args.output:
            raise ValueError("Provide one of --dataset-root, --in-place, or --output.")
        out = pathlib.Path(args.output).resolve()
        if out.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out} exists. Pass --overwrite.")
            shutil.rmtree(out)
        print(f"Copying full dataset {data_config.local_files_path} → {out}")
        shutil.copytree(data_config.local_files_path, out)

    # ── Dataset + model ───────────────────────────────────────────────────────
    dataset = _build_dataset(config, data_config)
    n_frames = len(dataset)
    data_paths, data_rows, offsets = _data_files(out)
    if int(np.sum(data_rows)) != n_frames:
        raise RuntimeError(
            f"Frame-count mismatch: sum(data-file rows)={int(np.sum(data_rows))} != dataset frames={n_frames}. "
            "Global ordering would be misaligned; aborting."
        )
    info = json.loads((out / "meta" / "info.json").read_text())
    if not args.overwrite and args.num_shards == 1:
        if args.rl_tokens and args.column_name in info["features"]:
            raise FileExistsError(f"'{args.column_name}' already in features. Pass --overwrite.")
        if args.base_actions and args.base_action_column in info["features"]:
            raise FileExistsError(f"'{args.base_action_column}' already in features. Pass --overwrite.")

    model = _load_model(config, checkpoint)
    N, S = args.num_action_samples, args.num_flow_steps
    extract_token = nnx.jit(lambda m, obs: m.extract_rl_token(obs))
    sample_actions = nnx.jit(lambda m, rng, obs: m.sample_base_actions(rng, obs, num_samples=N, num_steps=S))
    to_raw_actions = _build_action_decoder(data_config) if args.base_actions else None
    base_rng = jax.random.key(args.seed)

    # ── This shard's data files (round-robin keeps sizes balanced) ────────────
    my_files = [k for k in range(len(data_paths)) if k % args.num_shards == args.shard_index]
    print(f"Shard handles {len(my_files)}/{len(data_paths)} data files: {my_files}")

    import torch  # local import (only needed here)

    def _file_done(path):
        import pyarrow.parquet as pq
        # schema_arrow gives the logical top-level names (rl_token/base_action);
        # the low-level .schema.names reports list-leaf names ("element").
        cols = set(pq.ParquetFile(path).schema_arrow.names)
        need = ([args.column_name] if args.rl_tokens else []) + (
            [args.base_action_column] if args.base_actions else [])
        return all(c in cols for c in need)

    def process_file(k):
        # Resume: a written file already carries the columns (atomic replace → no
        # partial state), so skip it unless --overwrite.
        if not args.overwrite and _file_done(data_paths[k]):
            print(f"file{k}: already annotated, skipping (resume).")
            return
        rows = data_rows[k]
        g0 = offsets[k]
        sub = torch.utils.data.Subset(dataset, range(g0, g0 + rows))
        loader = torch.utils.data.DataLoader(
            sub, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, collate_fn=_collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        buf_z, buf_a = [], []
        file_rng = jax.random.fold_in(base_rng, k)
        n_batches = (rows + args.batch_size - 1) // args.batch_size
        for step, batch in enumerate(tqdm.tqdm(loader, total=n_batches, desc=f"file{k}")):
            obs = _model.Observation.from_dict(batch)
            if args.rl_tokens:
                z = np.asarray(jax.device_get(extract_token(model, obs)), np.float32)
                buf_z.extend(z)
            if args.base_actions:
                a = np.asarray(jax.device_get(
                    sample_actions(model, jax.random.fold_in(file_rng, step), obs)), np.float32)  # [B,N,H,Dm]
                a = to_raw_actions(a, np.asarray(obs.state)).astype(np.float16)                   # [B,N,H,14]
                buf_a.extend(a)
        z_arr = np.stack(buf_z, axis=0) if args.rl_tokens else None
        ba_arr = np.stack(buf_a, axis=0) if args.base_actions else None
        if z_arr is not None and len(z_arr) != rows:
            raise RuntimeError(f"file{k}: got {len(z_arr)} rows, expected {rows}.")
        _write_data_file(data_paths[k], args.column_name, z_arr, args.base_action_column, ba_arr)

    for k in my_files:
        process_file(k)

    # ── Register features (single-process only; for shards run --register-features-only after wait) ──
    if args.num_shards == 1:
        _register_features(out, args, config, data_config)

    done = [c for c, on in [(args.column_name, args.rl_tokens), (args.base_action_column, args.base_actions)] if on]
    print(f"Shard {args.shard_index}/{args.num_shards} done @ {out}: wrote {done} for files {my_files}.")


if __name__ == "__main__":
    main()
