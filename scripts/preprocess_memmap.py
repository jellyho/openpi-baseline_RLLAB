"""Convert an annotated LeRobot dataset to FRAME-INDEXED memmap arrays for the fast,
index-based critic loader (``openpi.rlt_critic.data_mm.MemmapVLADataset``).

Why: profiling the parquet loader showed the cost is NOT the numpy decode (~2ms) but
(a) ``read_row_group`` (~430ms/group) and (b) building/concatenating/permuting the giant
``next_candidates`` tensor (224MB/group, ~5x duplicated). A flat frame-indexed memmap kills
both: random O(1) gather at RAM speed (the OS page cache holds it once, shared read-only
across all DDP loader workers — no per-process duplication), and the loader carries only
INDICES, gathering ``next_candidates`` lazily for the 256-sample batch (no 1.8GB churn).

One-time cost: read the parquet (~180GB) + write the memmap (~175GB). Then training reads
incur zero parquet decode. Disk: needs ~dataset size free at the output dir.

Layout (``<out>/``):
    meta.json           n_frames, dims, dtypes, has_done, fps, source
    rl_token.dat        (N, 2048)            float32
    action.dat          (N, 14)              float32
    base_action.dat     (N, 32*50*14=22400)  float16   (flattened per frame)
    reward.dat          (N,)                 float32
    mc_return.dat       (N,)                 float32
    episode_index.dat   (N,)                 int64
    last_idx.dat        (N,)                 int64     (index of the LAST frame of each frame's episode)
    done.dat            (N,)                 int8      (explicit terminal if present, else inferred)
    commander.dat       (N,)                 int8      (0=teleop, 1=inference, -1=other)

Usage:
    .venv/bin/python scripts/preprocess_memmap.py --input <dataset_root> --out <memmap_dir> \
        --workers 4 [--max-files N]      # --max-files for a quick subset (benchmark)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from openpi.rlt_critic.data import (ACTION_DIM, BASE_ACTION_SHAPE, LATENT_DIM,
                                    _list_col_to_numpy, find_parquet_files)

BASE_FLAT = int(np.prod(BASE_ACTION_SHAPE))          # 32*50*14 = 22400
_READ_COLS = ["rl_token", "action", "reward", "mc_return", "base_action",
              "episode_index", "observation.commander_state"]


def _file_rows(paths):
    return [pq.ParquetFile(p).metadata.num_rows for p in paths]


def _has_done(path):
    return "done" in pq.ParquetFile(path).schema_arrow.names


def _commander_code(vals):
    """str commander_state -> int8 (0 teleop, 1 inference, -1 other)."""
    out = np.full(len(vals), -1, np.int8)
    out[np.asarray(vals) == "teleop"] = 0
    out[np.asarray(vals) == "inference"] = 1
    return out


def _convert_file(args):
    """Worker: decode one parquet file's row-groups and write them into the global memmap
    slice [offset, offset+rows). Each worker opens the memmaps in 'r+' and writes a DISJOINT
    range, so concurrent writers are safe."""
    out, path, offset, rows, has_done = args
    out = pathlib.Path(out)
    cols = list(_READ_COLS) + (["done"] if has_done else [])

    def mm(name, dtype, shape):
        return np.memmap(out / name, dtype=dtype, mode="r+", shape=shape)

    N_total = None  # not needed; we index by absolute offset into a full-size memmap
    rl = mm("rl_token.dat", np.float32, (offset + rows, LATENT_DIM))
    ac = mm("action.dat", np.float32, (offset + rows, ACTION_DIM))
    ba = mm("base_action.dat", np.float16, (offset + rows, BASE_FLAT))
    rw = mm("reward.dat", np.float32, (offset + rows,))
    mc = mm("mc_return.dat", np.float32, (offset + rows,))
    ep = mm("episode_index.dat", np.int64, (offset + rows,))
    dn = mm("done.dat", np.int8, (offset + rows,))
    cm = mm("commander.dat", np.int8, (offset + rows,))

    pf = pq.ParquetFile(path)
    o = offset
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=cols)
        n = t.num_rows
        rl[o:o + n] = _list_col_to_numpy(t["rl_token"], (LATENT_DIM,))
        ac[o:o + n] = _list_col_to_numpy(t["action"], (ACTION_DIM,))
        ba[o:o + n] = _list_col_to_numpy(t["base_action"], (BASE_FLAT,), dtype=np.float16)
        rw[o:o + n] = np.asarray(t["reward"].to_pylist(), np.float32)
        mc[o:o + n] = np.asarray(t["mc_return"].to_pylist(), np.float32)
        ep[o:o + n] = np.asarray(t["episode_index"].to_pylist(), np.int64)
        cm[o:o + n] = _commander_code(t["observation.commander_state"].to_pylist())
        if has_done:
            dn[o:o + n] = np.asarray(t["done"].to_pylist(), np.int8)
        else:                                            # infer: success(0)/failure(penalty) terminal
            r = rw[o:o + n]
            dn[o:o + n] = ((r >= -1e-6) | (r <= -0.05)).astype(np.int8)
        o += n
    for a in (rl, ac, ba, rw, mc, ep, dn, cm):
        a.flush()
    return path, rows


def _compute_last_idx(ep: np.ndarray) -> np.ndarray:
    """For each frame, the index of the LAST frame of its (contiguous) episode block."""
    n = len(ep)
    last = np.empty(n, np.int64)
    end = n - 1
    for j in range(n - 1, -1, -1):
        if j < n - 1 and ep[j] != ep[j + 1]:
            end = j
        last[j] = end
    return last


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="annotated dataset root (has data/ + meta/)")
    p.add_argument("--out", required=True, help="output memmap dir")
    p.add_argument("--workers", type=int, default=4, help="parallel file workers")
    p.add_argument("--max-files", type=int, default=0, help=">0 -> only first N parquet files (subset/bench)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    paths = find_parquet_files(args.input)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    rows = _file_rows(paths)
    offs = np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)
    N = int(offs[-1])
    has_done = _has_done(paths[0])
    out = pathlib.Path(args.out).resolve()
    if out.exists() and not args.overwrite and (out / "meta.json").exists():
        raise FileExistsError(f"{out} already has a memmap. Pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== preprocess_memmap: {len(paths)} files, {N:,} frames -> {out} ===")
    print(f"    base_action: {N * BASE_FLAT * 2 / 1e9:.0f} GB  rl_token: {N * LATENT_DIM * 4 / 1e9:.0f} GB  has_done={has_done}")

    # Preallocate every memmap at full size (sparse w+), then workers open r+ and fill ranges.
    specs = [("rl_token.dat", np.float32, (N, LATENT_DIM)), ("action.dat", np.float32, (N, ACTION_DIM)),
             ("base_action.dat", np.float16, (N, BASE_FLAT)), ("reward.dat", np.float32, (N,)),
             ("mc_return.dat", np.float32, (N,)), ("episode_index.dat", np.int64, (N,)),
             ("done.dat", np.int8, (N,)), ("commander.dat", np.int8, (N,))]
    for name, dtype, shape in specs:
        np.memmap(out / name, dtype=dtype, mode="w+", shape=shape).flush()

    t0 = time.time()
    jobs = [(str(out), str(paths[i]), int(offs[i]), int(rows[i]), has_done) for i in range(len(paths))]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for path, r in ex.map(_convert_file, jobs):
            print(f"    wrote {r:,} rows from {pathlib.Path(path).name}  ({time.time()-t0:.0f}s)", flush=True)

    # last_idx (global episode boundaries).
    print("    computing last_idx (episode boundaries)...", flush=True)
    ep = np.memmap(out / "episode_index.dat", np.int64, "r", shape=(N,))
    last = np.memmap(out / "last_idx.dat", np.int64, "w+", shape=(N,))
    last[:] = _compute_last_idx(np.asarray(ep)); last.flush()

    meta = {"n_frames": N, "latent_dim": LATENT_DIM, "action_dim": ACTION_DIM,
            "base_action_shape": list(BASE_ACTION_SHAPE), "base_flat": BASE_FLAT,
            "has_done": has_done, "source": str(pathlib.Path(args.input).resolve()),
            "n_files": len(paths)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"=== done: {N:,} frames in {time.time()-t0:.0f}s -> {out} ===")


if __name__ == "__main__":
    main()
