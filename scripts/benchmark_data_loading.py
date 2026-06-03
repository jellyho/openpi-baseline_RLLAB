"""Benchmark the TRAINING data pipeline (video decode + transforms + collate).

Measures pure data-loading throughput so you can find the best
(batch_size, num_workers, prefetch_factor) without running the model.  This is
the thing that starves the GPU when GPU-util is low.

It runs **CPU-only** (forces JAX_PLATFORMS=cpu + hides CUDA) so it does NOT touch
training GPUs — the decode/transform cost it measures is the same regardless of
where the final batch would land.

Examples
--------
# sweep workers for the from-critic config at global batch 1024
uv run scripts/benchmark_data_loading.py --config pi05_rft_phase2_rl_from_critic \
    --batch-sizes 1024 --workers 16 32 48 64 --prefetch 4

# compare batch sizes and prefetch
uv run scripts/benchmark_data_loading.py --config pi05_rft_phase2_rl_from_critic \
    --batch-sizes 256 512 1024 --workers 32 --prefetch 2 4 8

# flag whether loading keeps up with a known GPU step time (e.g. 0.45 s/step)
uv run scripts/benchmark_data_loading.py --config pi05_rft_phase2_rl_from_critic \
    --batch-sizes 1024 --workers 32 64 --target-step-s 0.45
"""

# --- force CPU BEFORE importing jax so we never grab a training GPU ----------
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import gc
import itertools
import logging
import multiprocessing
import statistics
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import numpy as np
import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _build_dataset(train_config, skip_norm_stats: bool):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ds = _data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
    )
    ds = _data_loader.transform_dataset(ds, data_config, skip_norm_stats=skip_norm_stats)
    return ds, data_config


def _make_loader(ds, batch_size, num_workers, prefetch_factor):
    mp_ctx = multiprocessing.get_context("spawn") if num_workers > 0 else None
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        multiprocessing_context=mp_ctx,
        persistent_workers=num_workers > 0,
        collate_fn=_data_loader._collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def _img_shape(batch) -> str:
    # batch is a nested dict; find the first image array for a sanity print.
    try:
        imgs = batch["image"] if "image" in batch else batch.get("observation", {}).get("image", {})
        k = next(iter(imgs))
        return f"{k}:{tuple(np.asarray(imgs[k]).shape)}"
    except Exception:  # noqa: BLE001
        return "?"


def bench_one(ds, batch_size, num_workers, prefetch_factor, num_batches, warmup):
    print(
        f"  → bs={batch_size} workers={num_workers} pf={prefetch_factor}: spawning "
        f"{num_workers} workers + first batch ({batch_size} samples on 1 worker, "
        f"slow if bs is large)...",
        flush=True,
    )
    loader = _make_loader(ds, batch_size, num_workers, prefetch_factor)
    it = iter(loader)

    # First batch = cold start (worker spawn + JAX/torch import + first decode).
    t0 = time.perf_counter()
    first = next(it)
    cold = time.perf_counter() - t0
    shape = _img_shape(first)
    print(f"    first batch in {cold:.1f}s — timing {num_batches} batches...", flush=True)

    # A few warmup batches so the worker prefetch buffers fill.
    for _ in range(max(warmup, 0)):
        next(it)

    times = []
    for _ in range(num_batches):
        t = time.perf_counter()
        next(it)
        times.append(time.perf_counter() - t)

    del it, loader
    gc.collect()

    times.sort()
    p50 = statistics.median(times)
    p90 = times[int(0.9 * (len(times) - 1))]
    mean = statistics.mean(times)
    samples_per_s = batch_size / mean
    return {
        "cold": cold,
        "mean_ms": mean * 1e3,
        "p50_ms": p50 * 1e3,
        "p90_ms": p90 * 1e3,
        "samples_s": samples_per_s,
        "shape": shape,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="TrainConfig name (e.g. pi05_rft_phase2_rl_from_critic)")
    # samples/s (throughput) is ~independent of this batch size, so keep it SMALL
    # for a fast cold start — one worker decodes the whole first batch before any
    # timing.  Use the real train batch only if you also care about per-batch RAM.
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--workers", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--prefetch", type=int, nargs="+", default=[4])
    ap.add_argument("--num-batches", type=int, default=20, help="timed batches per combo")
    ap.add_argument("--warmup", type=int, default=3, help="untimed batches after the first")
    ap.add_argument("--skip-norm-stats", action="store_true", help="skip Normalize (if norm_stats missing)")
    ap.add_argument("--target-step-s", type=float, default=None,
                    help="GPU step time (s); flags whether loading keeps up")
    args = ap.parse_args()

    cfgs = {c.name: c for c in _config._CONFIGS}
    if args.config not in cfgs:
        raise SystemExit(f"unknown config '{args.config}'. available: {sorted(cfgs)[:20]} ...")
    train_config = cfgs[args.config]

    print(f"config           : {args.config}")
    print(f"repo_id          : {train_config.data.repo_id}")
    print(f"nproc (cpu count): {os.cpu_count()}")
    print(f"timed batches    : {args.num_batches} (warmup {args.warmup} + 1 cold)\n")

    ds, _ = _build_dataset(train_config, args.skip_norm_stats)
    print(f"dataset size     : {len(ds)} samples\n")

    combos = list(itertools.product(args.batch_sizes, args.workers, args.prefetch))
    print(f"{'bs':>5} {'workers':>7} {'pf':>3} | {'cold(s)':>8} {'p50 ms':>8} {'p90 ms':>8} "
          f"{'mean ms':>8} {'samples/s':>10}" + ("  keepup" if args.target_step_s else "") + "  image")
    print("-" * (78 + (8 if args.target_step_s else 0)))

    results = []
    for bs, nw, pf in combos:
        try:
            r = bench_one(ds, bs, nw, pf, args.num_batches, args.warmup)
        except KeyboardInterrupt:
            print("\n  interrupted — stopping sweep, reporting what finished.", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"{bs:>5} {nw:>7} {pf:>3} | FAILED: {str(e)[:60]}")
            continue
        keepup = ""
        if args.target_step_s:
            need = bs / args.target_step_s              # samples/s the GPU consumes
            ok = r["samples_s"] >= need
            keepup = f"  {'OK ' if ok else 'SLOW'}({r['samples_s']/need:4.2f}x)"
        print(f"{bs:>5} {nw:>7} {pf:>3} | {r['cold']:>8.1f} {r['p50_ms']:>8.0f} {r['p90_ms']:>8.0f} "
              f"{r['mean_ms']:>8.0f} {r['samples_s']:>10.0f}" + keepup + f"  {r['shape']}")
        results.append(((bs, nw, pf), r))

    if results:
        best = max(results, key=lambda kr: kr[1]["samples_s"])
        (bs, nw, pf), r = best
        print(f"\nbest throughput  : bs={bs} workers={nw} prefetch={pf} "
              f"→ {r['samples_s']:.0f} samples/s ({r['mean_ms']:.0f} ms/batch)")
        if args.target_step_s:
            need = bs / args.target_step_s
            verdict = "data loading KEEPS UP" if r["samples_s"] >= need else "DATA LOADING IS THE BOTTLENECK"
            print(f"vs GPU step {args.target_step_s}s (needs {need:.0f} samples/s): {verdict}")


if __name__ == "__main__":
    main()
