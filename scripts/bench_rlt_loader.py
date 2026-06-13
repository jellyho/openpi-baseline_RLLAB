"""Profile the RLT/AQC critic data loaders: memmap (index-based) vs parquet (streaming).

Builds a synthetic memmap if --memmap not given (runs anywhere), then times:
  * memmap single-process (full N candidates)
  * memmap single-process + bootstrap_subset
  * parquet single-process (if --parquet <real dataset> given)
  * memmap multi-process (K workers, spawn) -> aggregate batches/s (the DDP-relevant number,
    since workers share the page-cached memmap read-only -> ~linear scaling, no re-decode)

Usage:
  JAX_PLATFORMS=cpu .venv/bin/python scripts/bench_rlt_loader.py \
    [--memmap <dir>] [--parquet <dataset_root>] [--synth-frames 150000] \
    [--batch 256] [--workers 1 4 8] [--subset 8]
"""
import argparse, json, pathlib, time, tempfile
import numpy as np

PREFIXES = [10, 20, 30, 40, 50]
LAT, AD, BF = 2048, 14, 32 * 50 * 14


def build_synth(d: pathlib.Path, NF: int):
    d.mkdir(parents=True, exist_ok=True)
    if (d / "meta.json").exists():
        return
    def w(name, dt, sh, val):
        a = np.memmap(d / name, dtype=dt, mode="w+", shape=sh); a[:] = val; a.flush()
    w("rl_token.dat", np.float32, (NF, LAT), 0.1); w("action.dat", np.float32, (NF, AD), 0.0)
    ba = np.memmap(d / "base_action.dat", np.float16, "w+", shape=(NF, BF)); ba[:] = np.float16(0.05); ba.flush()
    w("reward.dat", np.float32, (NF,), -1.0); w("mc_return.dat", np.float32, (NF,), -0.5)
    ep = np.repeat(np.arange(NF // 500 + 1), 500)[:NF].astype(np.int64)
    np.memmap(d / "episode_index.dat", np.int64, "w+", shape=(NF,))[:] = ep
    last = np.empty(NF, np.int64); e = NF - 1
    for j in range(NF - 1, -1, -1):
        if j < NF - 1 and ep[j] != ep[j + 1]: e = j
        last[j] = e
    np.memmap(d / "last_idx.dat", np.int64, "w+", shape=(NF,))[:] = last
    w("done.dat", np.int8, (NF,), 0); w("commander.dat", np.int8, (NF,), 0)
    json.dump({"n_frames": NF, "latent_dim": LAT, "action_dim": AD,
               "base_action_shape": [32, 50, 14], "base_flat": BF, "has_done": True},
              open(d / "meta.json", "w"))


def time_iter(it, secs=4.0, warm=5):
    for _ in range(warm): next(it)
    t = time.time(); n = 0
    while time.time() - t < secs:
        next(it); n += 1
    return n / (time.time() - t)


# module-level worker (spawn-picklable)
_MM_DIR = None; _BATCH = None; _SUBSET = 0
def _init(mm_dir, batch, subset):
    global _MM_DIR, _BATCH, _SUBSET; _MM_DIR, _BATCH, _SUBSET = mm_dir, batch, subset
def _worker(seed):
    from openpi.rlt_critic.data_mm import MemmapVLADataset
    ds = MemmapVLADataset(_MM_DIR, horizon=50, discount=0.9999, bootstrap_subset=_SUBSET)
    return time_iter(ds.iter_bootstrap_batches(_BATCH, PREFIXES, seed=seed), secs=4.0)


def main():
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default=None); p.add_argument("--parquet", default=None)
    p.add_argument("--synth-frames", type=int, default=150_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--subset", type=int, default=8)
    args = p.parse_args()

    mm = pathlib.Path(args.memmap) if args.memmap else pathlib.Path(tempfile.mkdtemp(prefix="bench_mm_", dir="/dev/shm"))
    if not args.memmap:
        print(f"building synthetic memmap: {args.synth_frames:,} frames "
              f"({args.synth_frames*BF*2/1e9:.1f} GB base_action) -> {mm}")
        build_synth(mm, args.synth_frames)
    from openpi.rlt_critic.data_mm import MemmapVLADataset
    ds = MemmapVLADataset(str(mm), horizon=50, discount=0.9999)
    _ = np.asarray(ds.ba[:min(ds.N, 300_000)]).sum()    # warm page cache

    print(f"\n=== single-process (batch={args.batch}) ===")
    r_full = time_iter(ds.iter_bootstrap_batches(args.batch, PREFIXES, seed=0))
    print(f"  memmap (full N=32):     {r_full:6.1f} batches/s")
    ds_s = MemmapVLADataset(str(mm), horizon=50, discount=0.9999, bootstrap_subset=args.subset)
    r_sub = time_iter(ds_s.iter_bootstrap_batches(args.batch, PREFIXES, seed=0))
    print(f"  memmap (subset={args.subset}):      {r_sub:6.1f} batches/s")
    if args.parquet:
        from openpi.rlt_critic.data import VLALeRobotDataset
        pq = VLALeRobotDataset(root=args.parquet, horizon=50, include_base_action=True,
                               mc_gamma=None, discount=0.9999, shuffle_buffer_groups=8, num_workers=8)
        it = pq.iter_bootstrap_batches(args.batch, PREFIXES, seed=0); next(it)
        r_pq = time_iter(it, warm=0)
        print(f"  parquet (8 threads):    {r_pq:6.1f} batches/s   -> memmap {r_full/r_pq:.1f}x")

    print("\n=== multi-process memmap (spawn, shared page cache) ===")
    ctx = mp.get_context("spawn")
    for K in args.workers:
        with ctx.Pool(K, initializer=_init, initargs=(str(mm), args.batch, 0)) as pool:
            rates = pool.map(_worker, list(range(K)))
        print(f"  x{K:2d} workers (full N=32): {sum(rates):6.0f} batches/s aggregate  (per-worker ~{sum(rates)/K:.0f})")

    if not args.memmap:
        import shutil; shutil.rmtree(mm, ignore_errors=True)


if __name__ == "__main__":
    main()
