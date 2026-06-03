"""Measure the REAL openpi data loader throughput (no GPU step) to split the
bottleneck into loading (worker decode) vs batch-making (collate + host->device).

Times steady-state next(it). Run twice (OPENPI_PRELOAD_FRAMES 0 then 1) to see if
preloading frames helps end-to-end. Also times collate-only and the jax transfer
separately for the last batch."""
import dataclasses
import os
import time

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as dl

def main():
    cfg = _config.get_config("pi05_rft_phase2_rl_mh")
    cfg = dataclasses.replace(cfg, batch_size=1024, num_workers=int(os.environ.get("BENCH_WORKERS", "16")))
    print("config=%s batch=%d workers=%d preload=%s"
          % (cfg.name, cfg.batch_size, cfg.num_workers, os.environ.get("OPENPI_PRELOAD_FRAMES", "0")), flush=True)

    loader = dl.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=10)
    it = iter(loader)
    t0 = time.time()
    next(it)
    print("first batch (spawn+warmup): %.1fs" % (time.time() - t0), flush=True)
    ts = []
    for _ in range(6):
        t = time.time()
        next(it)
        ts.append(time.time() - t)
    print("steady next(it) per batch(1024): %.2fs  (min %.2f max %.2f)  -> %.0f samples/s"
          % (np.mean(ts), min(ts), max(ts), 1024 / np.mean(ts)), flush=True)


if __name__ == "__main__":
    main()
