"""Throughput diagnostic: is training compute-bound (GPU) or I/O-bound (loader)?

Compares three throughputs on the real pipeline:
  (A) pure-GPU : the jitted train step looped on ONE fixed device batch (no loader).
  (B) full loop: load a batch + run the step, every iteration (the real training rate).
  (C) data-only: just pull batches from the loader (disk read + numpy assembly, no GPU).

Verdict:
  full ~ pure           -> COMPUTE-bound  (GPU saturated; Data Parallel will speed up ~N x).
  full << pure, ~ data  -> I/O-bound      (loader is the ceiling; need prefetch/parallel first).
"""
import time
import jax
import numpy as np

from vla_config import get_config
from vla_data import VLALeRobotDataset, prefetch
from vla_aqc import VLACriticTrainer, to_jax_batch

cfg = get_config("vla_aqc_td_macro")
print("jax devices:", jax.devices(), flush=True)

tr = VLACriticTrainer(cfg, seed=0)
step_fn = tr.make_train_step()
params, opt = tr.params, tr.opt_state
print(f"params {tr.num_params()/1e6:.1f}M  batch {cfg.optim.batch_size}  "
      f"agg={cfg.td.agg_mode}  mc_floor={cfg.td.mc_floor}", flush=True)

ds = VLALeRobotDataset(cfg.data_root, horizon=cfg.horizon, include_base_action=True,
                       mc_gamma=cfg.td.mc_gamma, discount=cfg.td.discount,
                       relabel_living=cfg.reward.relabel_living, relabel_fail=cfg.reward.relabel_fail)

def fresh():
    return ds.iter_bootstrap_batches(cfg.optim.batch_size, cfg.td.prefixes, seed=1)

N = 30

# --- compile / warmup on one fixed batch ---
it = fresh()
b0 = next(it)
jb0, pf0 = to_jax_batch(b0)
for _ in range(3):
    params, opt, info = step_fn(params, opt, jb0, pf0)
info["critic_loss"].block_until_ready()
print("compiled. timing...", flush=True)

# (A) pure-GPU: same device batch, no loader
t = time.time()
for _ in range(N):
    params, opt, info = step_fn(params, opt, jb0, pf0)
info["critic_loss"].block_until_ready()
pure = N / (time.time() - t)

# (B) full loop: load + step each iter (warm the loader's shuffle pool first)
for _ in range(5):
    next(it)
t = time.time(); n = 0
for b in it:
    jb, pf = to_jax_batch(b)
    params, opt, info = step_fn(params, opt, jb, pf)
    n += 1
    if n >= N:
        break
info["critic_loss"].block_until_ready()
full = n / (time.time() - t)

# (C) data-only: just pull batches (disk + numpy), no GPU, no H2D
it2 = fresh()
for _ in range(5):
    next(it2)
t = time.time(); n = 0
for b in it2:
    n += 1
    if n >= N:
        break
dataonly = n / (time.time() - t)

# (D) prefetched full loop: loader in a background thread (overlap load + GPU)
itp = prefetch(fresh(), depth=cfg.prefetch_depth)
for _ in range(5):
    next(itp)
t = time.time(); n = 0
for b in itp:
    jb, pf = to_jax_batch(b)
    params, opt, info = step_fn(params, opt, jb, pf)
    n += 1
    if n >= N:
        break
info["critic_loss"].block_until_ready()
pref = n / (time.time() - t)

print(f"\n=== TIMING (A100, batch {cfg.optim.batch_size}, {N} steps) ===")
print(f"(A) pure-GPU step        : {pure:6.2f} it/s   ({1000/pure:5.0f} ms/step)")
print(f"(B) full loop (serial)   : {full:6.2f} it/s   ({1000/full:5.0f} ms/step)")
print(f"(C) data-only (loader)   : {dataonly:6.2f} it/s   ({1000/dataonly:5.0f} ms/batch)")
print(f"(D) PREFETCHED full loop : {pref:6.2f} it/s   ({1000/pref:5.0f} ms/step)  <- Phase 1a")
print(f"    prefetch speedup vs (B): {pref/full:.2f}x   |  500k @ (D) = {500_000/pref/3600:.1f} h")
print(f"\nfull/pure = {full/pure:.2f}   (1.0 = GPU saturated)")
print(f"500k steps @ full = {500_000/full/3600:.1f} h")

if full / pure > 0.85:
    print("\n=> COMPUTE-BOUND: GPU가 병목. Data Parallel이 ~N배 효과. (DP 바로 가치)")
elif dataonly < pure * 0.9 and full <= dataonly * 1.1:
    print(f"\n=> I/O-BOUND: loader가 천장 (data-only {dataonly:.1f} ~ full {full:.1f} << pure {pure:.1f}).")
    print("   DP 전에 prefetch/parallel loader(Phase 1) 필요 — 안 그러면 GPU 늘려도 굶음.")
else:
    print(f"\n=> MIXED: full {full:.1f}, pure {pure:.1f}, data {dataonly:.1f}. "
          "GPU/IO 둘 다 기여 — prefetch로 둘을 겹치면 이득.")
