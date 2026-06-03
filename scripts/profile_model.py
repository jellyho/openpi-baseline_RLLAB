"""Break down LPS-RFT phase-2 step cost: VLM-prefix forward vs full forward vs
forward+backward, for crossq_joint_batch False/True. Uses one REAL batch from the
data loader (so next_obs/reward/done are correctly packed) + a random-init model
(timing is the same as a weight-loaded model). Run on 1 GPU."""
import dataclasses
import os
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.training.config as _config
import openpi.training.data_loader as dl


def _time(fn, *args, n=5):
    r = fn(*args)
    jax.block_until_ready(r)  # compile + warmup
    t = time.time()
    for _ in range(n):
        r = fn(*args)
    jax.block_until_ready(r)
    return (time.time() - t) / n


def run(crossq: bool, batch: int):
    cfg = _config.get_config("pi05_rft_phase2_rl_mh")
    cfg = dataclasses.replace(
        cfg, batch_size=batch, num_workers=8,
        model=dataclasses.replace(cfg.model, crossq_joint_batch=crossq),
    )
    loader = dl.create_data_loader(cfg, shuffle=True, num_batches=4)
    obs, act = next(iter(loader))
    model = cfg.model.create(jax.random.key(0))

    @nnx.jit
    def prefix(model, obs):
        return model._embed_prefix_kv(obs)

    @nnx.jit
    def fwd(model, obs, act):
        return model.compute_loss(jax.random.key(1), obs, act, train=True)[0]

    def lossfn(model, obs, act):
        return jnp.mean(model.compute_loss(jax.random.key(1), obs, act, train=True)[0])

    @nnx.jit
    def fwdbwd(model, obs, act):
        return nnx.value_and_grad(lossfn, argnums=nnx.DiffState(0, cfg.trainable_filter))(model, obs, act)

    tp = _time(prefix, model, obs)
    tf = _time(fwd, model, obs, act)
    tg = _time(fwdbwd, model, obs, act)
    print("crossq=%-5s batch=%d | prefix(VLM 1x)=%.0fms  full-fwd=%.0fms  fwd+bwd(step)=%.0fms"
          % (crossq, batch, tp * 1000, tf * 1000, tg * 1000), flush=True)


def main():
    b = int(os.environ.get("BB", "256"))
    run(False, b)
    run(True, b)


if __name__ == "__main__":
    main()
