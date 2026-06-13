"""Fast index-based critic loader over a frame-indexed memmap (built by
``scripts/preprocess_memmap.py``).

Drop-in for ``VLALeRobotDataset.iter_bootstrap_batches`` / ``iter_batches`` (same batch-dict
keys), but instead of streaming + decoding parquet row-groups and pooling/concatenating the
giant ``next_candidates`` tensor, it:

  * memmaps the decoded columns read-only — the OS page cache holds them ONCE in RAM and shares
    them across all DDP loader-worker processes (no per-worker duplication; abundant RAM = the
    whole 175GB stays resident),
  * samples each batch's start frames at random GLOBALLY (true shuffle, not a shuffle buffer),
  * carries only INDICES through the hot path and gathers ``next_candidates`` lazily for the
    256-sample batch (57MB), never the ~1.8GB-per-buffer concat/permute the parquet path does.

Net: the two profiled bottlenecks (≈430ms parquet read/group + ≈615ms next_candidates churn/
buffer) both vanish — batch build is pure RAM-speed fancy-index gather.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterator, Optional, Sequence

import numpy as np


def make_dataset(ds_kwargs: dict):
    """Factory: a ``MemmapVLADataset`` when ``ds_kwargs`` carries a truthy ``memmap_dir``,
    else the parquet-streaming ``VLALeRobotDataset``. Both expose the same
    ``iter_bootstrap_batches`` / ``iter_batches`` / ``summary`` interface, so callers (train
    loop, multiprocess loader) are agnostic. Kept jax-free so loader workers can import it."""
    if ds_kwargs.get("memmap_dir"):
        return MemmapVLADataset(**ds_kwargs)              # extra parquet-only kwargs are swallowed
    from openpi.rlt_critic.data import VLALeRobotDataset
    return VLALeRobotDataset(**{k: v for k, v in ds_kwargs.items() if k != "memmap_dir"})


class MemmapVLADataset:
    """Frame-indexed memmap dataset for the AQC critic (index-based batching)."""

    def __init__(
        self,
        memmap_dir: str,
        horizon: int = 50,
        commander_filter: Optional[set] = None,
        discount: float = 0.9999,
        mc_gamma: Optional[float] = None,           # accepted for API parity; memmap uses precomputed mc
        n_step: int = 0,
        bootstrap_subset: int = 0,
        **_ignored,                                 # swallow parquet-only kwargs (num_workers, shuffle_buffer_groups, ...)
    ):
        self.dir = pathlib.Path(memmap_dir).resolve()
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.N = int(self.meta["n_frames"])
        self.latent_dim = int(self.meta["latent_dim"])
        self.action_dim = int(self.meta["action_dim"])
        self.base_flat = int(self.meta["base_flat"])
        self.num_candidates = int(self.meta["base_action_shape"][0])     # 32
        self.horizon = horizon
        self.discount = discount
        self.n_step = int(n_step)
        self.bootstrap_subset = int(bootstrap_subset)

        def mm(name, dtype, shape):
            return np.memmap(self.dir / name, dtype=dtype, mode="r", shape=shape)

        self.rl = mm("rl_token.dat", np.float32, (self.N, self.latent_dim))
        self.act = mm("action.dat", np.float32, (self.N, self.action_dim))
        self.ba = mm("base_action.dat", np.float16, (self.N, self.base_flat))
        self.rew = mm("reward.dat", np.float32, (self.N,))
        self.mc = mm("mc_return.dat", np.float32, (self.N,))
        self.ep = mm("episode_index.dat", np.int64, (self.N,))
        self.last = mm("last_idx.dat", np.int64, (self.N,))
        self.done = mm("done.dat", np.int8, (self.N,))
        self.cmd = mm("commander.dat", np.int8, (self.N,))

        # Valid starts: at least one real next frame in the episode (last_idx > i), optionally
        # restricted to a commander_state. Precomputed once (small int64 index array).
        starts = np.arange(self.N, dtype=np.int64)
        keep = np.asarray(self.last) > starts
        if commander_filter:
            codes = {"teleop": 0, "inference": 1}
            allowed = np.array([codes[c] for c in commander_filter if c in codes], np.int8)
            keep &= np.isin(np.asarray(self.cmd), allowed)
        self.valid_starts = starts[keep]

    def summary(self) -> dict:
        return {"frames": self.N, "valid_starts": int(len(self.valid_starts)),
                "horizon": self.horizon, "memmap": str(self.dir)}

    # ---------------------------------------------------------------- bootstrap (TD)
    def iter_bootstrap_batches(self, batch_size: int, prefixes: Sequence[int],
                               seed: int = 0, drop_last: bool = True,
                               shard: tuple = (0, 1)) -> Iterator[dict]:
        pf = np.asarray(sorted(set(int(h) for h in prefixes)), np.int64)
        assert pf.min() >= 1 and pf.max() <= self.horizon
        P = len(pf)
        H, Dr, g, ns = self.horizon, self.action_dim, self.discount, self.n_step
        Ncand = self.num_candidates
        sub = (0 < self.bootstrap_subset < Ncand)
        rng = np.random.default_rng((seed, shard[0]))   # per-worker decorrelated stream
        vs = self.valid_starts
        maxlen = int(pf.max()) + ns
        jvec = np.arange(maxlen)
        disc = (g ** jvec).astype(np.float32)
        Harange = np.arange(H)
        cum_take = pf + ns - 1                          # column index into the cumsum

        while True:
            starts = vs[rng.integers(0, len(vs), batch_size)]            # (B,)
            Li = self.last[starts]                                       # (B,)
            end = starts[:, None] + (pf[None, :] + ns)                  # (B,P)
            valid = (end <= Li[:, None]).astype(np.float32)             # (B,P)
            end_c = np.clip(end, 0, self.N - 1)                         # (B,P) safe gather

            # h(+N)-step realized discounted reward (clamp within episode for masked prefixes).
            ridx = np.minimum(starts[:, None] + jvec[None, :], Li[:, None])     # (B,maxlen)
            rd = self.rew[ridx] * disc[None, :]                                 # (B,maxlen)
            cum = np.cumsum(rd, axis=1)[:, cum_take].astype(np.float32)         # (B,P)

            term = (self.done[end_c].astype(bool) & (valid > 0)).astype(np.float32)  # (B,P)

            pos = np.minimum(starts[:, None] + Harange[None, :], Li[:, None])   # (B,H) hold-last
            ef = end_c.ravel()                                                  # (B*P,)
            if sub:
                # Gather ONLY the subset candidates (strided index into the memmap) -- 4x less
                # data read + transferred than gathering all N then slicing.
                cidx = rng.choice(Ncand, self.bootstrap_subset, replace=False)
                ba3 = self.ba.reshape(self.N, Ncand, H * Dr)                    # view, no copy
                nc = np.asarray(ba3[ef[:, None], cidx[None, :]]).reshape(
                    batch_size, P, self.bootstrap_subset, H * Dr)              # (B,P,M,H*Dr)
            else:
                nc = np.asarray(self.ba[ef]).reshape(batch_size, P, Ncand, H * Dr)  # (B,P,N,H*Dr)

            yield {
                "observations": np.asarray(self.rl[starts]),                    # (B,2048)
                "action_chunks": np.asarray(self.act[pos]).reshape(batch_size, H * Dr),
                "cum_reward": cum,                                              # (B,P)
                "next_latents": np.asarray(self.rl[end_c.ravel()]).reshape(batch_size, P, self.latent_dim),
                "next_candidates": nc,                                          # (B,P,N',H*Dr) f16
                "next_mc_return": np.asarray(self.mc[end_c]),                   # (B,P)
                "mc_return": np.asarray(self.mc[starts]),                       # (B,)
                "term": term,                                                   # (B,P)
                "valid": valid,                                                 # (B,P)
                "prefixes": pf,                                                 # (P,)
            }

    # ---------------------------------------------------------------- MC (no base_action)
    def iter_batches(self, batch_size: int, seed: int = 0, drop_last: bool = True,
                     shard: tuple = (0, 1)) -> Iterator[dict]:
        H, Dr = self.horizon, self.action_dim
        rng = np.random.default_rng((seed, shard[0]))
        vs = self.valid_starts
        Harange = np.arange(H)
        while True:
            starts = vs[rng.integers(0, len(vs), batch_size)]
            Li = self.last[starts]
            pos = np.minimum(starts[:, None] + Harange[None, :], Li[:, None])
            yield {
                "observations": np.asarray(self.rl[starts]),
                "action_chunks": np.asarray(self.act[pos]).reshape(batch_size, H * Dr),
                "mc_return": np.asarray(self.mc[starts]),
                "reward": np.asarray(self.rew[starts]),
            }
