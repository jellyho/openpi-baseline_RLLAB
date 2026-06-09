"""Data loader for the annotated frozen-VLA dataset (LeRobot v3.0).

Targets the `insert-mouse-battery_annotated` dataset (and siblings) used for AQC critic
learning. Each frame carries:

  * ``rl_token``    (2048,)        frozen-VLA latent  -> critic state token
  * ``action``      (14,)          executed behaviour action that frame
  * ``base_action`` (32, 50, 14)   32 candidate action chunks (H=50)  [optional; huge]
  * ``reward``      ()             cost: -1e-4 living + -0.5 terminal-failure penalty
  * ``mc_return``   ()             discounted return-to-go, gamma=0.995, range [-0.5, 0]
  * ``episode_index``, ``frame_index``, ``observation.commander_state`` ('teleop'|'inference')

Critic training (RECAP-style) regresses Q(s_t, executed_chunk a_{t:t+H}) -> mc_return_t with
an HL-Gauss head over the fixed value support [-0.5, 0]. ``base_action`` (the 32 proposals)
is only needed for the AQC expected-prefix-max selection/bootstrap, so it is opt-in.

Design notes
------------
* Scalars (episode/frame index, reward, mc_return) are read fully into RAM (~tens of MB for
  3M frames) to build the frame index and episode boundaries. The heavy per-row arrays
  (``rl_token`` (8 KB/row) and ``action``) are read per Parquet row-group with a shuffle
  buffer -- sequential reads, no giant in-memory copy.
* For v1, an H-step executed chunk is only formed when the whole window stays inside one
  Parquet row-group AND one episode; windows crossing a row-group boundary are skipped
  (~H/rows_per_group of frames, a few %). This keeps reads sequential and simple.
"""

import glob
import os
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

# Dataset-specific constants (verified from meta/stats.json + gamma check).
VALUE_SUPPORT = (-0.5, 0.0)      # mc_return / reward range
DISCOUNT = 0.995                 # mc_t = r_t + 0.995 * mc_{t+1}
LATENT_DIM = 2048
ACTION_DIM = 14
BASE_ACTION_SHAPE = (32, 50, 14)
SCALAR_COLS = ["episode_index", "frame_index", "reward", "mc_return"]


def _list_col_to_numpy(col, trailing_shape, dtype=np.float32) -> np.ndarray:
    """Fast (nested) list-column -> dense ndarray, avoiding the slow per-row to_pylist.

    LeRobot stores array columns as nested ``list<...>`` whose leaf buffer is contiguous;
    descend ``.values`` to that leaf and reshape. ~4x faster than ``to_pylist`` for the big
    ``rl_token`` / ``base_action`` columns.
    """
    v = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    n = len(v)
    while hasattr(v, "values"):
        v = v.values
    arr = np.asarray(v.to_numpy(zero_copy_only=False)).astype(dtype, copy=False)
    return arr.reshape((n,) + tuple(trailing_shape))


def find_parquet_files(root: str) -> list[str]:
    """Return the sorted list of LeRobot data parquet shards under ``root``."""
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {root!r}")
    return files


class VLALeRobotDataset:
    """Row-group-streaming loader yielding MC critic-training batches.

    Args:
        root: dataset root (the dir containing ``data/`` and ``meta/``).
        horizon: executed-chunk length H (default 50, matching ``base_action``).
        commander_filter: keep only frames whose commander_state is in this set
            (e.g. {'inference'} or {'teleop'}); None keeps all.
        include_base_action: also load the (32,50,14) candidate chunks (expensive).
        shuffle_buffer_groups: how many row-groups to read into the shuffle pool at once.
    """

    def __init__(
        self,
        root: str,
        horizon: int = 50,
        commander_filter: Optional[set[str]] = None,
        include_base_action: bool = False,
        shuffle_buffer_groups: int = 8,
    ):
        self.root = root
        self.horizon = horizon
        self.commander_filter = commander_filter
        self.include_base_action = include_base_action
        self.shuffle_buffer_groups = shuffle_buffer_groups
        self.files = find_parquet_files(root)
        self._readers = {f: pq.ParquetFile(f) for f in self.files}
        self.value_support = VALUE_SUPPORT
        self.discount = DISCOUNT

    # ---- introspection -----------------------------------------------------------
    def summary(self) -> dict:
        n_groups = sum(r.metadata.num_row_groups for r in self._readers.values())
        n_rows = sum(r.metadata.num_rows for r in self._readers.values())
        return {"files": len(self.files), "row_groups": n_groups, "rows": n_rows,
                "horizon": self.horizon, "value_support": self.value_support}

    # ---- core sampling -----------------------------------------------------------
    def _row_cols(self) -> list[str]:
        cols = ["rl_token", "action"] + SCALAR_COLS + ["observation.commander_state"]
        if self.include_base_action:
            cols.append("base_action")
        return cols

    def _samples_from_table(self, t) -> Optional[dict]:
        """Form (state, chunk, target) samples from one row-group table.

        Windows that leave the table or cross an episode boundary are dropped.
        Returns a dict of stacked arrays, or None if the group yields no valid window.
        """
        H = self.horizon
        n = t.num_rows
        if n <= H:
            return None
        ep = np.asarray(t["episode_index"].to_pylist())
        act = _list_col_to_numpy(t["action"], (ACTION_DIM,))                 # (n, 14)
        rl = _list_col_to_numpy(t["rl_token"], (LATENT_DIM,))               # (n, 2048)
        mc = np.asarray(t["mc_return"].to_pylist(), dtype=np.float32)        # (n,)
        rew = np.asarray(t["reward"].to_pylist(), dtype=np.float32)          # (n,)
        cmd = t["observation.commander_state"].to_pylist()

        starts = np.arange(0, n - H)
        # window must stay in one episode: ep[i] == ep[i+H-1]
        same_ep = ep[starts] == ep[starts + H - 1]
        starts = starts[same_ep]
        if self.commander_filter is not None:
            keep = np.array([cmd[i] in self.commander_filter for i in starts], dtype=bool)
            starts = starts[keep]
        if len(starts) == 0:
            return None

        states = rl[starts]                                                  # (b, 2048)
        chunks = np.stack([act[i:i + H] for i in starts])                    # (b, H, 14)
        out = {
            "observations": states,
            "action_chunks": chunks.reshape(len(starts), H * ACTION_DIM),    # (b, H*14)
            "mc_return": mc[starts],                                         # (b,)
            "reward": rew[starts],                                           # (b,)
        }
        if self.include_base_action:
            ba = _list_col_to_numpy(t["base_action"], BASE_ACTION_SHAPE)     # (n,32,50,14)
            out["base_action"] = ba[starts]
        return out

    # ---- AQC-TD bootstrap sampling ----------------------------------------------
    def _episode_last_index(self, ep: np.ndarray) -> np.ndarray:
        """For each row, the index of the LAST frame of its (contiguous) episode block."""
        n = len(ep)
        last = np.empty(n, dtype=np.int64)
        end = n - 1
        for j in range(n - 1, -1, -1):
            if j < n - 1 and ep[j] != ep[j + 1]:
                end = j
            last[j] = end
        return last

    def _bootstrap_samples_from_table(self, t, prefixes: np.ndarray) -> Optional[dict]:
        """Form per-prefix AQC-TD bootstrap samples from one row-group table.

        For each start row i and each prefix length h in ``prefixes`` (1-indexed):
          * cum_reward = sum_{j<h} gamma^j r[i+j]                (the realized h-step return)
          * next state s_{i+h} = rl_token[i+h], candidates = base_action[i+h]   (N chunks)
          * term  = 1 if s_{i+h} is the episode terminal (penalty frame), else 0
          * valid = 1 if the h-step transition stays inside the episode, else 0
        The critic input chunk is the executed a_{i:i+H} = action[i:i+H].

        Windows are only formed where i+H < n (whole chunk + all next-states readable in
        this row-group); prefixes crossing the episode end are masked, not dropped, so the
        -0.5 failure-terminal transitions are kept.
        """
        H = self.horizon
        gamma = self.discount
        n = t.num_rows
        if n <= H + 1:
            return None
        ep = np.asarray(t["episode_index"].to_pylist())
        rl = _list_col_to_numpy(t["rl_token"], (LATENT_DIM,))               # (n, 2048)
        act = _list_col_to_numpy(t["action"], (ACTION_DIM,))               # (n, 14)
        rew = np.asarray(t["reward"].to_pylist(), dtype=np.float32)         # (n,)
        mc = np.asarray(t["mc_return"].to_pylist(), dtype=np.float32)       # (n,)
        ba = _list_col_to_numpy(t["base_action"], BASE_ACTION_SHAPE)        # (n,32,50,14)
        cmd = t["observation.commander_state"].to_pylist()
        last_idx = self._episode_last_index(ep)

        starts = np.arange(0, n - H)
        if self.commander_filter is not None:
            starts = starts[[cmd[i] in self.commander_filter for i in starts]]
        if len(starts) == 0:
            return None
        S, P = len(starts), len(prefixes)
        Li = last_idx[starts]                                               # (S,)
        end = starts[:, None] + prefixes[None, :]                           # (S,P) = i+h
        valid = (end <= Li[:, None]).astype(np.float32)                     # within episode
        term = ((end == Li[:, None]) & (valid > 0)).astype(np.float32)      # terminal next-state
        end_c = np.clip(end, 0, n - 1)                                      # safe gather idx

        # h-step discounted realized reward, per prefix (loop over the few prefixes).
        cum = np.zeros((S, P), np.float32)
        for p, h in enumerate(prefixes):
            acc = np.zeros(S, np.float32)
            for j in range(int(h)):
                acc += (gamma ** j) * rew[starts + j]
            cum[:, p] = acc

        N = ba.shape[1]
        out = {
            "observations": rl[starts],                                     # (S,2048)
            "action_chunks": np.stack([act[i:i + H] for i in starts]
                                      ).reshape(S, H * ACTION_DIM),         # (S,H*14)
            "cum_reward": cum,                                              # (S,P)
            "next_latents": rl[end_c],                                      # (S,P,2048)
            "next_candidates": ba[end_c].reshape(S, P, N, H * ACTION_DIM),  # (S,P,N,H*14)
            "next_mc_return": mc[end_c],                                    # (S,P)
            "term": term,                                                   # (S,P)
            "valid": valid,                                                 # (S,P)
        }
        return out

    def iter_bootstrap_batches(self, batch_size: int, prefixes: Sequence[int],
                               seed: int = 0, drop_last: bool = True) -> Iterator[dict]:
        """Yield AQC-TD bootstrap batches (needs ``base_action``; set it in the constructor).

        Args:
            batch_size: transitions per batch.
            prefixes: 1-indexed prefix lengths h to bootstrap on (the subsample grid),
                e.g. [1, 5, 10, 20, 35, 50]. Fewer => cheaper (the H=50 cost knob).
        """
        prefixes = np.asarray(sorted(set(int(h) for h in prefixes)), dtype=np.int64)
        assert prefixes.min() >= 1 and prefixes.max() <= self.horizon
        rng = np.random.default_rng(seed)
        cols = ["rl_token", "action", "reward", "mc_return", "base_action",
                "episode_index", "observation.commander_state"]
        work = [(f, g) for f in self.files
                for g in range(self._readers[f].metadata.num_row_groups)]
        rng.shuffle(work)
        pool, pool_n, buf = [], 0, 0

        def emit():
            nonlocal pool, pool_n
            if pool:
                big = {k: np.concatenate([p[k] for p in pool], 0) for k in pool[0]}
                idx = rng.permutation(pool_n)
                for s in range(0, pool_n - batch_size + 1, batch_size):
                    sel = idx[s:s + batch_size]
                    out = {k: v[sel] for k, v in big.items()}
                    out["prefixes"] = prefixes        # (P,) 1-indexed prefix grid
                    yield out
            pool, pool_n = [], 0

        for (f, g) in work:
            t = self._readers[f].read_row_group(g, columns=cols)
            s = self._bootstrap_samples_from_table(t, prefixes)
            if s is not None:
                pool.append(s); pool_n += len(s["valid"])
            buf += 1
            if buf >= self.shuffle_buffer_groups:
                yield from emit(); buf = 0
        yield from emit()

    def iter_batches(self, batch_size: int, seed: int = 0,
                     drop_last: bool = True) -> Iterator[dict]:
        """Yield shuffled training batches by streaming row-groups.

        Reads ``shuffle_buffer_groups`` row-groups into a pool, shuffles the pooled
        samples, and emits ``batch_size`` chunks until the pool drains, then refills.
        """
        rng = np.random.default_rng(seed)
        cols = self._row_cols()
        # (file, group) work-list, shuffled each epoch.
        work = [(f, g) for f in self.files
                for g in range(self._readers[f].metadata.num_row_groups)]
        rng.shuffle(work)

        pool: list[dict] = []
        pool_n = 0

        def emit_from_pool():
            nonlocal pool, pool_n
            if not pool:
                return
            big = {k: np.concatenate([p[k] for p in pool], 0) for k in pool[0]}
            idx = rng.permutation(pool_n)
            for s in range(0, pool_n - (batch_size if drop_last else 1) + 1, batch_size):
                sel = idx[s:s + batch_size]
                if drop_last and len(sel) < batch_size:
                    break
                yield {k: v[sel] for k, v in big.items()}
            pool, pool_n = [], 0

        buf = 0
        for (f, g) in work:
            t = self._readers[f].read_row_group(g, columns=cols)
            s = self._samples_from_table(t)
            if s is not None:
                pool.append(s)
                pool_n += len(s["mc_return"])
            buf += 1
            if buf >= self.shuffle_buffer_groups:
                yield from emit_from_pool()
                buf = 0
        yield from emit_from_pool()
