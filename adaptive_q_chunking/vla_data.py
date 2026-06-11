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
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq


_PREFETCH_SENTINEL = object()


def prefetch(iterable, depth: int = 3):
    """Run a batch generator in a background thread, yielding via a bounded queue.

    Overlaps disk read + numpy batch assembly with the consumer (the GPU train step):
    while the GPU computes step t, the worker prepares batch t+1. Thread-based because
    pyarrow reads and numpy ops release the GIL, so they run concurrently with JAX
    dispatch on the main thread. ``depth`` batches may sit in RAM (each VLA bootstrap
    batch is ~1.2 GB due to next_candidates), so keep it small.

    Exceptions from the worker propagate to the consumer; the thread is a daemon.
    """
    if depth <= 0:
        yield from iterable
        return
    q: "queue.Queue" = queue.Queue(maxsize=depth)

    def _worker():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as e:                 # propagate to consumer
            q.put(e)
        else:
            q.put(_PREFETCH_SENTINEL)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _PREFETCH_SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        yield item

# Dataset-specific constants (verified from meta/stats.json + gamma check).
VALUE_SUPPORT = (-0.5, 0.0)      # precomputed mc_return range (gamma=0.995)
VALUE_SUPPORT_UNDISCOUNTED = (-1.0, 0.0)  # undiscounted MC range (wider; covers long episodes)
DISCOUNT = 0.995                 # mc_t = r_t + 0.995 * mc_{t+1} (precomputed column)
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
        mc_gamma: Optional[float] = 1.0,
        discount: float = DISCOUNT,
        relabel_living: Optional[float] = None,
        relabel_fail: float = -0.6,
        num_workers: int = 8,
    ):
        """
        Args:
            mc_gamma: discount used to compute the MC return target.
                ``1.0`` (default) = undiscounted sum of raw rewards (RECAP-style).
                ``None`` = use the precomputed ``mc_return`` column (gamma=0.995).
                Any other float = re-compute with that discount from raw rewards.
            discount: gamma used for the TD-bootstrap cumulative reward (cum_reward).
                Must match ``cfg.td.discount`` so cum_reward and the bootstrap weight agree.
            relabel_living: if set, rescale the raw living cost (-1e-4) to this value in-loader
                (e.g. -4e-4); the success terminal (0.0) scales to 0.0. ``None`` keeps raw.
            relabel_fail: failure-terminal penalty after relabel (raw -0.5 -> this). Only used
                when ``relabel_living`` is set.
        """
        self.root = root
        self.horizon = horizon
        self.commander_filter = commander_filter
        self.include_base_action = include_base_action
        self.shuffle_buffer_groups = shuffle_buffer_groups
        self.num_workers = num_workers
        self.mc_gamma = mc_gamma
        self.relabel_living = relabel_living
        self.relabel_fail = relabel_fail
        self.files = find_parquet_files(root)
        self._readers = {f: pq.ParquetFile(f) for f in self.files}
        self.value_support = VALUE_SUPPORT_UNDISCOUNTED if mc_gamma == 1.0 else VALUE_SUPPORT
        self.discount = discount

    # ---- introspection -----------------------------------------------------------
    def summary(self) -> dict:
        n_groups = sum(r.metadata.num_row_groups for r in self._readers.values())
        n_rows = sum(r.metadata.num_rows for r in self._readers.values())
        return {"files": len(self.files), "row_groups": n_groups, "rows": n_rows,
                "horizon": self.horizon, "value_support": self.value_support}

    # ---- reward relabel (in-loader; avoids re-annotating 181GB of parquet) -------
    def _relabel(self, rew: np.ndarray) -> np.ndarray:
        """Map the raw reward column onto the chosen value scale, no re-annotation.

        Raw encoding (verified on disk): living step = -1e-4, success terminal = 0.0,
        failure terminal = -0.5 (exactly). The living cost is rescaled by a scalar
        (relabel_living/-1e-4, e.g. -4e-4 -> x4; 0.0 stays 0.0); the failure penalty is
        remapped *separately* (-0.5 -> relabel_fail) because it is NOT the same scale
        (failure ~x1.2, living x4). Returns ``rew`` unchanged when relabel is off.
        """
        if self.relabel_living is None:
            return rew
        scale = self.relabel_living / -1e-4                      # living -1e-4 -> relabel_living
        out = rew * scale                                        # living & success(0.0) rescaled
        out = np.where(rew <= -0.05, self.relabel_fail, out)     # failure -0.5 -> relabel_fail
        return out.astype(np.float32)

    # ---- MC return computation ---------------------------------------------------
    def _compute_mc(self, rew: np.ndarray, ep: np.ndarray) -> np.ndarray:
        """Compute per-frame MC return from raw rewards within a row-group.

        For ``mc_gamma=None`` this is not called (the precomputed column is used).
        For ``mc_gamma=1.0`` returns undiscounted sum-to-end-of-episode (RECAP-style).
        For other gamma computes discounted RTG with that factor.

        Backward pass: reset at episode boundaries (detected by episode_index change).
        """
        n = len(rew)
        mc = np.zeros(n, dtype=np.float32)
        g = 1.0 if self.mc_gamma is None else float(self.mc_gamma)
        running = 0.0
        for i in range(n - 1, -1, -1):
            running = float(rew[i]) + g * running
            mc[i] = running
            if i == 0 or ep[i - 1] != ep[i]:  # start of a new episode block -> reset
                running = 0.0
        return mc

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
        rew = self._relabel(np.asarray(t["reward"].to_pylist(), dtype=np.float32))   # (n,)
        if self.mc_gamma is None:
            mc = np.asarray(t["mc_return"].to_pylist(), dtype=np.float32)   # precomputed
        else:
            mc = self._compute_mc(rew, ep)                                   # recomputed
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
        rew = self._relabel(np.asarray(t["reward"].to_pylist(), dtype=np.float32))   # (n,)
        if self.mc_gamma is None:
            mc = np.asarray(t["mc_return"].to_pylist(), dtype=np.float32)  # precomputed
        else:
            mc = self._compute_mc(rew, ep)                                  # recomputed
        # Keep base_action as fp16 (it is stored as halffloat): skips the f32 astype (~89->5ms)
        # and halves the next_candidates gather + host->device transfer. Cast to f32 on GPU.
        ba = _list_col_to_numpy(t["base_action"], BASE_ACTION_SHAPE, dtype=np.float16)  # (n,32,50,14)
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
        end_c = np.clip(end, 0, n - 1)                                      # safe gather idx
        # True episode terminal iff the next-state carries a TERMINAL reward (success 0.0 /
        # failure penalty), NOT the living cost. This is robust to the ~1000-row row-group
        # boundaries: every episode spans several groups, so a boundary frame holds a living
        # reward and is NOT mis-flagged. (The old `end == Li` test treated every group-end as
        # a terminal -> ~half of all flagged terminals were false, bootstrapping from a living
        # value (-1e-4) instead of the critic. Verified: reward in {0.0, -0.5} only at true ends.)
        is_term_frame = (rew >= -1e-6) | (rew <= -0.05)                     # (n,) bool
        term = ((valid > 0) & is_term_frame[end_c]).astype(np.float32)      # (S,P) genuine terminal

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
            "mc_return": mc[starts],                                        # (S,) start-state V^beta (Cal-QL floor)
            "term": term,                                                   # (S,P)
            "valid": valid,                                                 # (S,P)
        }
        return out

    def iter_bootstrap_batches(self, batch_size: int, prefixes: Sequence[int],
                               seed: int = 0, drop_last: bool = True,
                               shard: tuple[int, int] = (0, 1)) -> Iterator[dict]:
        """Yield AQC-TD bootstrap batches (needs ``base_action``; set it in the constructor).

        Args:
            batch_size: transitions per batch.
            prefixes: 1-indexed prefix lengths h to bootstrap on (the subsample grid),
                e.g. [1, 5, 10, 20, 35, 50]. Fewer => cheaper (the H=50 cost knob).
            shard: ``(i, n)`` -- take every n-th row-group of the shuffled work list,
                starting at i. Used by the multiprocess loader (vla_loader) so n worker
                processes stream DISJOINT data; the shuffle uses ``seed`` only, so all
                shards agree on the partition. (0, 1) = everything (default).
        """
        prefixes = np.asarray(sorted(set(int(h) for h in prefixes)), dtype=np.int64)
        assert prefixes.min() >= 1 and prefixes.max() <= self.horizon
        rng = np.random.default_rng(seed)
        cols = ["rl_token", "action", "reward", "mc_return", "base_action",
                "episode_index", "observation.commander_state"]
        work = [(f, g) for f in self.files
                for g in range(self._readers[f].metadata.num_row_groups)]
        rng.shuffle(work)
        si, sn = shard
        if sn > 1:
            work = work[si::sn]
            rng = np.random.default_rng((seed, si))   # decorrelate pool permutations
        pool, pool_n = [], 0

        def read_form(fg):
            # Fresh ParquetFile per read: read_row_group on a SHARED handle is not thread-safe;
            # the footer read is cheap. pyarrow decode releases the GIL, so concurrent reads of
            # the heavy base_action column (~828ms/group) actually parallelize across threads.
            f, g = fg
            t = pq.ParquetFile(f).read_row_group(g, columns=cols)
            return self._bootstrap_samples_from_table(t, prefixes)

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

        # Read shuffle_buffer_groups row-groups concurrently (num_workers threads), then emit.
        # The executor is created AND shut down within each chunk (before any yield) so no
        # worker threads stay alive across a yield -> avoids the abandoned-generator join error
        # at interpreter shutdown if the loop is broken early.
        n_workers = max(1, self.num_workers)
        for i in range(0, len(work), self.shuffle_buffer_groups):
            chunk = work[i:i + self.shuffle_buffer_groups]
            with ThreadPoolExecutor(max_workers=min(n_workers, len(chunk))) as ex:
                samples = list(ex.map(read_form, chunk))
            for s in samples:
                if s is not None:
                    pool.append(s); pool_n += len(s["valid"])
            yield from emit()

    def iter_batches(self, batch_size: int, seed: int = 0,
                     drop_last: bool = True,
                     shard: tuple[int, int] = (0, 1)) -> Iterator[dict]:
        """Yield shuffled training batches by streaming row-groups.

        Reads ``shuffle_buffer_groups`` row-groups into a pool, shuffles the pooled
        samples, and emits ``batch_size`` chunks until the pool drains, then refills.
        ``shard=(i, n)``: stream every n-th row-group only (see iter_bootstrap_batches).
        """
        rng = np.random.default_rng(seed)
        cols = self._row_cols()
        # (file, group) work-list, shuffled each epoch.
        work = [(f, g) for f in self.files
                for g in range(self._readers[f].metadata.num_row_groups)]
        rng.shuffle(work)
        si, sn = shard
        if sn > 1:
            work = work[si::sn]
            rng = np.random.default_rng((seed, si))   # decorrelate pool permutations

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
