"""Multiprocess batch loading for VLA critic training (the openpi pattern).

The training pipeline is loader-bound (~217ms/batch host-side assembly vs ~135ms GPU step
at B=256), and the in-process thread prefetch can't scale past the GIL'd serial parts
(pool concat / permutation / batch gather). This module scales the HOST side the way
openpi does (src/openpi/training/data_loader.py): torch is used ONLY for its DataLoader
worker machinery — the critic model and update stay pure JAX.

  * ``VLABatchIterable``: a torch IterableDataset. Each worker PROCESS constructs its own
    ``VLALeRobotDataset`` (ParquetFile handles aren't picklable) and runs the existing
    row-group-streaming generator on a DISJOINT shard of the shuffled work list
    (``shard=(worker_id, num_workers)``), yielding COMPLETE numpy batches. With
    ``DataLoader(batch_size=None)`` torch does no collation; the main process just
    round-robins ready batches across workers => N processes ~= N x throughput, no GIL.
  * ``make_torch_loader``: the DataLoader wiring (spawn context, persistent workers,
    per-worker prefetch), mirroring openpi's TorchDataLoader.

The consumer (vla_train.shard_batch) scatters each numpy batch over the data-parallel
device mesh with ``jax.make_array_from_process_local_data``. This file must NOT import
jax: worker processes import it to unpickle the iterable, and they should stay jax-free.
"""

import multiprocessing

import torch.utils.data as tud

from vla_data import VLALeRobotDataset


class VLABatchIterable(tud.IterableDataset):
    """Infinite stream of ready-made training batches, sharded per worker process.

    Args:
        ds_kwargs: constructor kwargs for VLALeRobotDataset (must be picklable).
        batch_size: GLOBAL batch size — every yielded batch has this leading dim, so the
            jit'd train step never recompiles; the mesh split happens downstream.
        target_kind: 'td' -> iter_bootstrap_batches, 'mc' -> iter_batches.
        prefixes: TD prefix grid (ignored for 'mc').
        seed: base seed; epoch e re-shuffles with seed + 100_000*e (same partition on
            every worker, disjoint shards within it).
    """

    def __init__(self, ds_kwargs: dict, batch_size: int, target_kind: str,
                 prefixes, seed: int):
        self.ds_kwargs = ds_kwargs
        self.batch_size = batch_size
        self.target_kind = target_kind
        self.prefixes = tuple(prefixes)
        self.seed = int(seed)

    def __iter__(self):
        info = tud.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        ds = VLALeRobotDataset(**self.ds_kwargs)
        epoch = 0
        while True:                                    # infinite: consumer never restarts us
            seed = self.seed + 100_000 * epoch
            if self.target_kind == "mc":
                yield from ds.iter_batches(self.batch_size, seed=seed, shard=(wid, nw))
            else:
                yield from ds.iter_bootstrap_batches(self.batch_size, self.prefixes,
                                                     seed=seed, shard=(wid, nw))
            epoch += 1


def _identity(x):
    """Module-level (picklable) no-op collate: batches arrive fully formed as numpy."""
    return x


def make_torch_loader(iterable: VLABatchIterable, num_workers: int,
                      prefetch_factor: int = 2) -> tud.DataLoader:
    """DataLoader over pre-batched items: N worker processes, no torch collation.

    spawn (not fork): the parent holds JAX/CUDA state that must not be forked.
    persistent_workers: keep the (slow, jax-importing) worker bootstrap a one-time cost.
    """
    if num_workers <= 0:                               # in-process fallback (debug)
        return tud.DataLoader(iterable, batch_size=None, num_workers=0,
                              collate_fn=_identity)
    return tud.DataLoader(
        iterable,
        batch_size=None,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        persistent_workers=True,
        prefetch_factor=max(1, prefetch_factor),
        collate_fn=_identity,
    )
