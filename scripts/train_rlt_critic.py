"""Training entry for VLA AQC critic learning.

Ties together vla_config (what to run) + vla_data (the data) + vla_aqc (the critic update),
with run management modelled on openpi: a self-documenting run dir on lustre, config.json
dump, CSV + W&B logging, periodic checkpoints (save/keep/resume), and offline eval
(value calibration vs mc_return). No environment rollout (frozen VLA, offline critic).

Run a named preset:
    python vla_train.py --config vla_aqc_td_a51 --exp_name my_run
Quick throughput probe (no checkpoints):
    python vla_train.py --config vla_aqc_td_a51 --timing_steps 100
"""

import dataclasses
import json
import pathlib
import sys
import time
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import numpy as np
import flax.serialization as fs
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm import tqdm

from openpi.rlt_critic.config import VLAAQCConfig, get_config
from openpi.rlt_critic.data import VLALeRobotDataset, prefetch
from openpi.rlt_critic.data_mm import make_dataset
from openpi.rlt_critic.agent import VLACriticTrainer


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(ckpt_dir: pathlib.Path, step: int, params, opt_state):
    d = ckpt_dir / f"step_{step:08d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.msgpack").write_bytes(fs.to_bytes(jax.device_get(params)))
    (d / "opt_state.msgpack").write_bytes(fs.to_bytes(jax.device_get(opt_state)))
    (d / "meta.json").write_text(json.dumps({"step": step}))
    return d


def list_checkpoints(ckpt_dir: pathlib.Path):
    if not ckpt_dir.exists():
        return []
    return sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*") if p.is_dir())


def load_checkpoint(ckpt_dir: pathlib.Path, step: int, params, opt_state):
    d = ckpt_dir / f"step_{step:08d}"
    params = fs.from_bytes(params, (d / "params.msgpack").read_bytes())
    opt_state = fs.from_bytes(opt_state, (d / "opt_state.msgpack").read_bytes())
    return params, opt_state


def prune_checkpoints(ckpt_dir: pathlib.Path, keep_period: Optional[int]):
    steps = list_checkpoints(ckpt_dir)
    if not steps:
        return
    latest = steps[-1]
    for s in steps[:-1]:
        if keep_period and s % keep_period == 0:
            continue
        if s == latest:
            continue
        import shutil
        shutil.rmtree(ckpt_dir / f"step_{s:08d}", ignore_errors=True)


# --------------------------------------------------------------------------- logging
class RunLogger:
    def __init__(self, cfg: VLAAQCConfig):
        self.cfg = cfg
        self.csv_path = cfg.run_dir / "metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header = None
        self._f = open(self.csv_path, "a")
        self.wandb = None
        if cfg.wandb_enabled:
            try:
                import wandb
                self.wandb = wandb.init(
                    project=cfg.wandb_project, entity=cfg.wandb_entity,
                    name=cfg.exp, group=cfg.name, config=cfg.to_dict(),
                    dir=str(cfg.run_dir), tags=[cfg.run_name[:64]])
            except Exception as e:
                print(f"[wandb] disabled ({e})")

    def log(self, step, metrics: dict, prefix="train"):
        row = {"step": step, **{f"{prefix}/{k}": float(v) for k, v in metrics.items()}}
        if self._header is None:
            self._header = list(row)
            self._f.write(",".join(self._header) + "\n")
        self._f.write(",".join(str(row.get(k, "")) for k in self._header) + "\n")
        self._f.flush()
        if self.wandb is not None:
            self.wandb.log(row, step=step)

    def log_image(self, step, key, fig):
        """Log a matplotlib figure to W&B (e.g. eval/value_curves). No local save."""
        if self.wandb is not None:
            import wandb
            self.wandb.log({key: wandb.Image(fig)}, step=step)

    def close(self):
        self._f.close()
        if self.wandb is not None:
            self.wandb.finish()


def reward_scale(cfg: VLAAQCConfig) -> float:
    """Scalar applied to reward/return so values match the support (reward_norm only).

    The dataset return range is ~[-0.5, 0]; mapping to a fixed [v_min, 0] uses scale =
    |v_min| / 0.5. For support_mode='fixed' (default) this is 1.0 (no scaling).
    """
    if cfg.dist.support_mode == "reward_norm":
        return abs(cfg.dist.v_min) / 0.5
    return 1.0


# Offline eval = trajectory value-curve visualization (no env rollout): query the critic
# along recorded success/failure episodes and compare to mc_return. See vla_eval.py.


# --------------------------------------------------------------------------- train
def _warm_page_cache(memmap_dir) -> tuple:
    """Sequentially read every .dat into the OS page cache so the loader's RANDOM frame gathers
    hit RAM rather than cold lustre. A FRESH (just-built) memmap is uncached -> random 1024-frame
    gathers cost ~200ms over the network FS, and under DDP lockstep that collapses throughput
    (the slowest rank gates every all-reduce: 0.2 it/s, GPUs 0/100/0/100). One sequential pass at
    lustre seq-BW (~25s for 167GB) fixes it; with abundant RAM the pages then stay resident."""
    import glob
    t = time.time(); total = 0
    for f in sorted(glob.glob(str(pathlib.Path(memmap_dir) / "*.dat"))):
        with open(f, "rb", buffering=0) as fh:
            while True:
                chunk = fh.read(1 << 28)            # 256MB sequential reads
                if not chunk:
                    break
                total += len(chunk)
    return total, time.time() - t


def train(cfg: VLAAQCConfig, timing_steps: int = 0, resume: bool = False, warm_cache: bool = True):
    # Multi-process DDP (one OS process per GPU, jax.distributed): n_proc>1 here. The chief
    # (process 0) owns all host-side side effects -- logging, checkpoints, the one-time memmap
    # build -- while every process drives its single local GPU and XLA inserts the cross-process
    # grad all-reduce. n_proc==1 (single-process multi-GPU, or 1 GPU) is the unchanged old path.
    n_proc = jax.process_count()
    proc_id = jax.process_index()
    chief = (proc_id == 0)
    seed_off = proc_id * 1_000_003               # decorrelate each process's data shard sampling
    print(f"=== run: {cfg.run_name} ===")
    print(f"    dir: {cfg.run_dir}")
    if chief:
        cfg.save()                               # dump config.json (self-documenting run)

    trainer = VLACriticTrainer(cfg, seed=cfg.seed)
    print(f"    critic params: {trainer.num_params()/1e6:.2f}M  (n_embd={cfg.arch.n_embd}, "
          f"{cfg.arch.num_layers}L)  target_kind={cfg.td.target_kind}  "
          f"support=[{cfg.dist.v_min},{cfg.dist.v_max}]  discount={cfg.td.discount}")
    step_fn = trainer.make_train_step()
    params, opt_state = trainer.params, trainer.opt_state
    start_step = 0
    if resume and list_checkpoints(cfg.checkpoint_dir):
        start_step = list_checkpoints(cfg.checkpoint_dir)[-1]
        params, opt_state = load_checkpoint(cfg.checkpoint_dir, start_step, params, opt_state)
        print(f"    resumed from step {start_step}")
    target_params = params   # EMA target network state (== online params when td.target_tau == 0)

    # Data-parallel mesh (openpi-style, single process): params replicated on every GPU,
    # each batch split along its leading axis -> jit inserts the grad all-reduce itself.
    # With 1 device this is the identity setup, so the single-GPU path is unchanged.
    devices = jax.devices()                          # GLOBAL devices (across all processes)
    n_dev = len(devices)
    assert cfg.optim.batch_size % n_dev == 0, \
        f"batch_size {cfg.optim.batch_size} must be divisible by {n_dev} devices"
    # Each process feeds only its OWN shard of the global batch (B/n_proc); make_array_from_
    # process_local_data then stitches the per-process shards into the global sharded array.
    # n_proc==1 => local_B == global B (old single-process path, unchanged).
    local_B = cfg.optim.batch_size // n_proc
    mesh = Mesh(np.asarray(devices), ("dp",))
    data_sharding = NamedSharding(mesh, PartitionSpec("dp"))
    repl_sharding = NamedSharding(mesh, PartitionSpec())
    params = jax.device_put(params, repl_sharding)
    opt_state = jax.device_put(opt_state, repl_sharding)
    target_params = jax.device_put(target_params, repl_sharding)
    print(f"    devices: {n_dev} x {devices[0].device_kind}  procs={n_proc} (local={len(jax.local_devices())})  "
          f"(global B={cfg.optim.batch_size} -> {local_B}/proc -> {cfg.optim.batch_size // n_dev}/device, "
          f"loader_processes={cfg.loader_processes})")

    # preload (whole dataset -> RAM) only makes sense on the single-process path; under
    # loader_processes>0 each worker would duplicate the cache, so ignore it there.
    preload = cfg.preload
    if preload and cfg.loader_processes > 0:
        print("    [preload] ignored: requires loader_processes=0 (would duplicate per worker)")
        preload = False
    ds_kwargs = dict(
        root=cfg.data_root, horizon=cfg.horizon,
        commander_filter=set(cfg.commander_filter) if cfg.commander_filter else None,
        include_base_action=(cfg.td.target_kind == "td"),
        shuffle_buffer_groups=cfg.shuffle_buffer_groups,
        mc_gamma=cfg.td.mc_gamma, discount=cfg.td.discount,
        relabel_living=cfg.reward.relabel_living,
        relabel_fail=cfg.reward.relabel_fail,
        num_workers=cfg.num_workers, bootstrap_subset=cfg.td.bootstrap_subset,
        n_step=cfg.td.n_step, preload=preload, memmap_dir=cfg.memmap_dir)
    if cfg.memmap_dir:
        # Auto-build the memmap on first run (one-time) so a SINGLE `train` command handles
        # preprocessing -> no separate step. Runs in this main process before the loader
        # workers spawn; subsequent runs reuse it (idempotent: skips if meta.json exists).
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # scripts/ -> preprocess_memmap
        from preprocess_memmap import build_memmap, memmap_ready
        if not memmap_ready(cfg.memmap_dir):
            # Only the chief builds (else N processes corrupt the same files); the others block
            # on the barrier until it's done. (For a fresh, slow build prefer pre-building once
            # via `--build_memmap_only` in the launcher so this barrier is instant.)
            if chief:
                print(f"    [memmap] not found at {cfg.memmap_dir} -> building from {cfg.data_root} (one-time)...")
                build_memmap(cfg.data_root, cfg.memmap_dir, workers=cfg.num_workers)
            if n_proc > 1:
                multihost_utils.sync_global_devices("rlt_memmap_built")
    ds = make_dataset(ds_kwargs)           # MemmapVLADataset if cfg.memmap_dir set, else parquet
    if cfg.memmap_dir:
        print(f"    [memmap] fast index loader on {cfg.memmap_dir}")
    print(f"    data: {ds.summary()}")
    if cfg.memmap_dir and warm_cache:
        # Pull the memmap into the page cache up front so random gathers hit RAM. The chief warms
        # it (the cache is shared across all DDP processes); the others wait on the barrier. Without
        # this, a freshly-built dataset trains at ~0.2 it/s until the cache slowly self-warms.
        if chief:
            gb, dt = _warm_page_cache(cfg.memmap_dir)
            print(f"    [memmap] page-cache warm: {gb/1e9:.0f} GB in {dt:.0f}s (random gathers now hit RAM)")
        if n_proc > 1:
            multihost_utils.sync_global_devices("rlt_memmap_warm")
    rscale = reward_scale(cfg)

    # warmup-skip: during the beta=0 MC warmup, train a pure-MC step that never reads base_action.
    # A separate no-base_action handle (cheap; decoded on the fly even under preload) feeds it.
    warmup_skip = (cfg.td.target_kind == "td" and cfg.td.mc_floor
                   and cfg.td.warmup_skip and cfg.td.mc_warmup_steps > start_step)
    mc_ds = mc_step_fn = None
    if warmup_skip:
        mc_ds = make_dataset({**ds_kwargs, "include_base_action": False, "preload": False})
        mc_step_fn = trainer.make_train_step(kind="mc")
        print(f"    [warmup-skip] beta=0 for steps <{cfg.td.mc_warmup_steps}: pure-MC loader "
              f"(no base_action), then switch to the TD loader")

    def td_batch_iter(seed):
        # Host-side pipeline (the throughput bottleneck: ~217ms/batch assembly vs ~135ms
        # GPU step at B=256). loader_processes>0: N worker PROCESSES each stream a disjoint
        # row-group shard and yield full global batches (openpi pattern; vla_loader.py) --
        # this is what lets a bigger multi-GPU batch keep the same it/s. 0 = legacy
        # single background thread.
        if cfg.loader_processes > 0:
            from openpi.rlt_critic.loader import VLABatchIterable, make_torch_loader
            it = VLABatchIterable(ds_kwargs, local_B, cfg.td.target_kind,
                                  cfg.td.prefixes, seed)
            yield from make_torch_loader(it, cfg.loader_processes, cfg.prefetch_depth)
            return
        if cfg.td.target_kind == "mc":
            gen = ds.iter_batches(local_B, seed=seed)
        else:
            gen = ds.iter_bootstrap_batches(local_B, cfg.td.prefixes, seed=seed)
        yield from prefetch(gen, depth=cfg.prefetch_depth)

    def mc_warmup_iter(seed):
        # In-process MC stream (no base_action -> cheap, GPU-bound ~44 it/s; single-thread loader
        # keeps up, so no worker processes needed here regardless of loader_processes).
        yield from prefetch(mc_ds.iter_batches(local_B, seed=seed),
                            depth=cfg.prefetch_depth)

    def shard_batch(b):
        """numpy loader batch -> jax arrays split over the dp mesh (prefixes replicated)."""
        jb = {k: jax.make_array_from_process_local_data(data_sharding, v)
              for k, v in b.items() if k != "prefixes"}
        pf = (jax.device_put(np.asarray(b["prefixes"], np.int32), repl_sharding)
              if "prefixes" in b else None)
        return jb, pf

    def scale_batch(b):
        if rscale != 1.0:
            for k in ("mc_return", "cum_reward", "next_mc_return"):
                if k in b:
                    b[k] = b[k] * rscale
        return b

    logger = RunLogger(cfg) if (chief and not timing_steps) else None   # only chief logs/ckpts
    # Offline eval (trajectory value-curve viz vs mc_return) is OPTIONAL: it only runs if an
    # eval module (openpi.rlt_critic.eval_curves) is present. Stage-1 critic training doesn't
    # need it, so skip cleanly when it's absent.
    vla_eval, eval_set = None, None
    # Offline eval (value-curve viz) runs single-device EAGER ops on the chief only (net.apply on
    # a host copy of the replicated params -> no cross-process collective).  Under multi-process
    # DDP we keep the collective lockstep in sync by bracketing the chief's eval with
    # sync_global_devices barriers (in the loop below), so it is safe to enable for n_proc>1.
    if logger:
        try:
            from openpi.rlt_critic import eval_curves as vla_eval
            eval_set = vla_eval.build_eval_set(ds, n_success=cfg.eval_n_success,
                                               n_fail=cfg.eval_n_fail,
                                               n_intervention=cfg.eval_n_intervention, seed=cfg.seed)
            import collections as _c
            _cc = _c.Counter(e.get("category", "?") for e in eval_set)
            print(f"    eval set: {dict(_cc)} episodes (cached)")
        except Exception as e:
            print(f"    [eval disabled] {type(e).__name__}: {e}")

    # Phase: during [start_step, mc_warmup_steps) run the pure-MC warmup loader+step (warmup-skip);
    # otherwise the TD loader+step. The boundary switch swaps both the iterator and the train step.
    warmup_end = cfg.td.mc_warmup_steps if warmup_skip else 0
    in_warmup = start_step < warmup_end
    it = (mc_warmup_iter if in_warmup else td_batch_iter)(cfg.seed + start_step + seed_off)
    cur_step_fn = mc_step_fn if in_warmup else step_fn
    n_steps = timing_steps or cfg.optim.num_train_steps
    t0 = time.time(); t_log = t0

    is_tty = sys.stderr.isatty()
    end_step = start_step + n_steps
    pbar = tqdm(range(start_step, end_step), desc=cfg.name,
                dynamic_ncols=True, smoothing=0.1, disable=not is_tty or not chief)
    for step in pbar:
        if in_warmup and step >= warmup_end:           # warmup done -> switch to the TD loader/step
            in_warmup = False
            it = td_batch_iter(cfg.seed + step + seed_off)
            cur_step_fn = step_fn
            target_params = params                      # resync EMA target to the warmed-up online weights
            pbar.write(f"  [warmup-skip] step {step}: warmup done -> TD (base_action) loader engaged")
        try:
            b = scale_batch(next(it))
        except StopIteration:
            it = (mc_warmup_iter if in_warmup else td_batch_iter)(cfg.seed + step + seed_off)
            b = scale_batch(next(it))
        jb, pf = shard_batch(b)
        # ReLU-blend MC-warmup coefficient for this step (jnp scalar -> traced, no recompile).
        beta = jnp.asarray(cfg.mc_blend_beta(step), jnp.float32)
        params, opt_state, target_params, info = cur_step_fn(params, opt_state, target_params, jb, pf, beta)

        if timing_steps and step == start_step:
            info["critic_loss"].block_until_ready()
            t0 = time.time(); t_log = t0   # exclude compile from timing
        if logger and (step % cfg.log_interval == 0):
            sps = cfg.log_interval / max(time.time() - t_log, 1e-6); t_log = time.time()
            m = {k: float(v) for k, v in info.items()}; m["steps_per_sec"] = sps
            logger.log(step, m)                                # CSV + W&B (full record)
            if is_tty:
                pbar.set_postfix(loss=f"{m['critic_loss']:.4f}",
                                 q=f"{m.get('q_mean', 0):.4f}", refresh=False)
            else:                                              # non-tty heartbeat line
                print(f"  step {step:>7}/{end_step} loss={m['critic_loss']:.4f} "
                      f"q={m.get('q_mean', 0):.4f} {sps:.1f} it/s", flush=True)
        # Offline value-curve eval.  The trigger is purely STEP-based so every process evaluates it
        # identically and enters the barriers together (collective lockstep preserved).  Only the
        # chief runs the actual single-device eval + W&B log between the barriers; the other procs
        # idle at eval_post until it returns.  device_get of the replicated params is a local d2h
        # copy and net.apply is eager single-device -> NO collective inside the bracket.
        if step > start_step and step % cfg.eval_interval == 0:
            if n_proc > 1:
                multihost_utils.sync_global_devices(f"eval_pre_{step}")
            if logger and vla_eval is not None and eval_set:
                import matplotlib.pyplot as plt
                curves = vla_eval.compute_curves(trainer, jax.device_get(params), eval_set,
                                                 cfg.horizon, cfg.action_dim)
                fig = vla_eval.plot_curves(curves, cfg.dist.v_min, cfg.dist.v_max)
                logger.log_image(step, "eval/value_curves", fig)
                plt.close(fig)
                pbar.write(f"  [eval {step}] logged eval/value_curves to W&B")
            if n_proc > 1:
                multihost_utils.sync_global_devices(f"eval_post_{step}")
        if logger and step > start_step and step % cfg.save_interval == 0:
            save_checkpoint(cfg.checkpoint_dir, step, params, opt_state)
            prune_checkpoints(cfg.checkpoint_dir, cfg.keep_period)
    pbar.close()

    if timing_steps:
        dt = time.time() - t0
        print(f"\n=== timing: {timing_steps-1} steps in {dt:.1f}s -> {(timing_steps-1)/dt:.2f} it/s")
        print(f"    => 500k steps ~= {500_000/max((timing_steps-1)/dt,1e-9)/3600:.1f} h")
    elif chief:
        save_checkpoint(cfg.checkpoint_dir, start_step + n_steps, params, opt_state)
        logger.close()
        print("=== done ===")


def main():
    import os
    # Multi-process DDP launch (one process per GPU): the launcher sets these env vars and
    # CUDA_VISIBLE_DEVICES=<one gpu> per process. jax.distributed.initialize MUST run before any
    # jax backend use (first jax.devices()/jit). Absent the env (single-process), this is skipped
    # and everything behaves exactly as before.
    _nproc = int(os.environ.get("RLT_NUM_PROCESSES", "1"))
    if _nproc > 1:
        jax.distributed.initialize(
            coordinator_address=os.environ.get("RLT_COORDINATOR", "127.0.0.1:29500"),
            num_processes=_nproc,
            process_id=int(os.environ["RLT_PROCESS_ID"]))
    import tyro
    @dataclasses.dataclass
    class Args:
        config: str = "vla_aqc_warmup"     # registry key (see config.CONFIGS)
        task: str = ""                     # override dataset task (see config.TASKS); "" = config default
        data_root: str = ""                # override dataset path (config.data_root_override); "" = task default
        exp_name: str = ""
        seed: int = 0
        timing_steps: int = 0              # >0 => throughput probe, no checkpoints
        resume: bool = False
        batch_size: int = 0                # >0 overrides optim.batch_size (e.g. 1024)
        lr: float = 0.0                    # >0 overrides optim.lr (scale with batch if desired)
        mc_floor: Optional[bool] = None    # override td.mc_floor (ReLU-blend floor on/off)
        mc_warmup_steps: int = -1          # >=0 overrides td.mc_warmup_steps (beta=0 phase length)
        mc_ramp_steps: int = -1            # >=0 overrides td.mc_ramp_steps (beta 0->1 ramp length)
        agg_beta: float = 0.0              # >0 overrides td.agg_beta (softmax temperature sweep)
        loader_processes: int = -1         # >=0 overrides cfg.loader_processes (0 = thread loader)
        target_tau: float = -1.0           # >=0 overrides td.target_tau (EMA target net; 0 = no target net)
        bootstrap_subset: int = -1         # >=0 overrides td.bootstrap_subset (random N-of-32 candidate subset; 0 = all)
        n_step: int = -1                   # >=0 overrides td.n_step (N-step return; 0 = standard h-step backup)
        preload: Optional[bool] = None     # override cfg.preload (decode whole dataset into RAM; loader_processes=0 only)
        warmup_skip: Optional[bool] = None # override td.warmup_skip (pure-MC beta=0 phase, no base_action read)
        checkpoint_base_dir: str = ""      # override where runs are written (run dir = base/<name>/<exp>); "" = config default
        memmap_dir: str = ""               # fast index loader: a path, or "auto" (= <data_root>_memmap, derived from the config's dataset); "" = parquet
        build_memmap_only: bool = False    # build the memmap (if missing) then exit -- pre-build once before a multi-process DDP launch
        no_warm: bool = False              # skip the startup page-cache warm (use only if the memmap is already resident in RAM)
    args = tyro.cli(Args)
    cfg = get_config(args.config)
    if args.loader_processes >= 0:
        cfg = dataclasses.replace(cfg, loader_processes=args.loader_processes)
    if args.preload is not None:
        cfg = dataclasses.replace(cfg, preload=args.preload)
    if args.batch_size > 0 or args.lr > 0:
        cfg = dataclasses.replace(cfg, optim=dataclasses.replace(
            cfg.optim,
            batch_size=args.batch_size or cfg.optim.batch_size,
            lr=args.lr or cfg.optim.lr))
    if (args.mc_floor is not None or args.agg_beta > 0 or args.target_tau >= 0
            or args.bootstrap_subset >= 0 or args.n_step >= 0 or args.mc_warmup_steps >= 0
            or args.mc_ramp_steps >= 0 or args.warmup_skip is not None):
        cfg = dataclasses.replace(cfg, td=dataclasses.replace(
            cfg.td,
            mc_floor=cfg.td.mc_floor if args.mc_floor is None else args.mc_floor,
            agg_beta=args.agg_beta or cfg.td.agg_beta,
            target_tau=cfg.td.target_tau if args.target_tau < 0 else args.target_tau,
            bootstrap_subset=cfg.td.bootstrap_subset if args.bootstrap_subset < 0 else args.bootstrap_subset,
            n_step=cfg.td.n_step if args.n_step < 0 else args.n_step,
            mc_warmup_steps=cfg.td.mc_warmup_steps if args.mc_warmup_steps < 0 else args.mc_warmup_steps,
            mc_ramp_steps=cfg.td.mc_ramp_steps if args.mc_ramp_steps < 0 else args.mc_ramp_steps,
            warmup_skip=cfg.td.warmup_skip if args.warmup_skip is None else args.warmup_skip))
    cfg = dataclasses.replace(cfg, seed=args.seed, exp_name=args.exp_name or "",
                              task=args.task or cfg.task,
                              data_root_override=args.data_root or cfg.data_root_override,
                              checkpoint_base_dir=args.checkpoint_base_dir or cfg.checkpoint_base_dir,
                              memmap_dir=args.memmap_dir or cfg.memmap_dir)
    if cfg.memmap_dir == "auto":
        # Derive the memmap location FROM the dataset (cfg.data_root = data_root_override or
        # TASKS[task]) -> <data_root>_memmap. So just `--config X --memmap_dir auto` builds + reads
        # the config's own dataset; no need to repeat the path. (Override with an explicit path,
        # e.g. --memmap_dir /dev/shm/foo, to put it on tmpfs.)
        cfg = dataclasses.replace(cfg, memmap_dir=cfg.data_root.rstrip("/") + "_memmap")
    if args.build_memmap_only:
        # Pre-build the memmap once (single process) so the DDP launch never races/barriers on it.
        assert cfg.memmap_dir, "--build_memmap_only requires --memmap_dir (path or 'auto')"
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from preprocess_memmap import build_memmap, memmap_ready
        if memmap_ready(cfg.memmap_dir):
            print(f"[memmap] ready at {cfg.memmap_dir} (reuse)")
        else:
            build_memmap(cfg.data_root, cfg.memmap_dir, workers=cfg.num_workers)
        gb, dt = _warm_page_cache(cfg.memmap_dir)      # warm now so the DDP procs start hot
        print(f"[memmap] page-cache warm: {gb/1e9:.0f} GB in {dt:.0f}s")
        return
    train(cfg, timing_steps=args.timing_steps, resume=args.resume, warm_cache=not args.no_warm)


if __name__ == "__main__":
    main()
