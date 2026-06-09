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
import time
from typing import Optional

import jax
import numpy as np
import flax.serialization as fs

from vla_config import VLAAQCConfig, get_config
from vla_data import VLALeRobotDataset
from vla_aqc import VLACriticTrainer, to_jax_batch


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(ckpt_dir: pathlib.Path, step: int, params, target_params, opt_state):
    d = ckpt_dir / f"step_{step:08d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.msgpack").write_bytes(fs.to_bytes(jax.device_get(params)))
    (d / "target.msgpack").write_bytes(fs.to_bytes(jax.device_get(target_params)))
    (d / "opt_state.msgpack").write_bytes(fs.to_bytes(jax.device_get(opt_state)))
    (d / "meta.json").write_text(json.dumps({"step": step}))
    return d


def list_checkpoints(ckpt_dir: pathlib.Path):
    if not ckpt_dir.exists():
        return []
    return sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*") if p.is_dir())


def load_checkpoint(ckpt_dir: pathlib.Path, step: int, params, target_params, opt_state):
    d = ckpt_dir / f"step_{step:08d}"
    params = fs.from_bytes(params, (d / "params.msgpack").read_bytes())
    target_params = fs.from_bytes(target_params, (d / "target.msgpack").read_bytes())
    opt_state = fs.from_bytes(opt_state, (d / "opt_state.msgpack").read_bytes())
    return params, target_params, opt_state


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


def make_eval_fn(trainer: VLACriticTrainer, cfg: VLAAQCConfig):
    """Offline eval: critic loss + value-calibration (predicted full-prefix Q vs mc_return)."""
    import jax.numpy as jnp
    def evaluate(params, batches):
        losses, qs, mcs = [], [], []
        for b in batches:
            jb, pf = to_jax_batch(b)
            if cfg.td.target_kind == "mc":
                loss, info = trainer.critic_loss_mc(params, jb)
            else:
                loss, info = trainer.critic_loss_td(params, params, jb, pf)
            losses.append(float(info["critic_loss"]))
            # calibration on the executed-chunk full-prefix value vs mc_return
            logits = trainer.net.apply(params, jb["observations"], jb["action_chunks"])
            qfull = trainer.from_probs(jax.nn.softmax(logits, -1))[:, :, -1].mean(0)  # (B,)
            qs.append(np.asarray(qfull)); mcs.append(np.asarray(jb.get("mc_return",
                                                       jb.get("next_mc_return"))[..., -1]
                                                       if "mc_return" not in jb else jb["mc_return"]))
        q = np.concatenate(qs); mc = np.concatenate(mcs)
        corr = float(np.corrcoef(q, mc)[0, 1]) if len(q) > 2 else 0.0
        return {"eval_loss": float(np.mean(losses)), "q_vs_mc_corr": corr,
                "q_mean": float(q.mean()), "mc_mean": float(mc.mean())}
    return evaluate


# --------------------------------------------------------------------------- train
def train(cfg: VLAAQCConfig, timing_steps: int = 0, resume: bool = False):
    print(f"=== run: {cfg.run_name} ===")
    print(f"    dir: {cfg.run_dir}")
    cfg.save()                                   # dump config.json (self-documenting run)

    trainer = VLACriticTrainer(cfg, seed=cfg.seed)
    print(f"    critic params: {trainer.num_params()/1e6:.2f}M  (n_embd={cfg.arch.n_embd}, "
          f"{cfg.arch.num_layers}L)  target_kind={cfg.td.target_kind}")
    step_fn = trainer.make_train_step()
    params, target_params, opt_state = trainer.params, trainer.target_params, trainer.opt_state
    start_step = 0
    if resume and list_checkpoints(cfg.checkpoint_dir):
        start_step = list_checkpoints(cfg.checkpoint_dir)[-1]
        params, target_params, opt_state = load_checkpoint(
            cfg.checkpoint_dir, start_step, params, target_params, opt_state)
        print(f"    resumed from step {start_step}")

    ds = VLALeRobotDataset(cfg.data_root, horizon=cfg.horizon,
                           commander_filter=set(cfg.commander_filter) if cfg.commander_filter else None,
                           include_base_action=(cfg.td.target_kind == "td"),
                           shuffle_buffer_groups=cfg.shuffle_buffer_groups)
    print(f"    data: {ds.summary()}")
    rscale = reward_scale(cfg)

    def batch_iter(seed):
        if cfg.td.target_kind == "mc":
            yield from ds.iter_batches(cfg.optim.batch_size, seed=seed)
        else:
            yield from ds.iter_bootstrap_batches(cfg.optim.batch_size, cfg.td.prefixes, seed=seed)

    def scale_batch(b):
        if rscale != 1.0:
            for k in ("mc_return", "cum_reward", "next_mc_return"):
                if k in b:
                    b[k] = b[k] * rscale
        return b

    logger = None if timing_steps else RunLogger(cfg)
    evaluate = make_eval_fn(trainer, cfg)

    it = batch_iter(cfg.seed + start_step)
    n_steps = timing_steps or cfg.optim.num_train_steps
    t0 = time.time(); t_log = t0
    for step in range(start_step, start_step + n_steps):
        try:
            b = scale_batch(next(it))
        except StopIteration:
            it = batch_iter(cfg.seed + step); b = scale_batch(next(it))
        jb, pf = to_jax_batch(b)
        params, target_params, opt_state, info = step_fn(params, target_params, opt_state, jb, pf)

        if timing_steps and step == start_step:
            info["critic_loss"].block_until_ready()
            t0 = time.time(); t_log = t0   # exclude compile from timing
        if logger and (step % cfg.log_interval == 0):
            sps = cfg.log_interval / max(time.time() - t_log, 1e-6); t_log = time.time()
            m = {k: float(v) for k, v in info.items()}; m["steps_per_sec"] = sps
            logger.log(step, m)
            print(f"  step {step:>7} loss={m['critic_loss']:.4f} q={m.get('q_mean',0):.4f} "
                  f"{sps:.1f} it/s")
        if logger and step > start_step and step % cfg.eval_interval == 0:
            ev = evaluate(params, [next(batch_iter(99999)) for _ in range(4)])
            logger.log(step, ev, prefix="eval")
            print(f"  [eval {step}] loss={ev['eval_loss']:.4f} q_vs_mc_corr={ev['q_vs_mc_corr']:.3f}")
        if logger and step > start_step and step % cfg.save_interval == 0:
            save_checkpoint(cfg.checkpoint_dir, step, params, target_params, opt_state)
            prune_checkpoints(cfg.checkpoint_dir, cfg.keep_period)

    if timing_steps:
        dt = time.time() - t0
        print(f"\n=== timing: {timing_steps-1} steps in {dt:.1f}s -> {(timing_steps-1)/dt:.2f} it/s")
        print(f"    => 500k steps ~= {500_000/max((timing_steps-1)/dt,1e-9)/3600:.1f} h")
    else:
        save_checkpoint(cfg.checkpoint_dir, start_step + n_steps, params, target_params, opt_state)
        logger.close()
        print("=== done ===")


def main():
    import tyro
    @dataclasses.dataclass
    class Args:
        config: str = "vla_aqc_td_a51"     # registry key (see vla_config.CONFIGS)
        exp_name: str = ""
        seed: int = 0
        timing_steps: int = 0              # >0 => throughput probe, no checkpoints
        resume: bool = False
    args = tyro.cli(Args)
    cfg = get_config(args.config)
    cfg = dataclasses.replace(cfg, seed=args.seed,
                              exp_name=args.exp_name or "")
    train(cfg, timing_steps=args.timing_steps, resume=args.resume)


if __name__ == "__main__":
    main()
