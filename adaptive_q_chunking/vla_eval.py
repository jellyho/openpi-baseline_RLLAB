"""Trajectory value-curve evaluation for the VLA AQC critic (no env rollout).

Picks a few success and failure episodes, queries the trained critic along each
*recorded* trajectory — Q(s_t, executed_chunk a_{t:t+H}) at every timestep — and plots
it against the ground-truth mc_return(s_t). This is a visual calibration check: does the
critic's value track the true discounted return-to-go, and does it drop toward -0.5 on
failure trajectories while staying near 0/-0.4 on successes?

The figure is logged to W&B under ``eval/value_curves`` (the training loop calls
``compute_curves`` + ``plot_curves`` every eval_interval). It can also be run standalone
on a checkpoint:

    python vla_eval.py --config vla_aqc_td_macro --step 200000 --out eval_curves.png

Note: the critic full-prefix value at the last valid start (t = n-H) already covers the
chunk reaching the episode terminal, so the failure drop IS captured even though the curve
stops H frames before the episode end.
"""

import argparse
import dataclasses
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")               # headless; we never show(), only log/save
import matplotlib.pyplot as plt

from vla_data import VLALeRobotDataset, LATENT_DIM, ACTION_DIM, _list_col_to_numpy


# --------------------------------------------------------------------------- episode access
def _leaf_index(pf, name: str) -> int:
    """Flat parquet column index of a scalar leaf column (for row-group statistics)."""
    rg = pf.metadata.row_group(0)
    for i in range(rg.num_columns):
        if rg.column(i).path_in_schema == name:
            return i
    raise KeyError(name)


def scan_episodes(ds: VLALeRobotDataset):
    """FAST episode index: episode_index -> [(file, group), ...] and is_failure map.

    Uses parquet row-group STATISTICS (min/max) so the episode->group mapping is built from
    metadata only (no data decode). Episodes are contiguous & sorted, so a group spanning
    [emin, emax] holds frames of exactly episodes emin..emax. Only the few row-groups whose
    reward-min hits the -0.5 failure terminal are actually read (to resolve which episode
    failed). Falls back to reading a group if its statistics are missing.
    """
    groups: dict[int, list] = {}
    is_fail: dict[int, bool] = {}
    for f in ds.files:
        pf = ds._readers[f]
        ei, ri = _leaf_index(pf, "episode_index"), _leaf_index(pf, "reward")
        for g in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(g)
            es, rs = rg.column(ei).statistics, rg.column(ri).statistics
            if es is None or not es.has_min_max:          # no stats -> read scalars for this group
                t = pf.read_row_group(g, columns=["episode_index", "reward"])
                ep = np.asarray(t["episode_index"].to_pylist())
                rew = np.asarray(t["reward"].to_pylist(), dtype=np.float32)
                for e in np.unique(ep):
                    groups.setdefault(int(e), []).append((f, g))
                for e in np.unique(ep[rew <= -0.05]):
                    is_fail[int(e)] = True
                continue
            for e in range(int(es.min), int(es.max) + 1):  # contiguous episodes in this group
                groups.setdefault(e, []).append((f, g))
            if rs is not None and rs.has_min_max and rs.min <= -0.05:  # group holds a fail terminal
                t = pf.read_row_group(g, columns=["episode_index", "reward"])
                ep = np.asarray(t["episode_index"].to_pylist())
                rew = np.asarray(t["reward"].to_pylist(), dtype=np.float32)
                for e in np.unique(ep[rew <= -0.05]):
                    is_fail[int(e)] = True
    return groups, is_fail


def load_episode(ds: VLALeRobotDataset, ep: int, groups: dict) -> dict:
    """Gather all frames of one episode (across row-groups), sorted by frame_index."""
    cols = ["episode_index", "frame_index", "action", "rl_token", "mc_return"]
    fis, rls, acts, mcs = [], [], [], []
    for (f, g) in groups[ep]:
        t = ds._readers[f].read_row_group(g, columns=cols)
        epv = np.asarray(t["episode_index"].to_pylist())
        m = epv == ep
        if not m.any():
            continue
        fis.append(np.asarray(t["frame_index"].to_pylist())[m])
        acts.append(_list_col_to_numpy(t["action"], (ACTION_DIM,))[m])
        rls.append(_list_col_to_numpy(t["rl_token"], (LATENT_DIM,))[m])
        mcs.append(np.asarray(t["mc_return"].to_pylist(), dtype=np.float32)[m])
    fi = np.concatenate(fis)
    order = np.argsort(fi)
    return {
        "frame_index": fi[order],
        "rl_token": np.concatenate(rls)[order],
        "action": np.concatenate(acts)[order],
        "mc_return": np.concatenate(mcs)[order],
    }


def build_eval_set(ds: VLALeRobotDataset, n_success: int = 3, n_fail: int = 3,
                   seed: int = 0) -> list[dict]:
    """Pick + load a fixed set of success/failure episodes (do this ONCE, reuse every eval)."""
    groups, is_fail = scan_episodes(ds)
    all_eps = sorted(groups)
    fail_eps = [e for e in all_eps if is_fail.get(e)]
    succ_eps = [e for e in all_eps if not is_fail.get(e)]
    rng = np.random.default_rng(seed)
    sel_fail = sorted(rng.choice(fail_eps, size=min(n_fail, len(fail_eps)), replace=False).tolist()) if fail_eps else []
    sel_succ = sorted(rng.choice(succ_eps, size=min(n_success, len(succ_eps)), replace=False).tolist()) if succ_eps else []
    out = []
    for e in sel_succ:
        d = load_episode(ds, e, groups); d["ep"] = e; d["fail"] = False; out.append(d)
    for e in sel_fail:
        d = load_episode(ds, e, groups); d["ep"] = e; d["fail"] = True; out.append(d)
    return out


# --------------------------------------------------------------------------- critic query
def _episode_curve(trainer, params, ep: dict, horizon: int, action_dim: int,
                   batch: int = 512) -> dict:
    """Query Q(s_t, executed chunk) at every valid start t of one episode.

    Returns {x (timesteps), mc (mc_return at t), q (ensemble-mean full-prefix Q at t)}.
    """
    rl, act = ep["rl_token"], ep["action"]
    n = len(rl); m = n - horizon
    if m <= 0:
        return {"x": np.array([]), "mc": np.array([]), "q": np.array([]),
                "ep": ep["ep"], "fail": ep["fail"]}
    starts = np.arange(m)
    qs = []
    for s in range(0, m, batch):
        idx = starts[s:s + batch]
        obs = jnp.asarray(rl[idx])
        chunks = jnp.asarray(np.stack([act[i:i + horizon] for i in idx])
                             .reshape(len(idx), horizon * action_dim))
        logits = trainer.net.apply(params, obs, chunks)        # (K, b, macro_H, atoms)
        q = trainer.from_probs(jax.nn.softmax(logits, -1))     # (K, b, macro_H)
        qs.append(np.asarray(q[..., -1].mean(0)))              # ensemble-mean full-prefix value
    return {"x": ep["frame_index"][starts], "mc": ep["mc_return"][starts],
            "q": np.concatenate(qs), "ep": ep["ep"], "fail": ep["fail"]}


def compute_curves(trainer, params, eval_set: list[dict], horizon: int,
                   action_dim: int) -> list[dict]:
    """Per-episode {x, mc, q, ep, fail}. Separated from plotting so plot is testable."""
    return [_episode_curve(trainer, params, ep, horizon, action_dim) for ep in eval_set]


# --------------------------------------------------------------------------- plotting
def plot_curves(curves: list[dict], v_min: float = -1.0, v_max: float = 0.0):
    """2-row grid (success / failure), one subplot per episode: mc_return vs critic Q."""
    succ = [c for c in curves if not c["fail"]]
    fail = [c for c in curves if c["fail"]]
    ncol = max(len(succ), len(fail), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 6.2), squeeze=False)
    for row, (grp, label, color) in enumerate(
            [(succ, "success", "#127a3d"), (fail, "failure", "#b42318")]):
        for c in range(ncol):
            ax = axes[row][c]
            if c >= len(grp):
                ax.axis("off"); continue
            cur = grp[c]
            if len(cur["x"]) == 0:
                ax.set_title(f"{label} ep{cur['ep']} (too short)"); continue
            ax.plot(cur["x"], cur["mc"], color="#1c2330", lw=1.8, label="mc_return")
            ax.plot(cur["x"], cur["q"], color=color, lw=1.3, ls="--", label="critic Q")
            ax.set_title(f"{label} · ep {cur['ep']}", fontsize=10)
            ax.set_xlabel("timestep"); ax.set_ylim(v_min - 0.05, v_max + 0.05)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel("value")
            if row == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- standalone
def main():
    parser = argparse.ArgumentParser(description="VLA critic trajectory value-curve eval")
    parser.add_argument("--config", default="vla_aqc_td_macro")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    parser.add_argument("--n_success", type=int, default=3)
    parser.add_argument("--n_fail", type=int, default=3)
    parser.add_argument("--out", default="eval_curves.png", help="PNG path (standalone)")
    parser.add_argument("--wandb", action="store_true", help="log to W&B eval/ instead of file")
    args = parser.parse_args()

    from vla_config import get_config
    from vla_aqc import VLACriticTrainer
    from vla_train import load_checkpoint, list_checkpoints

    cfg = get_config(args.config)
    if args.exp_name:
        cfg = dataclasses.replace(cfg, exp_name=args.exp_name)
    trainer = VLACriticTrainer(cfg, seed=cfg.seed)
    params = trainer.params
    steps = list_checkpoints(cfg.checkpoint_dir)
    if steps:
        step = args.step if args.step >= 0 else steps[-1]
        params, _ = load_checkpoint(cfg.checkpoint_dir, step, trainer.params, trainer.opt_state)
        print(f"loaded checkpoint step {step}")
    else:
        step = 0
        print("no checkpoint found — using randomly-initialised critic (plumbing check)")

    ds = VLALeRobotDataset(cfg.data_root, horizon=cfg.horizon,
                           commander_filter=set(cfg.commander_filter) if cfg.commander_filter else None,
                           include_base_action=False, mc_gamma=cfg.td.mc_gamma,
                           discount=cfg.td.discount, relabel_living=cfg.reward.relabel_living,
                           relabel_fail=cfg.reward.relabel_fail)
    eval_set = build_eval_set(ds, args.n_success, args.n_fail)
    curves = compute_curves(trainer, params, eval_set, cfg.horizon, cfg.action_dim)
    fig = plot_curves(curves, cfg.dist.v_min, cfg.dist.v_max)

    if args.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                         name=cfg.exp, group=cfg.name, resume="allow")
        run.log({"eval/value_curves": wandb.Image(fig)}, step=step)
        run.finish()
        print(f"logged to W&B eval/value_curves (step {step})")
    else:
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
