"""Trajectory value-curve evaluation for the VLA AQC critic (no env rollout).

Picks a few SUCCESS / INTERVENTION / FAILURE episodes, queries the trained critic along
each *recorded* trajectory — Q(s_t, executed_chunk a_{t:t+H}) at every timestep — and plots
it against the ground-truth mc_return(s_t). Visual calibration check: does the critic's value
track the true discounted return-to-go, drop on failures, and stay high on successes?

Categories (by terminal reward + commander_state):
  success      = not a failure terminal, and NOT a human-intervention episode
  intervention = episode containing BOTH inference & teleop frames (policy ran, human took over);
                 the teleop span is SHADED so you can see where the intervention is.
  failure      = failure terminal (reward<=-0.05), and not an intervention episode

Logged to W&B under ``eval/value_curves`` (the train loop calls compute_curves + plot_curves
every eval_interval). Standalone:  python eval_curves.py --config vla_aqc_mini --step -1 --out x.png
"""

import argparse
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")               # headless; we never show(), only log/save
import matplotlib.pyplot as plt

from openpi.rlt_critic.data import VLALeRobotDataset, LATENT_DIM, ACTION_DIM, _list_col_to_numpy


# --------------------------------------------------------------------------- episode access
def _leaf_index(pf, name: str) -> int:
    """Flat parquet column index of a scalar leaf column (for row-group statistics)."""
    rg = pf.metadata.row_group(0)
    for i in range(rg.num_columns):
        if rg.column(i).path_in_schema == name:
            return i
    raise KeyError(name)


def scan_episodes(ds: VLALeRobotDataset):
    """episode_index -> [(file, group)], plus per-episode is_fail / has_inference / has_teleop.

    Episode->group map uses row-group STATISTICS (metadata only). Failure + commander composition
    are read once per file from the small scalar/string columns (reward, commander_state).
    """
    groups: dict[int, list] = {}
    is_fail: dict[int, bool] = {}
    has_inf: dict[int, bool] = {}
    has_tel: dict[int, bool] = {}
    for f in ds.files:
        pf = ds._readers[f]
        ei = _leaf_index(pf, "episode_index")
        for g in range(pf.metadata.num_row_groups):
            es = pf.metadata.row_group(g).column(ei).statistics
            if es is not None and es.has_min_max:
                for e in range(int(es.min), int(es.max) + 1):
                    groups.setdefault(e, []).append((f, g))
            else:                                          # no stats -> read this group's ep col
                ep = np.asarray(pf.read_row_group(g, columns=["episode_index"])["episode_index"].to_pylist())
                for e in np.unique(ep):
                    groups.setdefault(int(e), []).append((f, g))
        # one whole-file read of the small columns -> classify (vectorised)
        t = pf.read(columns=["episode_index", "reward", "observation.commander_state"])
        ep = np.asarray(t["episode_index"].to_pylist())
        rew = np.asarray(t["reward"].to_pylist(), dtype=np.float32)
        cs = np.asarray([str(c).lower() for c in t["observation.commander_state"].to_pylist()])
        for e in np.unique(ep[rew <= -0.05]):  is_fail[int(e)] = True
        for e in np.unique(ep[cs == "inference"]): has_inf[int(e)] = True
        for e in np.unique(ep[cs == "teleop"]):    has_tel[int(e)] = True
    return groups, is_fail, has_inf, has_tel


def load_episode(ds: VLALeRobotDataset, ep: int, groups: dict) -> dict:
    """Gather all frames of one episode (across row-groups), sorted by frame_index."""
    cols = ["episode_index", "frame_index", "action", "rl_token", "mc_return",
            "observation.commander_state"]
    fis, rls, acts, mcs, cmds = [], [], [], [], []
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
        cmds.append(np.asarray([str(c).lower() for c in t["observation.commander_state"].to_pylist()])[m])
    fi = np.concatenate(fis)
    order = np.argsort(fi)
    return {
        "frame_index": fi[order],
        "rl_token": np.concatenate(rls)[order],
        "action": np.concatenate(acts)[order],
        "mc_return": np.concatenate(mcs)[order],
        "commander": np.concatenate(cmds)[order],
    }


def build_eval_set(ds: VLALeRobotDataset, n_success: int = 3, n_fail: int = 3,
                   n_intervention: int = 3, seed: int = 0) -> list[dict]:
    """Pick + load a fixed set of success / intervention / failure episodes (ONCE, reused)."""
    groups, is_fail, has_inf, has_tel = scan_episodes(ds)
    all_eps = sorted(groups)
    interv = [e for e in all_eps if has_inf.get(e) and has_tel.get(e)]
    iset = set(interv)
    fail_eps = [e for e in all_eps if is_fail.get(e) and e not in iset]
    succ_eps = [e for e in all_eps if not is_fail.get(e) and e not in iset]
    rng = np.random.default_rng(seed)
    def pick(pool, k):
        return sorted(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()) if pool else []
    out = []
    for cat, sel in [("success", pick(succ_eps, n_success)),
                     ("intervention", pick(interv, n_intervention)),
                     ("failure", pick(fail_eps, n_fail))]:
        for e in sel:
            d = load_episode(ds, e, groups)
            d["ep"] = e; d["category"] = cat; d["fail"] = (cat == "failure")
            out.append(d)
    return out


# --------------------------------------------------------------------------- critic query
def _episode_curve(trainer, params, ep: dict, horizon: int, action_dim: int,
                   batch: int = 512) -> dict:
    """Query Q(s_t, executed chunk) at every valid start t. Returns {x, mc, q, teleop, ...}."""
    rl, act = ep["rl_token"], ep["action"]
    n = len(rl); m = n - horizon
    meta = {"ep": ep["ep"], "fail": ep.get("fail", False), "category": ep.get("category", "")}
    if m <= 0:
        return {**meta, "x": np.array([]), "mc": np.array([]), "q": np.array([]), "teleop": None}
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
    teleop = None
    if ep.get("commander") is not None:
        teleop = (np.asarray(ep["commander"]) == "teleop")[starts]
    return {**meta, "x": ep["frame_index"][starts], "mc": ep["mc_return"][starts],
            "q": np.concatenate(qs), "teleop": teleop}


def compute_curves(trainer, params, eval_set: list[dict], horizon: int,
                   action_dim: int) -> list[dict]:
    """Per-episode {x, mc, q, teleop, ep, category, fail}. Separated from plotting (testable)."""
    return [_episode_curve(trainer, params, ep, horizon, action_dim) for ep in eval_set]


# --------------------------------------------------------------------------- plotting
def plot_curves(curves: list[dict], v_min: float = -1.0, v_max: float = 0.0):
    """3-row grid (success / intervention / failure); teleop span shaded on interventions."""
    cats = [("success", "#127a3d"), ("intervention", "#8e44ad"), ("failure", "#b42318")]
    def catof(c):
        return c.get("category") or ("failure" if c.get("fail") else "success")
    by = {name: [c for c in curves if catof(c) == name] for name, _ in cats}
    ncol = max(max(len(v) for v in by.values()), 1)
    fig, axes = plt.subplots(3, ncol, figsize=(4.2 * ncol, 9.0), squeeze=False)
    for row, (name, color) in enumerate(cats):
        grp = by[name]
        for c in range(ncol):
            ax = axes[row][c]
            if c >= len(grp):
                ax.axis("off"); continue
            cur = grp[c]
            if len(cur["x"]) == 0:
                ax.set_title(f"{name} ep{cur['ep']} (too short)"); ax.axis("off"); continue
            x = cur["x"]
            tm = cur.get("teleop")
            if tm is not None and bool(tm.any()):   # shade teleop (intervention) frames
                ax.fill_between(x, v_min - 0.05, v_max + 0.05, where=tm, step="mid",
                                color="#8e44ad", alpha=0.13, lw=0, label="teleop")
            ax.plot(x, cur["mc"], color="#1c2330", lw=1.8, label="mc_return")
            ax.plot(x, cur["q"], color=color, lw=1.3, ls="--", label="critic Q")
            ax.set_title(f"{name} · ep {cur['ep']}", fontsize=10)
            ax.set_xlabel("timestep"); ax.set_ylim(v_min - 0.05, v_max + 0.05)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel("value")
                ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- standalone
def main():
    parser = argparse.ArgumentParser(description="VLA critic trajectory value-curve eval")
    parser.add_argument("--config", default="vla_aqc_mini")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    parser.add_argument("--n_success", type=int, default=3)
    parser.add_argument("--n_intervention", type=int, default=3)
    parser.add_argument("--n_fail", type=int, default=3)
    parser.add_argument("--out", default="eval_curves.png", help="PNG path (standalone)")
    parser.add_argument("--wandb", action="store_true", help="log to W&B eval/ instead of file")
    args = parser.parse_args()

    from openpi.rlt_critic.config import get_config
    from openpi.rlt_critic.agent import VLACriticTrainer
    # load_checkpoint / list_checkpoints live in scripts/train_rlt_critic.py (not a package module).
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
    from train_rlt_critic import load_checkpoint, list_checkpoints

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
    eval_set = build_eval_set(ds, args.n_success, args.n_fail, args.n_intervention)
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
