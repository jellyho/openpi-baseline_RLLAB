"""Re-annotate the `reward` AND `mc_return` columns in place (RECAP/LPS convention).

Recomputed from first principles per task so both columns are guaranteed consistent:
    norm_len  = max episode length over the task (post restore-trim)
    C_fail    = cfail_frac * norm_len            (default 0.5 * norm_len)
    reward_t  = -1/norm_len   per step;  terminal = 0 (success) / -C_fail/norm_len = -0.5 (failure)
    mc_return_t = clip( Σ_{k>=t} γ^{k-t} · reward_k , -1, 0 )     (value normalized to [-1, 0])

success is read from meta/sources.jsonl (expert + success-and-hil = success;
failure = failure).  Only the parquet columns + the per-episode reward/mc_return
entries in meta/episodes_stats.jsonl are rewritten — videos are untouched (fast).

Use γ=1.0 (undiscounted) for the 60fps Phase-1 datasets: the terminal reward then
propagates across the whole episode (a clean normalized time-to-outcome signal),
instead of being washed out by γ=0.995 (effective horizon ~3.3s ≪ 40s episodes).

Usage
─────
    python scripts/reannotate_mc_return.py \
        --root /data5/jellyho/PFR_RSS/dataset/phase1_merged \
        --tasks insert-mouse-battery seal-water-bottle-cap tower-of-hanoi-game \
        --gamma 1.0
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd


def mc_returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    T = len(reward)
    g = np.zeros(T, dtype=np.float64)
    g[-1] = reward[-1]
    for t in range(T - 2, -1, -1):
        g[t] = reward[t] + gamma * g[t + 1]
    return np.clip(g, -1.0, 0.0).astype(np.float32)


def _num_stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64).reshape(-1, 1)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def reannotate_task(task_dir: pathlib.Path, gamma: float, cfail_frac: float = 0.5,
                    norm_pct: float = 99.0):
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    chunks = info["chunks_size"]
    data_tmpl = info["data_path"]
    eps = [json.loads(l) for l in (task_dir / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    lengths = {e["episode_index"]: int(e["length"]) for e in eps}

    # success per episode from sources.jsonl (expert + success-and-hil = success)
    success_of: dict[int, bool] = {}
    for s in [json.loads(l) for l in (task_dir / "meta" / "sources.jsonl").read_text().splitlines() if l.strip()]:
        for ei in range(s["episode_index_start"], s["episode_index_end"] + 1):
            success_of[ei] = bool(s["success"])

    # norm_len = p99 of SUCCESS episode lengths.  Using success-only (not max over
    # all) keeps the timeout-capped episodes from inflating the scale; p99 trims the
    # single longest success that sits at the hard timeout cap.
    succ_lengths = [lengths[ei] for ei in lengths if success_of.get(ei, True)]
    norm_len = int(round(float(np.percentile(succ_lengths, norm_pct))))
    cfail = cfail_frac * norm_len

    new_r_stats, new_mc_stats = {}, {}
    rmin = mcmin = 1e9
    for e in eps:
        ei = e["episode_index"]
        p = task_dir / data_tmpl.format(episode_chunk=ei // chunks, episode_index=ei)
        df = pd.read_parquet(p)
        T = len(df)
        reward = np.full(T, -1.0 / norm_len, dtype=np.float32)
        reward[-1] = 0.0 if success_of.get(ei, True) else np.float32(-cfail / norm_len)
        mc = mc_returns(reward.astype(np.float64), gamma)
        df["reward"] = reward
        df["mc_return"] = mc
        df.to_parquet(p, index=False)
        new_r_stats[ei] = _num_stats(reward)
        new_mc_stats[ei] = _num_stats(mc)
        rmin = min(rmin, float(reward.min()))
        mcmin = min(mcmin, float(mc.min()))

    # patch episodes_stats.jsonl: reward + mc_return entries
    stats_path = task_dir / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        lines = [json.loads(l) for l in stats_path.read_text().splitlines() if l.strip()]
        for row in lines:
            ei = row["episode_index"]
            st = row.get("stats", {})
            if ei in new_r_stats and "reward" in st:
                st["reward"] = new_r_stats[ei]
            if ei in new_mc_stats and "mc_return" in st:
                st["mc_return"] = new_mc_stats[ei]
        stats_path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    print(f"[reannotate] {task_dir.name}: {len(eps)} eps | norm_len={norm_len} cfail={cfail:.0f} gamma={gamma} "
          f"-> reward min={rmin:.4f}  mc_return min={mcmin:.4f}  (both in [-1,0])")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--cfail-frac", type=float, default=0.5)
    ap.add_argument("--norm-pct", type=float, default=99.0,
                    help="percentile of SUCCESS episode lengths used as norm_len")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    for t in args.tasks:
        reannotate_task(root / t, args.gamma, args.cfail_frac, args.norm_pct)
    print("[reannotate] done.")


if __name__ == "__main__":
    main()
