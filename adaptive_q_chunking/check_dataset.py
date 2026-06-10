"""
Check the reward / mc_return / failure structure of the annotated VLA dataset.
Run with the repo venv:
    cd /home/gwanwoo13/projects/rss_ptf/openpi-baseline_RLLAB
    .venv/bin/python adaptive_q_chunking/check_dataset.py
"""
import glob
import numpy as np
import pyarrow.parquet as pq
from collections import Counter, defaultdict

DATA = "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/insert-mouse-battery_annotated"


def scan_all_episodes():
    """Scan every parquet file and aggregate per-episode stats.

    IMPORTANT: an episode can be split across multiple row-groups AND multiple
    files, so we must accumulate per episode_index across the whole dataset
    before computing last_reward / mc_return. (This is the bug I hit earlier:
    looking at a single row-group only sees a subset of episodes.)
    """
    files = sorted(glob.glob(f"{DATA}/data/chunk-*/file-*.parquet"))
    print(f"parquet files: {len(files)}")

    # episode_index -> list of (frame_index, reward, mc_return) arrays
    eps = defaultdict(lambda: {"fr": [], "rw": [], "mc": [], "cmd": None})
    cols = ["episode_index", "frame_index", "reward", "mc_return",
            "observation.commander_state"]

    for fp in files:
        pf = pq.ParquetFile(fp)
        for gi in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(gi, columns=cols)
            ep = np.asarray(t["episode_index"].to_pylist())
            fr = np.asarray(t["frame_index"].to_pylist())
            rw = np.asarray(t["reward"].to_pylist(), np.float32)
            mc = np.asarray(t["mc_return"].to_pylist(), np.float32)
            cmd = t["observation.commander_state"].to_pylist()
            for e in np.unique(ep):
                m = ep == e
                rec = eps[int(e)]
                rec["fr"].append(fr[m])
                rec["rw"].append(rw[m])
                rec["mc"].append(mc[m])
                if rec["cmd"] is None:
                    rec["cmd"] = cmd[np.where(m)[0][0]]
    return eps


def summarize(eps):
    rows = []  # (eid, cmd, length, last_reward, min_reward, mc0, mc_min)
    for eid, rec in eps.items():
        fr = np.concatenate(rec["fr"])
        rw = np.concatenate(rec["rw"])
        mc = np.concatenate(rec["mc"])
        o = np.argsort(fr)
        rw, mc = rw[o], mc[o]
        rows.append((eid, rec["cmd"], len(fr),
                     float(rw[-1]), float(rw.min()),
                     float(mc[0]), float(mc.min())))
    return rows


def main():
    eps = scan_all_episodes()
    rows = summarize(eps)
    print(f"\ntotal episodes: {len(rows)}")

    by_cmd = defaultdict(list)
    for r in rows:
        by_cmd[r[1]].append(r)

    for cmd, rs in sorted(by_cmd.items()):
        lens = np.array([r[2] for r in rs])
        n_fail = sum(1 for r in rs if r[4] <= -0.4)   # min_reward <= -0.4
        mc0 = np.array([r[5] for r in rs])
        print(f"\n[{cmd}]  n={len(rs)}")
        print(f"  length     : min={lens.min()} max={lens.max()} mean={lens.mean():.0f}")
        print(f"  mc_return[0]: min={mc0.min():.4f} max={mc0.max():.4f} mean={mc0.mean():.4f}")
        print(f"  failure episodes (reward<=-0.4): {n_fail}/{len(rs)}")

    # global reward / mc_return support
    last_rw = np.array([r[3] for r in rows])
    mc_min = np.array([r[6] for r in rows])
    print(f"\n=== value support check ===")
    print(f"  reward values present: {sorted(set(np.round(last_rw,4)))[:8]} ...")
    print(f"  mc_return min across episodes: {mc_min.min():.4f}")
    print(f"  -> suggested support: v_min={mc_min.min()-0.05:.2f}, v_max=0.0")


if __name__ == "__main__":
    main()
