"""
Compact a LeRobot v2.1 dataset so episode_index is contiguous 0..N-1.

If some episodes were dropped (leaving a gap in episode_index / missing files),
this renumbers the remaining episodes to be contiguous: renames parquet + video
files, patches the `episode_index` and global `index` columns inside each moved
parquet, and rebuilds meta (episodes.jsonl, episodes_stats.jsonl, info.json
totals, sources.jsonl ranges).  No video re-encoding.

Safe to run on an already-contiguous dataset (no-op).

Usage:
    python scripts/compact_lerobot_episodes.py --root /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    meta = root / "meta"

    info = json.loads((meta / "info.json").read_text())
    chunks_size = info["chunks_size"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    vkeys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    eplines = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
    eplines.sort(key=lambda e: e["episode_index"])
    old_idx = [e["episode_index"] for e in eplines]
    N = len(eplines)
    old2new = {old: new for new, old in enumerate(old_idx)}
    gap = sorted(set(range(old_idx[0], old_idx[-1] + 1)) - set(old_idx))
    print(f"episodes={N} | old range [{old_idx[0]},{old_idx[-1]}] | missing(old)={gap}")
    if all(o == n for o, n in old2new.items()):
        print("Already contiguous — nothing to do.")
        return

    stats = {}
    sp = meta / "episodes_stats.jsonl"
    if sp.exists():
        for l in sp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                stats[d["episode_index"]] = d["stats"]

    def dp(idx):
        return root / data_tmpl.format(episode_chunk=idx // chunks_size, episode_index=idx)

    def vp(idx, vk):
        return root / video_tmpl.format(episode_chunk=idx // chunks_size, video_key=vk, episode_index=idx)

    new_eplines, new_stats, gidx = [], [], 0
    # ascending new order; downward shifts are safe (target slot already vacated)
    for new, old in enumerate(old_idx):
        e = next(x for x in eplines if x["episode_index"] == old)
        T = e["length"]
        if old != new:
            # move + patch parquet
            df = pd.read_parquet(dp(old))
            df["episode_index"] = np.int64(new)
            df["index"] = np.arange(gidx, gidx + T, dtype=np.int64)
            tmp = dp(new).with_name(dp(new).name + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(tmp, index=False)
            tmp.replace(dp(new))
            dp(old).unlink()
            for vk in vkeys:
                src, dst = vp(old, vk), vp(new, vk)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
        else:
            # sanity: ensure index column already aligns (no-op otherwise)
            pass

        ne = dict(e); ne["episode_index"] = new
        new_eplines.append(ne)
        if old in stats:
            st = json.loads(json.dumps(stats[old]))
            if "episode_index" in st:
                st["episode_index"] = {"min": [new], "max": [new], "mean": [float(new)],
                                       "std": [0.0], "count": [T]}
            if "index" in st:
                idxs = np.arange(gidx, gidx + T, dtype=np.float64)
                st["index"] = {"min": [int(gidx)], "max": [int(gidx + T - 1)],
                               "mean": [float(idxs.mean())], "std": [float(idxs.std())], "count": [T]}
            new_stats.append({"episode_index": new, "stats": st})
        gidx += T

    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in new_eplines) + "\n")
    if new_stats:
        (meta / "episodes_stats.jsonl").write_text("\n".join(json.dumps(s) for s in new_stats) + "\n")

    # sources.jsonl: remap ranges through old2new (clamp to kept episodes)
    srcp = meta / "sources.jsonl"
    if srcp.exists():
        out = []
        for l in srcp.read_text().splitlines():
            if not l.strip():
                continue
            s = json.loads(l)
            lo, hi = s.get("episode_index_start"), s.get("episode_index_end")
            kept = [old2new[o] for o in old_idx if lo <= o <= hi]
            if kept:
                s["episode_index_start"] = min(kept)
                s["episode_index_end"] = max(kept)
                s["num_episodes"] = len(kept)
            out.append(s)
        srcp.write_text("\n".join(json.dumps(s) for s in out) + "\n")

    info["total_episodes"] = N
    info["total_frames"] = int(gidx)
    info["total_chunks"] = (N - 1) // chunks_size + 1
    if "total_videos" in info:
        info["total_videos"] = N * len(vkeys)
    info["splits"] = {"train": f"0:{N}"}
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    print(f"Done: {N} contiguous episodes, {gidx} frames.")


if __name__ == "__main__":
    main()
