"""
Build a derived LeRobot v2.1 dataset by SELECTING episodes from a source and
recomputing `reward` + `mc_return`.  Re-indexes episodes/frames, copies the
selected parquets + videos, and rebuilds the meta (info.json, episodes.jsonl,
episodes_stats.jsonl, sources.jsonl).

Used to split the merged rl_mc (150 eps = rl 50 + rollouts 100) into:
    rl  = all episodes              (--keep all)
    bc  = success-only episodes     (--keep success)

Reward (same definition / scale as scripts/compute_mc_returns.py):
    r_t        = (-1 per step, terminal 0 if success else -C_fail) / norm_len
    mc_return_t = clip(G_t / norm_len, -1, 0)
`norm_len` defaults to the max episode length over the WHOLE source (NOT the kept
subset), so derived datasets (bc, rl) stay on the SAME normalized scale and the
telescoping identity holds across them.

Usage
─────
    python scripts/build_subset_dataset.py \\
        --src $HF_LEROBOT_HOME/jellyho/aloha_handover_box_joint_pos_rl_mc \\
        --dst $HF_LEROBOT_HOME/jellyho/aloha_handover_box_joint_pos_bc \\
        --keep success --gamma 0.995 --overwrite
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pandas as pd


def episode_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    T = len(rewards)
    g = np.zeros(T, dtype=np.float32)
    g[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        g[t] = rewards[t] + gamma * g[t + 1]
    return g


def episode_success(ep: pd.DataFrame, success_key: str) -> bool:
    v = ep[success_key].to_numpy()
    return bool(np.asarray([x[0] if hasattr(x, "__len__") else x for x in v]).any())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Source dataset root.")
    ap.add_argument("--dst", required=True, help="Destination dataset root.")
    ap.add_argument("--keep", choices=["all", "success", "failure"], default="all")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--cfail", type=float, default=None, help="Failure penalty (default 0.5*norm_len).")
    ap.add_argument("--norm-len", type=int, default=None,
                    help="Normalization denominator (default: max episode length over the WHOLE source).")
    ap.add_argument("--reward-key", default="next.reward")
    ap.add_argument("--success-key", default="next.success")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists. Pass --overwrite.")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = info["chunks_size"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    n_src = info["total_episodes"]

    def src_ep_path(ei):
        return src / data_tmpl.format(episode_chunk=ei // chunks_size, episode_index=ei)

    # ── Pass 1: success per episode, lengths, keep-list, norm_len ──────────────
    lengths, succ = {}, {}
    for ei in range(n_src):
        df = pd.read_parquet(src_ep_path(ei), columns=[args.reward_key, args.success_key])
        lengths[ei] = len(df)
        succ[ei] = episode_success(df, args.success_key)

    keep = [ei for ei in range(n_src)
            if args.keep == "all" or (args.keep == "success") == succ[ei]]
    norm_len = args.norm_len if args.norm_len is not None else max(lengths.values())
    cfail = args.cfail if args.cfail is not None else 0.5 * norm_len
    print(f"src eps={n_src} (succ={sum(succ.values())}) | keep[{args.keep}]={len(keep)} "
          f"| norm_len={norm_len} cfail={cfail} gamma={args.gamma}")

    # ── Load source meta to re-index ──────────────────────────────────────────
    src_eplines = {}
    for line in (src / "meta" / "episodes.jsonl").read_text().splitlines():
        d = json.loads(line)
        src_eplines[d["episode_index"]] = d
    src_stats = {}
    stats_path = src / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            d = json.loads(line)
            src_stats[d["episode_index"]] = d["stats"]

    # ── Build dst skeleton + copy static meta ─────────────────────────────────
    (dst / "meta").mkdir(parents=True)
    for fn in ["tasks.jsonl"]:
        if (src / "meta" / fn).exists():
            shutil.copy(src / "meta" / fn, dst / "meta" / fn)
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", dst / "README.md")

    # ── Pass 2: copy + re-index parquets/videos, recompute reward/mc_return ────
    new_eplines, new_stats, gidx = [], [], 0
    for new_ei, old_ei in enumerate(keep):
        df = pd.read_parquet(src_ep_path(old_ei))
        T = len(df)
        rewards = np.full(T, -1.0, dtype=np.float32)
        rewards[-1] = 0.0 if succ[old_ei] else -cfail
        g = np.clip(episode_mc_returns(rewards, args.gamma) / norm_len, -1.0, 0.0)
        df["reward"] = (rewards / norm_len).astype(np.float32)
        df["mc_return"] = g.astype(np.float32)
        df["episode_index"] = np.int64(new_ei)
        df["index"] = np.arange(gidx, gidx + T, dtype=np.int64)

        outp = dst / data_tmpl.format(episode_chunk=new_ei // chunks_size, episode_index=new_ei)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outp)

        for vk in video_keys:
            sv = src / video_tmpl.format(episode_chunk=old_ei // chunks_size, video_key=vk, episode_index=old_ei)
            dv = dst / video_tmpl.format(episode_chunk=new_ei // chunks_size, video_key=vk, episode_index=new_ei)
            dv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(sv, dv)

        el = dict(src_eplines[old_ei]); el["episode_index"] = new_ei
        new_eplines.append(el)

        if old_ei in src_stats:
            st = json.loads(json.dumps(src_stats[old_ei]))  # deep copy
            if "episode_index" in st:
                st["episode_index"] = {"min": [new_ei], "max": [new_ei],
                                       "mean": [float(new_ei)], "std": [0.0], "count": [T]}
            if "index" in st:
                idxs = np.arange(gidx, gidx + T, dtype=np.float64)
                st["index"] = {"min": [int(gidx)], "max": [int(gidx + T - 1)],
                               "mean": [float(idxs.mean())], "std": [float(idxs.std())], "count": [T]}
            new_stats.append({"episode_index": new_ei, "stats": st})
        gidx += T

    # ── Rebuild meta ──────────────────────────────────────────────────────────
    (dst / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in new_eplines) + "\n")
    if new_stats:
        (dst / "meta" / "episodes_stats.jsonl").write_text("\n".join(json.dumps(s) for s in new_stats) + "\n")
    (dst / "meta" / "sources.jsonl").write_text(
        json.dumps({"source_path": str(src), "keep": args.keep, "num_episodes": len(keep)}) + "\n")

    info["total_episodes"] = len(keep)
    info["total_frames"] = int(gidx)
    info["total_chunks"] = (len(keep) - 1) // chunks_size + 1
    if "total_videos" in info:
        info["total_videos"] = len(keep) * len(video_keys)
    info["splits"] = {"train": f"0:{len(keep)}"}
    info["features"]["reward"] = {"dtype": "float32", "shape": [1], "names": None}
    info["features"]["mc_return"] = {"dtype": "float32", "shape": [1], "names": None}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"Done: {dst}  ({len(keep)} eps, {gidx} frames, {len(keep) * len(video_keys)} videos)")


if __name__ == "__main__":
    main()
