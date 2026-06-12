"""
Dump a few annotated-dataset episodes to compact ``.npz`` files for FAST, repeatable
critic visualization/testing — so you don't reload the full (multi-million-frame)
LeRobotDataset (slow instantiation + per-frame video decode) every time.

Pays the video-decode cost ONCE here; ``visualize_aqc_critic.py --episode-npz`` then
loads instantly.

Episode categories (auto-detected from per-episode meta + commander_state):
  • success      : reward/min > -0.1  (no failure penalty), policy ('inference') rollout
  • failure      : reward/min ≈ -0.5  (failure penalty baked in by compute_mc_returns)
  • intervention : commander_state has BOTH 'inference' (policy) AND 'teleop' (human took over)

Each ``.npz`` holds (T = episode length):
  rl_token[T,L] f32 · base_action[T,N,H,Dr] f16 · gt_action[T,H,Dr] f32 ·
  mc_return[T] f32 · reward[T] f32 · commander[T] str · cam[T,h,w,3] uint8 ·
  + scalars: episode_index, category, fps, action_horizon.

Usage:
    uv run scripts/dump_episodes.py \\
        --dataset jellyho/insert-mouse-battery_annotated \\
        --local-root /data5/gwanwoo/rss_pft/phase1/insert-mouse-battery_annotated \\
        --auto --out data/critic_vis/episodes
    # or explicit:  --episodes 977:success 947:intervention 1013:failure
"""

import argparse
import glob
import pathlib

import numpy as np

from openpi.shared.lerobot_compat import LeRobotDataset


def _to_np(v):
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def _cs_str(v):
    v = v[0] if (hasattr(v, "__len__") and not isinstance(v, str)) else v
    return str(v)


def _auto_select(local_root):
    """Return [(ep, category)] — one shortest representative per category."""
    import pyarrow.parquet as pq
    import pandas as pd

    root = pathlib.Path(local_root)
    ep = pq.read_table(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))[0]).to_pandas()
    rmin = ep["stats/reward/min"].apply(lambda v: float(np.asarray(v).reshape(-1)[0]))
    d = pd.DataFrame({"ep": ep["episode_index"].astype(int), "len": ep["length"].astype(int), "rmin": rmin})

    # commander_state set per episode (from data parquets; no video decode).
    ep_cs = {}
    for dp in sorted(glob.glob(f"{root}/data/*/*.parquet")):
        t = pq.read_table(dp, columns=["episode_index", "observation.commander_state"]).to_pandas()
        cs = t["observation.commander_state"].apply(_cs_str)
        for e, g in t.assign(cs=cs).groupby("episode_index"):
            ep_cs[int(e)] = set(g["cs"].unique())
    d["mixed"] = d["ep"].map(lambda e: {"teleop", "inference"}.issubset(ep_cs.get(e, set())))

    out = []
    succ = d[(d.rmin > -0.1) & (~d.mixed)].nsmallest(1, "len")
    fail = d[d.rmin < -0.1].nsmallest(1, "len")
    interv = d[d.mixed].nsmallest(1, "len")
    for cat, sub in [("success", succ), ("intervention", interv), ("failure", fail)]:
        if len(sub):
            out.append((int(sub.iloc[0]["ep"]), cat))
        else:
            print(f"  [warn] no episode found for category '{cat}'")
    return out


def _episode_range(ds, episode):
    if hasattr(ds, "episode_data_index"):
        return int(ds.episode_data_index["from"][episode].item()), int(ds.episode_data_index["to"][episode].item())
    eps = ds.meta.episodes
    row = eps.iloc[episode] if hasattr(eps, "iloc") else eps[episode]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def _resize(img, max_side):
    if max_side <= 0 or max(img.shape[:2]) <= max_side:
        return img
    try:
        from PIL import Image
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        return np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR))
    except Exception:
        return img  # PIL missing → store native


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="annotated LeRobot repo id (rl_token + base_action).")
    p.add_argument("--local-root", default=None, help="local dataset root (skip HF download).")
    p.add_argument("--out", required=True, help="output dir for the .npz files.")
    p.add_argument("--auto", action="store_true", help="auto-pick 1 success / 1 intervention / 1 failure.")
    p.add_argument("--episodes", nargs="*", default=[], help="explicit 'IDX[:category]' entries.")
    p.add_argument("--cam-key", default="observation.images.cam_high")
    p.add_argument("--max-side", type=int, default=256, help="resize camera longest side (0 = native).")
    p.add_argument("--action-horizon", type=int, default=50)
    args = p.parse_args()

    if args.auto:
        if not args.local_root:
            raise ValueError("--auto needs --local-root (reads meta + data parquets directly).")
        targets = _auto_select(args.local_root)
    else:
        targets = []
        for e in args.episodes:
            idx, _, cat = e.partition(":")
            targets.append((int(idx), cat or "ep"))
    if not targets:
        raise ValueError("nothing to dump — pass --auto or --episodes.")
    print("dumping:", targets)

    H = args.action_horizon
    fps = LeRobotDataset(args.dataset, root=args.local_root).meta.fps
    ds = LeRobotDataset(args.dataset, root=args.local_root,
                        delta_timestamps={"action": [t / fps for t in range(H)]})
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep, cat in targets:
        a, b = _episode_range(ds, ep)
        T = b - a
        print(f"  episode {ep} ({cat}): {T} frames ...")
        rl, ba, gt, mc, rw, cs, cam = [], [], [], [], [], [], []
        for i in range(T):
            s = ds[a + i]
            rl.append(_to_np(s["rl_token"]).astype(np.float32).reshape(-1))
            ba.append(_to_np(s["base_action"]).astype(np.float16))
            gt.append(_to_np(s["action"]).astype(np.float32).reshape(H, -1))
            mc.append(float(np.asarray(_to_np(s["mc_return"])).reshape(-1)[0]))
            rw.append(float(np.asarray(_to_np(s["reward"])).reshape(-1)[0]) if "reward" in s else np.nan)
            cs.append(_cs_str(s.get("observation.commander_state", "")))
            img = _to_np(s[args.cam_key])
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            cam.append(_resize(img, args.max_side))
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{T}")
        path = out_dir / f"ep{ep:05d}_{cat}.npz"
        np.savez_compressed(
            path,
            rl_token=np.stack(rl), base_action=np.stack(ba), gt_action=np.stack(gt),
            mc_return=np.array(mc, np.float32), reward=np.array(rw, np.float32),
            commander=np.array(cs), cam=np.stack(cam),
            episode_index=ep, category=cat, fps=fps, action_horizon=H,
        )
        print(f"    saved {path}  ({path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
