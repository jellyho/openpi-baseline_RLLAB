"""Dump seal_mini episodes (the EXACT data the critic trained on) to .npz for critic viz,
pulling the camera video from the frame-aligned source dataset (which has videos/).

seal_mini has no video (images were stripped for fast training); its episodes were copied
from a source ``*_annotated`` dataset (see seal_mini/categories.json: new_ep <- src_ep). The
source and seal_mini share identical rl_token AND identical per-episode frame counts, so the
critic inputs (rl_token/base_action/action/mc_return) come from seal_mini == what training/eval
saw (so the curves match the W&B eval graphs), while only the cam_high frames are decoded from
the source episode ``src_ep`` (frame-aligned).

4-way category (success/fail x intervention) from seal_mini's own reward + commander_state.
npz layout matches visualize_aqc_critic.load_from_npz.
"""

import argparse
import glob
import json
import pathlib

import av
import numpy as np
import pyarrow.parquet as pq

ACTION_DIM, BASE_ACTION_SHAPE, LATENT_DIM = 14, (32, 50, 14), 2048


def _leaf(col, trailing, dtype):
    v = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    n = len(v)
    while hasattr(v, "values"):
        v = v.values
    return np.asarray(v.to_numpy(zero_copy_only=False)).astype(dtype, copy=False).reshape((n,) + trailing)


def _cs(v):
    v = v[0] if (hasattr(v, "__len__") and not isinstance(v, str)) else v
    return str(v)


def mini_episode(data_root, new_ep, H):
    """Read one seal_mini episode's critic tensors (rl_token, base_action f16, gt H-window, mc)."""
    pf = pq.ParquetFile(sorted(glob.glob(f"{data_root}/data/chunk-*/file-*.parquet"))[0])
    epcol = np.asarray(pf.read(columns=["episode_index"])["episode_index"].to_pylist())
    loc = np.nonzero(epcol == new_ep)[0]
    # row-groups have VARIABLE size (one episode per group, different lengths): map the global
    # row range [lo,hi) to overlapping groups via the cumulative group-size boundaries.
    nrows = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    bounds = np.cumsum([0] + nrows)
    lo, hi = int(loc.min()), int(loc.max()) + 1
    rgs = [i for i in range(len(nrows)) if bounds[i] < hi and bounds[i + 1] > lo]
    t = pf.read_row_groups(rgs, columns=["episode_index", "frame_index", "rl_token", "base_action",
                                         "action", "mc_return", "reward", "observation.commander_state"])
    ep = np.asarray(t["episode_index"].to_pylist())
    sel = np.nonzero(ep == new_ep)[0]
    sel = sel[np.argsort(np.asarray(t["frame_index"].to_pylist())[sel])]
    rl = _leaf(t["rl_token"], (LATENT_DIM,), np.float32)[sel]
    ba = _leaf(t["base_action"], BASE_ACTION_SHAPE, np.float16)[sel]
    act = _leaf(t["action"], (ACTION_DIM,), np.float32)[sel]
    mc = np.asarray(t["mc_return"].to_pylist(), np.float32)[sel]
    rew = np.asarray(t["reward"].to_pylist(), np.float32)[sel]
    cs = np.array([_cs(x) for x in t["observation.commander_state"].to_pylist()])[sel]
    T = len(sel)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None, :], T - 1)
    return {"rl": rl, "base": ba, "gt": act[idx], "mc": mc, "reward": rew, "cs": cs, "T": T}


def video_meta(video_root):
    """src_ep -> (chunk, file, from_ts, to_ts) for cam_high, from the source meta/episodes."""
    f = sorted(glob.glob(f"{video_root}/meta/episodes/*/*.parquet"))
    key = "videos/observation.images.cam_high"
    cols = ["episode_index", f"{key}/chunk_index", f"{key}/file_index",
            f"{key}/from_timestamp", f"{key}/to_timestamp"]
    rows = pq.read_table(f, columns=cols).to_pylist()
    return {int(r["episode_index"]): (int(r[f"{key}/chunk_index"]), int(r[f"{key}/file_index"]),
                                      float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"]))
            for r in rows}


def decode_cam(video_root, vinfo, n_frames, max_side):
    chunk, file, from_ts, _ = vinfo
    vp = f"{video_root}/videos/observation.images.cam_high/chunk-{chunk:03d}/file-{file:03d}.mp4"
    c = av.open(vp); s = c.streams.video[0]
    c.seek(int(max(from_ts, 0) / s.time_base), stream=s, backward=True, any_frame=False)
    frames = []
    for fr in c.decode(s):
        if fr.pts is None or float(fr.pts * s.time_base) < from_ts - 1e-3:
            continue
        img = fr.to_ndarray(format="rgb24")
        if 0 < max_side < max(img.shape[:2]):
            from PIL import Image
            h, w = img.shape[:2]; sc = max_side / max(h, w)
            img = np.asarray(Image.fromarray(img).resize((round(w * sc), round(h * sc)), Image.BILINEAR))
        frames.append(img)
        if len(frames) >= n_frames:
            break
    c.close()
    if len(frames) < n_frames:
        frames += [frames[-1]] * (n_frames - len(frames))
    return np.stack(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/lustre/jellyho/seal_mini", help="seal_mini (trained data, no video).")
    p.add_argument("--video-root", required=True, help="frame-aligned source dataset WITH videos/.")
    p.add_argument("--mapping", default="/lustre/jellyho/seal_mini/categories.json")
    p.add_argument("--out", required=True)
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--max-side", type=int, default=256)
    p.add_argument("--fail-thresh", type=float, default=-0.1)
    p.add_argument("--episodes", nargs="*", type=int, default=[], help="explicit seal_mini new_ep ids.")
    args = p.parse_args()

    H = args.action_horizon
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mapping = {e["new_ep"]: e for e in json.load(open(args.mapping))["episodes"]}
    vmeta = video_meta(args.video_root)

    # classify all seal_mini episodes 4-way from their commander_state + reward (cheap 2-col read).
    dpf = pq.ParquetFile(sorted(glob.glob(f"{args.data_root}/data/chunk-*/file-*.parquet"))[0])
    tt = dpf.read(columns=["episode_index", "observation.commander_state", "reward"])
    ei = np.asarray(tt["episode_index"].to_pylist())
    csv = np.array([_cs(x) for x in tt["observation.commander_state"].to_pylist()])
    rwv = np.asarray(tt["reward"].to_pylist(), np.float32)
    cats = {}
    for e in np.unique(ei):
        m = ei == e
        teleop = "teleop" in set(csv[m]); fail = rwv[m].min() < args.fail_thresh
        cat = ("intervention+failure" if (teleop and fail) else "intervention+success" if teleop else
               "failure" if fail else "success")
        cats.setdefault(cat, []).append((int(e), int(m.sum())))

    if args.episodes:
        picks = [(e, next(c for c, v in cats.items() if any(x[0] == e for x in v))) for e in args.episodes]
    else:
        picks = []
        for cat in ("success", "intervention+success", "intervention+failure", "failure"):
            for e, _ in sorted(cats.get(cat, []), key=lambda x: x[1])[:args.per_category]:
                picks.append((e, cat))
            if not cats.get(cat):
                print(f"  [warn] no seal_mini episode for '{cat}'")
    print("category counts:", {c: len(v) for c, v in cats.items()})
    print("picks (new_ep, cat):", picks)

    for new_ep, cat in picks:
        src = mapping[new_ep]["src_ep"]
        d = mini_episode(args.data_root, new_ep, H)
        print(f"  new_ep{new_ep} (src_ep{src}) [{cat}] T={d['T']} — decoding cam from source...")
        cam = decode_cam(args.video_root, vmeta[src], d["T"], args.max_side)
        fn = out / f"mini{new_ep:02d}_src{src}_{cat}.npz"
        np.savez_compressed(fn, rl_token=d["rl"], base_action=d["base"], gt_action=d["gt"],
                            mc_return=d["mc"], reward=d["reward"], commander=d["cs"], cam=cam,
                            episode_index=new_ep, category=cat, fps=60, action_horizon=H)
        print(f"    -> {fn}  (cam {cam.shape})")
    print("done.")


if __name__ == "__main__":
    main()
