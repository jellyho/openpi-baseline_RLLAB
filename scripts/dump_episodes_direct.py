"""Dump annotated-dataset episodes to compact .npz for critic visualization — WITHOUT
LeRobotDataset.

``LeRobotDataset`` (v3) loads the data via HuggingFace ``datasets.from_parquet``, which
materializes the WHOLE dataset (3M frames incl. the huge base_action column) into a
memory-mapped Arrow cache (~65 GB) before a single frame is read. This script avoids that
entirely: it reads the chosen episode's rows straight from the data parquet and decodes only
that episode's ``[from_ts, to_ts]`` span from the cam mp4 (pyav). No HF cache, no full scan.

Categories (4): success / intervention+success / intervention+failure / failure, where
  * failure      = stats/reward/min < ``--fail-thresh`` (failure penalty baked in)
  * intervention = the episode's commander_state set contains BOTH 'inference' and 'teleop'.

npz layout (matches visualize_aqc_critic.load_from_npz):
  rl_token[T,L] f32 · base_action[T,N,H,Dr] f16 · gt_action[T,H,Dr] f32 ·
  mc_return[T] f32 · reward[T] f32 · commander[T] str · cam[T,h,w,3] uint8 ·
  + scalars episode_index, category, fps, action_horizon.

Usage:
  uv run scripts/dump_episodes_direct.py \
      --root /lustre/gwanwoo13/.../seal-water-bottle-cap_annotated \
      --out data/critic_vis/episodes --per-category 1
"""

import argparse
import glob
import json
import pathlib

import av
import numpy as np
import pyarrow.parquet as pq

ACTION_DIM = 14
BASE_ACTION_SHAPE = (32, 50, 14)
LATENT_DIM = 2048


def _leaf_numpy(col, trailing, dtype):
    """nested list<...> arrow column -> dense (n, *trailing) ndarray (leaf-buffer fast path)."""
    v = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    n = len(v)
    while hasattr(v, "values"):
        v = v.values
    return np.asarray(v.to_numpy(zero_copy_only=False)).astype(dtype, copy=False).reshape((n,) + trailing)


def _cs_str(v):
    v = v[0] if (hasattr(v, "__len__") and not isinstance(v, str)) else v
    return str(v)


def load_episode_meta(root: pathlib.Path):
    """Per-episode meta as a list of dicts (one row of meta/episodes), + commander sets."""
    ep_files = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index",
            "data/chunk_index", "data/file_index",
            "videos/observation.images.cam_high/chunk_index",
            "videos/observation.images.cam_high/file_index",
            "videos/observation.images.cam_high/from_timestamp",
            "videos/observation.images.cam_high/to_timestamp",
            "stats/reward/min"]
    rows = pq.read_table(ep_files, columns=cols).to_pylist()
    meta = {}
    for r in rows:
        meta[int(r["episode_index"])] = {
            "ep": int(r["episode_index"]), "length": int(r["length"]),
            "from": int(r["dataset_from_index"]), "to": int(r["dataset_to_index"]),
            "data_chunk": int(r["data/chunk_index"]), "data_file": int(r["data/file_index"]),
            "vid_chunk": int(r["videos/observation.images.cam_high/chunk_index"]),
            "vid_file": int(r["videos/observation.images.cam_high/file_index"]),
            "from_ts": float(r["videos/observation.images.cam_high/from_timestamp"]),
            "to_ts": float(r["videos/observation.images.cam_high/to_timestamp"]),
            "rmin": float(np.asarray(r["stats/reward/min"]).reshape(-1)[0]),
        }
    # commander set per episode (read only the two tiny columns from each data parquet).
    for dp in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(dp, columns=["episode_index", "observation.commander_state"])
        eps = np.asarray(t["episode_index"].to_pylist())
        cs = [_cs_str(x) for x in t["observation.commander_state"].to_pylist()]
        for e in np.unique(eps):
            s = {cs[i] for i in np.nonzero(eps == e)[0]}
            if int(e) in meta:
                meta[int(e)].setdefault("cs", set()).update(s)
    return meta


def categorize(m, fail_thresh):
    fail = m["rmin"] < fail_thresh
    interv = {"teleop", "inference"}.issubset(m.get("cs", set()))
    if interv and not fail:
        return "intervention+success"
    if interv and fail:
        return "intervention+failure"
    return "failure" if fail else "success"


def select(meta, per_category, fail_thresh):
    by_cat = {}
    for m in meta.values():
        by_cat.setdefault(categorize(m, fail_thresh), []).append(m)
    picks = []
    for cat in ("success", "intervention+success", "intervention+failure", "failure"):
        cand = sorted(by_cat.get(cat, []), key=lambda m: m["length"])  # shortest first (fast decode)
        if not cand:
            print(f"  [warn] no episode for category '{cat}'")
        for m in cand[:per_category]:
            picks.append((m, cat))
    return picks


def read_episode_data(root, m, H):
    """rl_token, base_action(f16), gt_action(H-window, hold-last pad), mc, reward, commander.

    Reads ONLY the row-groups that the episode spans (each group is 1000 rows): reading the
    whole file's base_action column and ``combine_chunks`` overflows the int32 list offsets.
    """
    dp = root / f"data/chunk-{m['data_chunk']:03d}/file-{m['data_file']:03d}.parquet"
    pf = pq.ParquetFile(str(dp))
    epcol = np.asarray(pf.read(columns=["episode_index"])["episode_index"].to_pylist())
    loc = np.nonzero(epcol == m["ep"])[0]
    rg0 = pf.metadata.row_group(0).num_rows                      # uniform group size (1000)
    rgs = list(range(int(loc.min()) // rg0, int(loc.max()) // rg0 + 1))
    t = pf.read_row_groups(rgs, columns=["episode_index", "frame_index", "rl_token",
                                         "base_action", "action", "mc_return", "reward",
                                         "observation.commander_state"])
    ep = np.asarray(t["episode_index"].to_pylist())
    sel = np.nonzero(ep == m["ep"])[0]
    order = np.argsort(np.asarray(t["frame_index"].to_pylist())[sel])
    sel = sel[order]
    rl = _leaf_numpy(t["rl_token"], (LATENT_DIM,), np.float32)[sel]
    ba = _leaf_numpy(t["base_action"], BASE_ACTION_SHAPE, np.float16)[sel]
    act = _leaf_numpy(t["action"], (ACTION_DIM,), np.float32)[sel]
    mc = np.asarray(t["mc_return"].to_pylist(), np.float32)[sel]
    rew = np.asarray(t["reward"].to_pylist(), np.float32)[sel]
    cs = np.array([_cs_str(x) for x in t["observation.commander_state"].to_pylist()])[sel]
    T = len(sel)
    # gt_action: per-frame H-step executed chunk, hold-last padded past the episode end
    # (matches the training loader). gt[i] = act[i:i+H] clamped to T-1.
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None, :], T - 1)   # (T,H)
    gt = act[idx]                                                            # (T,H,Dr)
    return {"rl": rl, "base": ba, "gt": gt, "mc": mc, "reward": rew, "cs": cs, "T": T}


def decode_cam(root, m, n_frames, max_side):
    """Decode exactly ``n_frames`` cam_high frames from [from_ts, to_ts] of the episode's mp4."""
    vp = root / f"videos/observation.images.cam_high/chunk-{m['vid_chunk']:03d}/file-{m['vid_file']:03d}.mp4"
    container = av.open(str(vp))
    stream = container.streams.video[0]
    from_ts, frames = m["from_ts"], []
    container.seek(int(max(from_ts, 0) / stream.time_base), stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if float(frame.pts * stream.time_base) < from_ts - 1e-3:
            continue
        img = frame.to_ndarray(format="rgb24")
        if 0 < max_side < max(img.shape[:2]):
            from PIL import Image
            h, w = img.shape[:2]; s = max_side / max(h, w)
            img = np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BILINEAR))
        frames.append(img)
        if len(frames) >= n_frames:
            break
    container.close()
    if len(frames) < n_frames:                       # pad with last frame if decode came up short
        frames += [frames[-1]] * (n_frames - len(frames))
    return np.stack(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="annotated dataset root (data/ + meta/ + videos/).")
    p.add_argument("--out", required=True)
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--max-side", type=int, default=256)
    p.add_argument("--fail-thresh", type=float, default=-0.1)
    p.add_argument("--episodes", nargs="*", type=int, default=[], help="explicit episode ids (skip auto-select).")
    args = p.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    H = args.action_horizon
    fps = json.load(open(root / "meta/info.json"))["fps"]

    print("Loading episode meta...")
    meta = load_episode_meta(root)
    if args.episodes:
        picks = [(meta[e], categorize(meta[e], args.fail_thresh)) for e in args.episodes]
    else:
        picks = select(meta, args.per_category, args.fail_thresh)
    print(f"selected {len(picks)}:", [(m['ep'], c, m['length']) for m, c in picks])

    for m, cat in picks:
        print(f"  ep{m['ep']} [{cat}] T={m['length']} — reading data + decoding {m['length']} cam frames...")
        d = read_episode_data(root, m, H)
        cam = decode_cam(root, m, d["T"], args.max_side)
        fn = out / f"ep{m['ep']:04d}_{cat}.npz"
        np.savez_compressed(fn, rl_token=d["rl"], base_action=d["base"], gt_action=d["gt"],
                            mc_return=d["mc"], reward=d["reward"], commander=d["cs"], cam=cam,
                            episode_index=m["ep"], category=cat, fps=fps, action_horizon=H)
        print(f"    -> {fn}  (cam {cam.shape}, rl {d['rl'].shape})")
    print("done.")


if __name__ == "__main__":
    main()
