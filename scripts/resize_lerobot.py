"""
Make a rescaled copy of a LeRobot v2.1 dataset: every camera video is resized to
`--size` x `--size` with EXACTLY the training-time transform (openpi
`image_tools.resize_with_pad`, aspect-preserving letterbox) and re-encoded.  All
parquet columns (state / action / reward / mc_return / next.* / indices) are
copied verbatim — only the image resolution changes.

Use this for datasets that already carry their reward/mc_return columns and need
no frame filtering (e.g. the tabletop aloha_handover_box_* datasets), so they
match the 224x224 framing the model actually sees at train time (making the
train-time ResizeImages(224,224) a no-op).

Videos are re-encoded with h264/crf18 by default (visually lossless, avoids the
double-AV1 compression you'd get re-encoding the AV1 source back to AV1).

Parallel + resumable (per-episode sidecars under <dst>/.cache_meta/).

Usage
─────
    JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/resize_lerobot.py \
        --src $HF_LEROBOT_HOME/jellyho/aloha_handover_box_joint_pos_rl_orig \
        --dst $HF_LEROBOT_HOME/jellyho/aloha_handover_box_joint_pos_rl_224 \
        --size 224 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import traceback

import numpy as np


def _lazy():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import av  # noqa
    import pandas as pd  # noqa
    import pyarrow.parquet as pq  # noqa
    from openpi.shared import image_tools  # noqa
    return av, pd, pq, image_tools


def _info(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _present_episodes(root: pathlib.Path, info: dict, cams: list[str]) -> list[int]:
    chunks = info["chunks_size"]
    out = []
    for ep in range(info["total_episodes"]):
        p = root / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
        if not p.exists():
            continue
        if all((root / info["video_path"].format(
                episode_chunk=ep // chunks, video_key=c, episode_index=ep)).exists() for c in cams):
            out.append(ep)
    return out


def _resize_blocks(image_tools, frames: np.ndarray, size: int, block: int = 256) -> np.ndarray:
    """resize_with_pad in fixed-size blocks so the jitted kernel compiles once."""
    T = frames.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for s in range(0, T, block):
        e = min(s + block, T)
        chunk = frames[s:e]
        pad = block - chunk.shape[0]
        if pad:
            chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), chunk.dtype)])
        r = np.asarray(image_tools.resize_with_pad(chunk, size, size)).astype(np.uint8)
        out[s:e] = r[: e - s]
    return out


def _encode_video(av, frames: np.ndarray, out_path: pathlib.Path, fps: int, args):
    tmp = out_path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), mode="w") as cont:
        stream = cont.add_stream(args.vcodec, rate=int(fps),
                                 options={"crf": str(args.crf), "g": str(args.gop)})
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = args.pix_fmt
        for f in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            for pkt in stream.encode(vf):
                cont.mux(pkt)
        for pkt in stream.encode():
            cont.mux(pkt)
    tmp.replace(out_path)


def _numeric_stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def _image_stats(frames: np.ndarray, max_samples: int = 300) -> dict:
    T = frames.shape[0]
    step = max(1, T // max_samples)
    sample = (frames[::step].astype(np.float64) / 255.0).reshape(-1, 3)
    def col(f):
        return [[[float(v)]] for v in f(sample, axis=0)]
    return {"min": col(np.min), "max": col(np.max), "mean": col(np.mean),
            "std": col(np.std), "count": [int(frames[::step].shape[0])]}


def process_episode(ep: int, src_str: str, dst_str: str, fps: int, args) -> dict:
    av, pd, pq, image_tools = _lazy()
    src, dst = pathlib.Path(src_str), pathlib.Path(dst_str)
    info = _info(src)
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    chunks = info["chunks_size"]

    src_pq = src / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
    dst_pq = dst / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
    dst_vid = {c: dst / info["video_path"].format(episode_chunk=ep // chunks, video_key=c, episode_index=ep)
               for c in cams}
    sidecar = dst / ".cache_meta" / f"ep_{ep:06d}.json"

    if (not args.overwrite and sidecar.exists() and dst_pq.exists()
            and all(v.exists() for v in dst_vid.values())):
        return json.loads(sidecar.read_text())

    df = pq.read_table(src_pq).to_pandas()
    T = len(df)

    # copy parquet verbatim
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pq, dst_pq)

    # numeric stats (all non-video features)
    stats: dict = {}
    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            continue
        col = df[key].to_numpy()
        arr = np.stack(col) if np.ndim(col[0]) > 0 else np.asarray(col).reshape(T, 1)
        stats[key] = _numeric_stats(arr.astype(np.float64))

    # resize + re-encode each camera
    for cam in cams:
        src_v = src / info["video_path"].format(episode_chunk=ep // chunks, video_key=cam, episode_index=ep)
        frames = []
        with av.open(str(src_v)) as cont:
            for fr in cont.decode(video=0):
                frames.append(fr.to_ndarray(format="rgb24"))
        frames = np.stack(frames)
        if frames.shape[0] < T:
            frames = np.concatenate([frames, np.repeat(frames[-1:], T - frames.shape[0], 0)])
        elif frames.shape[0] > T:
            frames = frames[:T]
        resized = _resize_blocks(image_tools, frames, args.size)
        stats[cam] = _image_stats(resized)
        dst_vid[cam].parent.mkdir(parents=True, exist_ok=True)
        _encode_video(av, resized, dst_vid[cam], fps, args)

    out = {"episode_index": ep, "length": T, "stats": stats}
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--vcodec", default="libx264", choices=["libx264", "h264", "hevc", "libsvtav1"])
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--gop", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src).resolve()
    dst = pathlib.Path(args.dst).resolve()
    _, _, pq, _ = _lazy()
    info = _info(src)
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    fps = info.get("fps", 25)
    eps = _present_episodes(src, info, cams)
    print(f"[resize] {src.name}: {len(eps)} eps, cams={[c.split('.')[-1] for c in cams]} "
          f"-> {args.size}x{args.size} {args.vcodec}/crf{args.crf}  dst={dst}")

    # static meta copy
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for fn in ["tasks.jsonl", "episodes.jsonl"]:
        if (src / "meta" / fn).exists():
            shutil.copy2(src / "meta" / fn, dst / "meta" / fn)
    if (src / "README.md").exists():
        shutil.copy2(src / "README.md", dst / "README.md")

    results, errors = [], []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_episode, ep, str(src), str(dst), fps, args): ep for ep in eps}
        done = 0
        for fut in cf.as_completed(futs):
            ep = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa
                errors.append((ep, repr(e)))
                print(f"[ERR] ep{ep}: {e}\n{traceback.format_exc()}")
            done += 1
            if done % 10 == 0 or done == len(eps):
                print(f"[resize] {done}/{len(eps)} episodes")

    # ── rebuild meta ──────────────────────────────────────────────────────────
    results = sorted(results, key=lambda r: r["episode_index"])
    codec_name = {"libsvtav1": "av1", "libx264": "h264", "h264": "h264", "hevc": "hevc"}[args.vcodec]
    new_info = json.loads(json.dumps(info))
    for c in cams:
        new_info["features"][c]["shape"] = [args.size, args.size, 3]
        vi = new_info["features"][c].setdefault("info", {})
        vi["video.height"] = args.size
        vi["video.width"] = args.size
        vi["video.codec"] = codec_name
        vi["video.pix_fmt"] = args.pix_fmt
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    st_lines = [{"episode_index": r["episode_index"], "stats": r["stats"]} for r in results]
    (dst / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(s) for s in st_lines) + "\n")

    if errors:
        print(f"[resize] {len(errors)} FAILED: {errors[:10]}")
    print(f"[resize] done -> {dst}  ({len(results)} eps)")


if __name__ == "__main__":
    main()
