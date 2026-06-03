"""
Process + per-task merge the RSS-2026 Phase-1 dataset for LPS-RFT training.

For each task (insert-mouse-battery / seal-water-bottle-cap / tower-of-hanoi-game)
this merges its three subsets (expert-data + success-and-hil-data + failure-data)
into ONE LeRobot v2.1 dataset, while:

  1. DROPPING the trailing `restore` (post-task scene-reset) frames.
     `restore` is always a contiguous suffix of the episode (verified across all
     episodes), so we truncate the episode at the first `restore` frame — keeping
     the trajectory contiguous (required for the chunked TD target / next-obs).

  2. ANNOTATING reward + mc_return with our RECAP / LPS-RFT convention
     (identical to scripts/compute_mc_returns.py, but success is taken from the
     SUBSET rather than a `next.success` column, since Phase-1 has none):
        success := subset != "failure-data"   (expert + success-and-hil = success)
        r_t      = -1 per step;  terminal = 0 (success) else -C_fail
        G_t      = Σ_{k>=t} γ^{k-t} r_k
        C_fail   = cfail_frac * norm_len           (default 0.5 * norm_len)
        norm_len = max FILTERED episode length over the WHOLE task (all 3 subsets)
        reward     = r_t / norm_len                (NOT clipped; in range)
        mc_return  = clip(G_t / norm_len, -1, 0)
     Same scale → telescoping identity holds:
        Σ_i γ^i reward_i == mc_return_t − γ^H mc_return_{t+H}.

  3. RESIZING every camera frame to `--size` (default 224x224) with EXACTLY the
     training-time transform — openpi `image_tools.resize_with_pad` (aspect-
     preserving letterbox).  Stored frames therefore equal what the model sees,
     so the train-time ResizeImages(224,224) becomes a no-op.  Videos are
     re-encoded with LeRobot's default params (libsvtav1 / yuv420p / g=2 / crf=30).

The output is a valid LeRobot v2.1 dataset per task:
    <dst-root>/<task>/{meta,data,videos}

Episodes are ordered expert → success-and-hil → failure; meta/sources.jsonl
records each subset's episode range.  `mcap_path` and `subtask` columns are
dropped (absolute paths / placeholder); `observation.commander_state` is kept.

Parallel + resumable: each episode writes a sidecar under <dst>/.cache_meta/;
re-runs skip episodes whose parquet + videos + sidecar already exist.

Usage
─────
    # smoke test: one task, first few episodes per subset
    JAX_PLATFORMS=cpu uv run scripts/process_phase1_merge.py \
        --tasks seal-water-bottle-cap --limit 2 --workers 4

    # full run (all tasks)
    JAX_PLATFORMS=cpu uv run scripts/process_phase1_merge.py --workers 32
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import pathlib
import shutil
import traceback

import numpy as np

SRC_ROOT_DEFAULT = "/data5/jellyho/PFR_RSS/dataset/phase1"
DST_ROOT_DEFAULT = "/data5/jellyho/PFR_RSS/dataset/phase1_merged"
TASKS_DEFAULT = ["insert-mouse-battery", "seal-water-bottle-cap", "tower-of-hanoi-game"]
SUBSETS_DEFAULT = ["expert-data", "success-and-hil-data", "failure-data"]
DROP_COLUMNS = ["mcap_path", "subtask"]
DROP_STATES = ("restore",)  # frames whose commander_state is in here -> truncate tail


# --------------------------------------------------------------------------- #
# Lazy heavy imports (so --help is fast and workers import jax on CPU once)
# --------------------------------------------------------------------------- #
def _lazy():
    # Default to CPU so a background data job never grabs training GPUs, and cap
    # per-process threads so N workers don't oversubscribe the machine.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true "
                                       "intra_op_parallelism_threads=2")
    import av  # noqa
    import pandas as pd  # noqa
    import pyarrow.parquet as pq  # noqa
    import jax  # noqa
    from openpi.shared import image_tools  # noqa
    return av, pd, pq, jax, image_tools


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class EpPlan:
    task: str
    subset: str
    src_ep: int          # episode index in the source subset
    new_ep: int          # episode index in the merged target
    gidx_start: int      # global frame-index offset of this episode in the target
    success: bool
    norm_len: int
    cfail: float
    gamma: float
    instruction: str


def _info(path: pathlib.Path) -> dict:
    return json.loads((path / "meta" / "info.json").read_text())


def _present_episodes(subset_dir: pathlib.Path, info: dict, cams: list[str]) -> list[int]:
    """Episodes that have a parquet AND all camera videos present."""
    chunks = info["chunks_size"]
    out = []
    for ep in range(info["total_episodes"]):
        pq_path = subset_dir / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
        if not pq_path.exists():
            continue
        ok = all(
            (subset_dir / info["video_path"].format(
                episode_chunk=ep // chunks, video_key=c, episode_index=ep)).exists()
            for c in cams
        )
        if ok:
            out.append(ep)
    return out


def _filtered_length(pq, pq_path: pathlib.Path) -> int:
    """Length after truncating the trailing DROP_STATES (restore) block."""
    df = pq.read_table(pq_path, columns=["observation.commander_state"]).to_pandas()
    states = df["observation.commander_state"].astype(str).to_numpy()
    n = len(states)
    drop = np.isin(states, DROP_STATES)
    if not drop.any():
        return n
    return int(np.argmax(drop))  # first index of a DROP_STATE == kept length


def build_plan(args) -> tuple[list[EpPlan], dict]:
    _, _, pq, _, _ = _lazy()
    src_root = pathlib.Path(args.src_root)
    plan: list[EpPlan] = []
    task_meta: dict = {}

    for task in args.tasks:
        # discover per-subset present episodes + filtered lengths
        per_subset = {}
        all_lengths = []
        instruction = None
        cams = None
        template_info = None
        for subset in args.subsets:
            sdir = src_root / task / subset
            if not sdir.exists():
                continue
            info = _info(sdir)
            if cams is None:
                cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
                template_info = info
            tasks_jsonl = (sdir / "meta" / "tasks.jsonl").read_text().splitlines()[0]
            instruction = json.loads(tasks_jsonl)["task"]
            eps = _present_episodes(sdir, info, cams)
            if args.limit:
                eps = eps[: args.limit]
            lengths = {}
            chunks = info["chunks_size"]
            kept = []
            for ep in eps:
                p = sdir / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
                L = _filtered_length(pq, p)
                if L <= 0:
                    # Degenerate episode (entirely restore/drop-state) → no task
                    # data; skip so episode_index stays contiguous.
                    print(f"[plan] SKIP {task}/{subset} ep{ep}: empty after restore-trim")
                    continue
                lengths[ep] = L
                kept.append(ep)
            per_subset[subset] = (kept, lengths)
            all_lengths.extend(lengths.values())

        if not all_lengths:
            print(f"[plan] {task}: no episodes found, skipping")
            continue
        norm_len = max(all_lengths)
        cfail = args.cfail_frac * norm_len

        new_ep = 0
        gidx = 0
        ranges = {}
        for subset in args.subsets:
            if subset not in per_subset:
                continue
            start = new_ep
            eps, lengths = per_subset[subset]
            success = subset != "failure-data"
            for src_ep in eps:
                plan.append(EpPlan(task, subset, src_ep, new_ep, gidx, success,
                                   norm_len, cfail, args.gamma, instruction))
                gidx += lengths[src_ep]
                new_ep += 1
            ranges[subset] = (start, new_ep - 1, new_ep - start)

        task_meta[task] = {
            "cams": cams, "norm_len": norm_len, "cfail": cfail,
            "instruction": instruction, "n_episodes": new_ep,
            "template_info": template_info, "ranges": ranges,
        }
        print(f"[plan] {task}: {new_ep} eps | norm_len={norm_len} cfail={cfail:.1f} "
              f"| ranges={ {k: v[2] for k, v in ranges.items()} }")
    return plan, task_meta


# --------------------------------------------------------------------------- #
# Per-episode worker
# --------------------------------------------------------------------------- #
def episode_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    T = len(rewards)
    g = np.zeros(T, dtype=np.float32)
    g[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        g[t] = rewards[t] + gamma * g[t + 1]
    return g


def _numeric_stats(arr: np.ndarray) -> dict:
    """arr: (T, D) -> lerobot per-dim stats (lists of length D)."""
    a = arr.astype(np.float64)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def _image_stats(frames: np.ndarray, max_samples: int = 300) -> dict:
    """frames: (T,H,W,3) uint8 -> per-channel (3,1,1) stats in [0,1]."""
    T = frames.shape[0]
    step = max(1, T // max_samples)
    sample = frames[::step].astype(np.float64) / 255.0   # (n,H,W,3)
    n = sample.shape[0]
    per_ch = sample.reshape(-1, 3)
    def col(f):
        return [[[float(v)]] for v in f(per_ch, axis=0)]  # (3,1,1)
    return {
        "min": col(np.min), "max": col(np.max),
        "mean": col(np.mean), "std": col(np.std),
        "count": [int(n)],
    }


def process_episode(ep: EpPlan, args) -> dict:
    av, pd, pq, jax, image_tools = _lazy()
    src_root = pathlib.Path(args.src_root)
    dst = pathlib.Path(args.dst_root) / ep.task
    cache_dir = dst / ".cache_meta"
    sidecar = cache_dir / f"ep_{ep.new_ep:06d}.json"

    sdir = src_root / ep.task / ep.subset
    sinfo = _info(sdir)
    cams = [k for k, v in sinfo["features"].items() if v.get("dtype") == "video"]
    chunks = sinfo["chunks_size"]
    src_pq = sdir / sinfo["data_path"].format(
        episode_chunk=ep.src_ep // chunks, episode_index=ep.src_ep)

    dst_chunks = 1000
    dst_pq = dst / f"data/chunk-{ep.new_ep // dst_chunks:03d}/episode_{ep.new_ep:06d}.parquet"
    dst_vid = {c: dst / f"videos/chunk-{ep.new_ep // dst_chunks:03d}/{c}/episode_{ep.new_ep:06d}.mp4"
               for c in cams}

    # resume: skip if everything already there
    if (not args.overwrite and sidecar.exists() and dst_pq.exists()
            and all(v.exists() for v in dst_vid.values())):
        return json.loads(sidecar.read_text())

    # ── read + truncate ──────────────────────────────────────────────────────
    df = pq.read_table(src_pq).to_pandas()
    states = df["observation.commander_state"].astype(str).to_numpy()
    drop = np.isin(states, DROP_STATES)
    T = int(np.argmax(drop)) if drop.any() else len(df)
    if T <= 0:
        raise ValueError(f"{ep.task}/{ep.subset} ep{ep.src_ep}: empty after restore-trim")
    df = df.iloc[:T].reset_index(drop=True)

    # ── reward + mc_return ───────────────────────────────────────────────────
    rewards = np.full(T, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if ep.success else -ep.cfail
    g = np.clip(episode_mc_returns(rewards, ep.gamma) / ep.norm_len, -1.0, 0.0)
    df["reward"] = (rewards / ep.norm_len).astype(np.float32)
    df["mc_return"] = g.astype(np.float32)

    # ── re-index + drop noise columns ────────────────────────────────────────
    for c in DROP_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=c)
    df["frame_index"] = np.arange(T, dtype=np.int64)
    df["episode_index"] = np.int64(ep.new_ep)
    df["index"] = np.arange(ep.gidx_start, ep.gidx_start + T, dtype=np.int64)
    df["task_index"] = np.int64(0)

    # ── per-episode stats (numeric) ──────────────────────────────────────────
    stats: dict = {}
    for key in ["observation.state", "action"]:
        stats[key] = _numeric_stats(np.stack(df[key].to_numpy()))
    for key in ["reward", "mc_return", "timestamp", "frame_index", "episode_index",
                "index", "task_index"]:
        stats[key] = _numeric_stats(np.asarray(df[key].to_numpy()).reshape(T, 1))

    # ── decode → resize_with_pad → re-encode each camera ─────────────────────
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        src_v = sdir / sinfo["video_path"].format(
            episode_chunk=ep.src_ep // chunks, video_key=cam, episode_index=ep.src_ep)
        frames = []
        with av.open(str(src_v)) as cont:
            for i, fr in enumerate(cont.decode(video=0)):
                if i >= T:
                    break
                frames.append(fr.to_ndarray(format="rgb24"))
        frames = np.stack(frames)                       # (t,180,320,3) uint8
        if frames.shape[0] < T:                         # pad short videos by repeating last
            frames = np.concatenate([frames, np.repeat(frames[-1:], T - frames.shape[0], 0)])

        resized = _resize_blocks(image_tools, frames, args.size)   # (T,224,224,3) uint8
        stats[cam] = _image_stats(resized)

        dst_vid[cam].parent.mkdir(parents=True, exist_ok=True)
        _encode_video(av, resized, dst_vid[cam], sinfo.get("fps", 60), args)

    df.to_parquet(dst_pq, index=False)

    out = {"new_ep": ep.new_ep, "length": T, "task": ep.task,
           "subset": ep.subset, "instruction": ep.instruction, "stats": stats}
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(out))
    return out


def _resize_blocks(image_tools, frames: np.ndarray, size: int, block: int = 256) -> np.ndarray:
    """resize_with_pad in fixed-size blocks (last block zero-padded then sliced) so
    the jitted kernel compiles for ONE batch shape instead of recompiling per T."""
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
    """Encode (T,H,W,3) uint8 RGB with LeRobot's default params."""
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


# --------------------------------------------------------------------------- #
# Meta assembly
# --------------------------------------------------------------------------- #
def write_meta(task: str, args, tmeta: dict, results: list[dict]):
    dst = pathlib.Path(args.dst_root) / task
    results = sorted(results, key=lambda r: r["new_ep"])
    n_eps = len(results)
    total_frames = sum(r["length"] for r in results)
    cams = tmeta["cams"]
    instruction = tmeta["instruction"]

    # info.json: start from a source template, patch features/totals/video info
    info = json.loads(json.dumps(tmeta["template_info"]))
    feats = info["features"]
    for c in DROP_COLUMNS:
        feats.pop(c, None)
    feats["reward"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["mc_return"] = {"dtype": "float32", "shape": [1], "names": None}
    codec_name = {"libsvtav1": "av1", "libx264": "h264", "h264": "h264", "hevc": "hevc"}[args.vcodec]
    for c in cams:
        feats[c]["shape"] = [args.size, args.size, 3]
        vinfo = feats[c].setdefault("info", {})
        vinfo["video.height"] = args.size
        vinfo["video.width"] = args.size
        vinfo["video.codec"] = codec_name
        vinfo["video.pix_fmt"] = args.pix_fmt
    info.update({
        "total_episodes": n_eps, "total_frames": total_frames,
        "total_tasks": 1, "total_videos": n_eps * len(cams),
        "total_chunks": (n_eps - 1) // 1000 + 1, "chunks_size": 1000,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    })

    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")

    ep_lines, st_lines, gidx = [], [], 0
    for r in results:
        ep_lines.append({"episode_index": r["new_ep"], "tasks": [instruction], "length": r["length"]})
        st_lines.append({"episode_index": r["new_ep"], "stats": r["stats"]})
        gidx += r["length"]
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in ep_lines) + "\n")
    (meta / "episodes_stats.jsonl").write_text("\n".join(json.dumps(s) for s in st_lines) + "\n")

    with (meta / "sources.jsonl").open("w") as f:
        for subset, (lo, hi, n) in tmeta["ranges"].items():
            f.write(json.dumps({
                "source_path": str(pathlib.Path(args.src_root) / task / subset),
                "subset": subset, "success": subset != "failure-data",
                "episode_index_start": lo, "episode_index_end": hi, "num_episodes": n,
            }) + "\n")
    print(f"[meta] {task}: wrote info/tasks/episodes/episodes_stats/sources "
          f"({n_eps} eps, {total_frames} frames)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=SRC_ROOT_DEFAULT)
    ap.add_argument("--dst-root", default=DST_ROOT_DEFAULT)
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS_DEFAULT)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--cfail-frac", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--vcodec", default="libsvtav1",
                    choices=["libsvtav1", "libx264", "h264", "hevc"])
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--gop", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="max episodes per subset (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    plan, task_meta = build_plan(args)
    if not plan:
        print("Nothing to do.")
        return

    by_task: dict[str, list[EpPlan]] = {}
    for p in plan:
        by_task.setdefault(p.task, []).append(p)

    print(f"[run] {len(plan)} episodes across {len(by_task)} tasks | workers={args.workers} "
          f"| size={args.size} codec={args.vcodec} crf={args.crf}")

    results_by_task: dict[str, list[dict]] = {t: [] for t in by_task}
    errors = []

    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_episode, p, args): p for p in plan}
        done = 0
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
                results_by_task[res["task"]].append(res)
            except Exception as e:  # noqa
                errors.append((p, repr(e)))
                print(f"[ERR] {p.task}/{p.subset} ep{p.src_ep}: {e}\n{traceback.format_exc()}")
            done += 1
            if done % 25 == 0 or done == len(plan):
                print(f"[run] {done}/{len(plan)} episodes done")

    for task, results in results_by_task.items():
        if results:
            write_meta(task, args, task_meta[task], results)

    if errors:
        print(f"\n[run] {len(errors)} episodes FAILED:")
        for p, e in errors[:20]:
            print(f"  {p.task}/{p.subset} ep{p.src_ep}: {e}")
    print("[run] done.")


if __name__ == "__main__":
    main()
