"""Fetch the dataset camera videos for the eval value-curve episodes.

The eval plot (eval/value_curves) shows a fixed set of success/failure episodes
(selected by seed in vla_eval.build_eval_set). This script cuts the corresponding
clips out of the dataset's aggregated LeRobot v3 videos so you can watch what the
critic is being evaluated on.

Episode -> video mapping comes from meta/episodes (per-episode video file_index +
from/to timestamps). Cutting uses the imageio-ffmpeg bundled ffmpeg with stream copy
(no re-encode; start may snap to the previous keyframe, fine for review).

Usage:
    python vla_eval_videos.py                                   # both ready tasks, cam_high
    python vla_eval_videos.py --task insert-mouse-battery --cam cam_left_wrist
    # output: docs/eval_videos/<task>/<success|failure>_ep<N>.mp4
"""

import argparse
import dataclasses
import glob
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq


def eval_episodes(task: str):
    """Reproduce vla_eval.build_eval_set's episode selection (same seed logic)."""
    import vla_eval
    from vla_config import get_config
    from vla_data import VLALeRobotDataset

    cfg = dataclasses.replace(get_config("vla_aqc_td_macro"), task=task)
    ds = VLALeRobotDataset(cfg.data_root, horizon=cfg.horizon, include_base_action=False,
                           mc_gamma=cfg.td.mc_gamma, discount=cfg.td.discount,
                           relabel_living=cfg.reward.relabel_living,
                           relabel_fail=cfg.reward.relabel_fail)
    groups, is_fail = vla_eval.scan_episodes(ds)
    all_eps = sorted(groups)
    fail_eps = [e for e in all_eps if is_fail.get(e)]
    succ_eps = [e for e in all_eps if not is_fail.get(e)]
    rng = np.random.default_rng(cfg.seed)
    n_s, n_f = cfg.eval_n_success, cfg.eval_n_fail
    sel_fail = sorted(rng.choice(fail_eps, size=min(n_f, len(fail_eps)), replace=False).tolist())
    sel_succ = sorted(rng.choice(succ_eps, size=min(n_s, len(succ_eps)), replace=False).tolist())
    return cfg.data_root, sel_succ, sel_fail


def video_index(data_root: str, cam: str):
    """episode_index -> (video_path, from_ts, to_ts) from meta/episodes."""
    files = sorted(glob.glob(os.path.join(data_root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    key = f"videos/observation.images.{cam}"
    out = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", f"{key}/chunk_index",
                                      f"{key}/file_index", f"{key}/from_timestamp",
                                      f"{key}/to_timestamp"])
        for i in range(t.num_rows):
            ep = t["episode_index"][i].as_py()
            ci = t[f"{key}/chunk_index"][i].as_py()
            fi = t[f"{key}/file_index"][i].as_py()
            path = os.path.join(data_root, "videos", f"observation.images.{cam}",
                                f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
            out[ep] = (path, t[f"{key}/from_timestamp"][i].as_py(),
                       t[f"{key}/to_timestamp"][i].as_py())
    return out


def cut(ffmpeg: str, src: str, start: float, end: float, dst: str):
    dur = end - start
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", src,
           "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero", dst]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="", help="one task; default = both ready tasks")
    ap.add_argument("--cam", default="cam_high",
                    choices=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    ap.add_argument("--out", default="docs/eval_videos")
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    tasks = [args.task] if args.task else ["insert-mouse-battery", "seal-water-bottle-cap"]
    for task in tasks:
        root, succ, fail = eval_episodes(task)
        vidx = video_index(root, args.cam)
        odir = os.path.join(args.out, task)
        os.makedirs(odir, exist_ok=True)
        print(f"[{task}] success={succ} failure={fail}  cam={args.cam}")
        for label, eps in [("success", succ), ("failure", fail)]:
            for ep in eps:
                src, ts0, ts1 = vidx[ep]
                dst = os.path.join(odir, f"{label}_ep{ep:04d}.mp4")
                cut(ffmpeg, src, ts0, ts1, dst)
                print(f"  {label} ep{ep}: {ts1-ts0:6.1f}s  "
                      f"{os.path.getsize(dst)/1e6:6.1f}MB  -> {dst}")


if __name__ == "__main__":
    main()
