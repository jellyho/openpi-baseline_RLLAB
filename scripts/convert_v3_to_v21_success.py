#!/usr/bin/env python3
"""
Convert a LeRobot v3.0 dataset to v2.1 format.

Usage:
    # Success only
    python scripts/convert_v3_to_v21_success.py \
        --src /data5/jellyho/tabletop/rl_rollouts \
        --tgt /data5/jellyho/tabletop/rl_rollouts_v21_success \
        --repo_id jellyho/aloha_handover_box_joint_pos_rl_success \
        --success_only

    # All episodes
    python scripts/convert_v3_to_v21_success.py \
        --src /data5/jellyho/tabletop/rl_rollouts \
        --tgt /data5/jellyho/tabletop/rl_rollouts_v21_all \
        --repo_id jellyho/aloha_handover_box_joint_pos_rl_new
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


VIDEO_KEYS = [
    "observation.images.back",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]


def get_successful_episode_indices(data_df: pd.DataFrame) -> list[int]:
    last_frames = data_df.groupby("episode_index").last()
    return sorted(
        int(idx) for idx in last_frames.index
        if bool(last_frames.loc[idx, "next.success"])
    )


def extract_video_segment(
    src_video: Path,
    tgt_video: Path,
    from_ts: float,
    to_ts: float,
    encoder: str,
) -> None:
    tgt_video.parent.mkdir(parents=True, exist_ok=True)
    duration = to_ts - from_ts

    if encoder == "libsvtav1":
        vcodec_args = ["-c:v", "libsvtav1", "-preset", "10", "-crf", "35"]
    else:  # copy
        vcodec_args = ["-c:v", "copy"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(from_ts),
        "-i", str(src_video),
        "-t", str(duration),
        *vcodec_args,
        "-pix_fmt", "yuv420p",
        "-an",
        str(tgt_video),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_video}:\n{result.stderr.decode()}"
        )


def detect_encoder() -> str:
    result = subprocess.run(
        ["ffmpeg", "-encoders"], capture_output=True, text=True
    )
    if "libsvtav1" in result.stdout:
        print("[encoder] using libsvtav1 (CPU fast)")
        return "libsvtav1"
    print("[encoder] falling back to stream copy")
    return "copy"


def build_v21_info(src_info: dict, total_episodes: int, total_frames: int) -> dict:
    info = dict(src_info)
    info["codebase_version"] = "v2.1"
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    info["chunks_size"] = 1000
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_videos"] = total_episodes * len(VIDEO_KEYS)
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    # Remove v3.0-specific fields
    for k in ["video_path_by_modality", "chunks"]:
        info.pop(k, None)
    return info


def convert(src: Path, tgt: Path, repo_id: str, success_only: bool = True) -> None:
    if tgt.exists() and any(tgt.iterdir()):
        print(f"ERROR: {tgt} already exists and is not empty.", file=sys.stderr)
        sys.exit(1)

    # Load source data
    print("[1/5] Loading source data ...")
    data_df = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    ep_meta_df = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
    src_info = json.loads((src / "meta/info.json").read_text())

    if src_info.get("codebase_version") != "v3.0":
        print(f"Warning: expected v3.0, got {src_info.get('codebase_version')}")

    # Identify episodes to keep
    total_src = src_info.get("total_episodes", len(ep_meta_df))
    all_indices = sorted(ep_meta_df["episode_index"].tolist())
    if success_only:
        success_indices = get_successful_episode_indices(data_df)
        print(
            f"[2/5] Successful episodes: {len(success_indices)}/{total_src} "
            f"({len(success_indices)/total_src:.1%})"
        )
        print(f"      Failed: {[i for i in all_indices if i not in success_indices]}")
    else:
        success_indices = all_indices
        print(f"[2/5] All episodes: {len(success_indices)}/{total_src}")

    encoder = detect_encoder()

    # Prepare output directories
    (tgt / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (tgt / "meta").mkdir(parents=True, exist_ok=True)
    for vk in VIDEO_KEYS:
        (tgt / f"videos/chunk-000/{vk}").mkdir(parents=True, exist_ok=True)

    # Process episodes
    print("[3/5] Converting episodes ...")
    episodes_meta = []
    total_frames = 0

    for new_ep_idx, old_ep_idx in enumerate(tqdm(success_indices, desc="episodes")):
        ep_row = ep_meta_df[ep_meta_df["episode_index"] == old_ep_idx].iloc[0]

        # --- Parquet ---
        ep_data = data_df[data_df["episode_index"] == old_ep_idx].copy()
        ep_length = len(ep_data)
        ep_data["episode_index"] = new_ep_idx
        ep_data["index"] = range(total_frames, total_frames + ep_length)
        ep_data.to_parquet(
            tgt / f"data/chunk-000/episode_{new_ep_idx:06d}.parquet",
            index=False,
        )

        # --- Videos ---
        for vk in VIDEO_KEYS:
            from_ts = float(ep_row[f"videos/{vk}/from_timestamp"])
            to_ts = float(ep_row[f"videos/{vk}/to_timestamp"])
            src_video = src / f"videos/{vk}/chunk-000/file-000.mp4"
            tgt_video = tgt / f"videos/chunk-000/{vk}/episode_{new_ep_idx:06d}.mp4"
            extract_video_segment(src_video, tgt_video, from_ts, to_ts, encoder)

        episodes_meta.append({
            "episode_index": new_ep_idx,
            "tasks": list(ep_row["tasks"]),
            "length": ep_length,
        })
        total_frames += ep_length

    # Write meta files
    print("[4/5] Writing metadata ...")

    # info.json
    new_info = build_v21_info(src_info, len(success_indices), total_frames)
    (tgt / "meta/info.json").write_text(json.dumps(new_info, indent=2))

    # tasks.jsonl (copy from source tasks.parquet)
    # v3.0 tasks.parquet: task string is the row index, task_index is the column
    tasks_pq = src / "meta/tasks.parquet"
    if tasks_pq.exists():
        tasks_df = pd.read_parquet(tasks_pq)
        with open(tgt / "meta/tasks.jsonl", "w") as f:
            for task_str, row in tasks_df.iterrows():
                f.write(json.dumps({"task_index": int(row["task_index"]), "task": str(task_str)}) + "\n")
    else:
        # Derive from episodes
        all_tasks = set()
        for ep in episodes_meta:
            all_tasks.update(ep["tasks"])
        with open(tgt / "meta/tasks.jsonl", "w") as f:
            for i, task in enumerate(sorted(all_tasks)):
                f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # episodes.jsonl
    with open(tgt / "meta/episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    print(f"[5/5] Done.")
    print(f"      Output: {tgt}")
    print(f"      Episodes : {len(success_indices)}")
    print(f"      Frames   : {total_frames}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source v3.0 dataset path")
    parser.add_argument("--tgt", required=True, help="target v2.1 output path")
    parser.add_argument("--repo_id", required=True, help="repo_id for the new dataset")
    parser.add_argument("--success_only", action="store_true", help="keep only successful episodes")
    args = parser.parse_args()

    convert(Path(args.src), Path(args.tgt), args.repo_id, success_only=args.success_only)


if __name__ == "__main__":
    main()
