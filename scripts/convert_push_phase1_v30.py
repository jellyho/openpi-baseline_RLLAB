"""
Convert the v2.1 phase1 merged datasets to LeRobot v3.0 and push to the hub
(creating the v3.0 tag), then mark public.

Runs lerobot's official ``convert_dataset_v21_to_v30`` but monkeypatches
``concatenate_video_files`` with a DTS-safe variant: lerobot stream-copies packets
through the ffmpeg concat demuxer, which raises "non monotonically increasing dts"
when an episode-boundary packet lands on a duplicate DTS (seen with short clips).
The patched version bumps any non-increasing DTS by the minimum delta — still a
pure stream copy (no re-encode, no quality loss).

Each dataset is expected at ``<root>/<repo_id>`` (we stage symlinks under
v30_stage/jellyho/<task>_rl_224 -> phase1_merged/<task>).

Run (in the v3.0 .venv):
    HF_HUB_ENABLE_HF_TRANSFER=1 JAX_PLATFORMS=cpu \
        ./.venv/bin/python scripts/convert_push_phase1_v30.py
"""

import shutil
import tempfile
from pathlib import Path

import av
from huggingface_hub import HfApi

import lerobot.datasets.video_utils as video_utils
import lerobot.datasets.v30.convert_dataset_v21_to_v30 as conv

STAGE = "/data5/jellyho/PFR_RSS/dataset/v30_stage"
TASKS = ["insert-mouse-battery", "tower-of-hanoi-game", "seal-water-bottle-cap"]


def concatenate_video_files_dts_safe(input_video_paths, output_video_path, overwrite: bool = True):
    """lerobot's concatenate_video_files + monotonic-DTS guard (stream copy)."""
    output_video_path = Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        return
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        for p in input_video_paths:
            f.write(f"file '{str(Path(p).resolve())}'\n")
        f.flush()
        concat_path = f.name

    input_container = av.open(concat_path, mode="r", format="concat", options={"safe": "0"})
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_out = tf.name
    output_container = av.open(tmp_out, mode="w", options={"movflags": "faststart"})

    stream_map = {}
    for s in input_container.streams:
        if s.type in ("video", "audio", "subtitle"):
            stream_map[s.index] = output_container.add_stream_from_template(template=s, opaque=True)
            stream_map[s.index].time_base = s.time_base

    last_dts: dict[int, int] = {}
    for packet in input_container.demux():
        if packet.stream.index not in stream_map:
            continue
        if packet.dts is None:
            continue
        out_stream = stream_map[packet.stream.index]
        packet.stream = out_stream
        prev = last_dts.get(out_stream.index)
        if prev is not None and packet.dts <= prev:
            shift = prev + 1 - packet.dts
            packet.dts += shift
            if packet.pts is not None:
                packet.pts += shift
        last_dts[out_stream.index] = packet.dts
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_out, output_video_path)
    Path(concat_path).unlink()


# Patch both the source module and the name imported into the converter module.
video_utils.concatenate_video_files = concatenate_video_files_dts_safe
conv.concatenate_video_files = concatenate_video_files_dts_safe


def main():
    api = HfApi()
    for task in TASKS:
        repo = f"jellyho/{task}_rl_224"
        print(f"\n==================== convert+push {repo} ====================", flush=True)
        conv.convert_dataset(repo_id=repo, root=STAGE, push_to_hub=True, force_conversion=True)
        api.update_repo_settings(repo_id=repo, repo_type="dataset", private=False)
        print(f"PUBLIC {repo}", flush=True)
        shutil.rmtree(Path(STAGE) / "jellyho" / f"{task}_rl_224_v30", ignore_errors=True)
        print(f"DONE {repo} -> https://huggingface.co/datasets/{repo}", flush=True)
    print("\nALL PHASE1 V30 CONVERT+PUSH DONE", flush=True)


if __name__ == "__main__":
    main()
