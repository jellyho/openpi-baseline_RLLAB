"""In-RAM frame preloading for LeRobot v3.0 datasets.

The training data loader was GPU-starved: lerobot decodes camera frames per
sample with torchcodec, and for v3.0 *packed* videos (many episodes per mp4) every
random access re-opens + indexes a large file over NFS.  With 64 workers hammering
the same packed files this can't keep the GPU fed (observed: 14-min loader init,
~13 s/step at batch 1024, sawtooth GPU utilization).

Fix: decode every frame ONCE into a per-camera uint8 memmap on /dev/shm (RAM-backed
tmpfs), then serve frames by index from the memmap — no per-step video decode, no
NFS, shared across workers via mmap.

Usage (opt-in, no behavior change unless enabled):
    OPENPI_PRELOAD_FRAMES=1 ./train.sh <config> ...
The cache is built on first use (idempotent) under
    $OPENPI_PRELOAD_DIR (default /dev/shm/openpi_framecache)/<repo_id>/
and reused by every worker.  Size = total_frames * H * W * 3 * n_cameras bytes
(e.g. aloha 224: 11829*224*224*3*3 ≈ 5.3 GB — trivially fits the node's RAM).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames


def cache_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("OPENPI_PRELOAD_DIR", "/dev/shm/openpi_framecache"))


def _safe(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _video_keys(meta) -> list[str]:
    return [k for k, v in meta.features.items() if v.get("dtype") == "video"]


def _frame_shape(meta, cam: str) -> tuple[int, int]:
    shp = meta.features[cam]["shape"]  # [H, W, C]
    return int(shp[0]), int(shp[1])


def build_frame_cache(repo_id: str, root: str | None = None, *, fps_round: bool = True) -> pathlib.Path:
    """Decode every frame of every camera into per-camera uint8 memmaps (HWC).

    Layout: <cache>/<repo>/<cam>.u8  shape [total_frames, H, W, 3]
            <cache>/<repo>/index.json  {fps, total_frames, cams, shapes, ep_starts}
    Idempotent: returns immediately if a complete cache already exists.
    """
    meta_ds = LeRobotDataset(repo_id, root=root)
    meta = meta_ds.meta
    cams = _video_keys(meta)
    fps = meta.fps
    total = meta.total_frames
    ep_lengths = [int(meta.episodes[e]["length"]) for e in range(meta.total_episodes)]
    ep_starts = np.cumsum([0] + ep_lengths)[:-1].tolist()  # global start index per episode

    out_dir = cache_root() / _safe(repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"

    shapes = {c: _frame_shape(meta, c) for c in cams}
    index = {
        "repo_id": repo_id, "fps": fps, "total_frames": int(total),
        "cams": cams, "shapes": {c: list(shapes[c]) for c in cams},
        "ep_starts": ep_starts, "ep_lengths": ep_lengths, "complete": False,
    }

    # Reuse if a *complete* cache for the same shape/total already exists.
    if index_path.exists():
        try:
            prev = json.loads(index_path.read_text())
            if prev.get("complete") and prev.get("total_frames") == int(total) and prev.get("cams") == cams:
                return out_dir
        except Exception:
            pass

    index_path.write_text(json.dumps(index))
    for cam in cams:
        H, W = shapes[cam]
        mm = np.memmap(out_dir / f"{cam}.u8", dtype=np.uint8, mode="w+", shape=(total, H, W, 3))
        for ep in range(meta.total_episodes):
            ep_obj = meta.episodes[ep]
            length = ep_lengths[ep]
            from_ts = ep_obj[f"videos/{cam}/from_timestamp"]
            # one batched decode for the whole episode (sequential -> fast)
            ts = [from_ts + i / fps for i in range(length)]
            video_path = meta_ds.root / meta.get_video_file_path(ep, cam)
            frames = decode_video_frames(video_path, ts, meta_ds.tolerance_s, meta_ds.video_backend)
            # frames: [length, C, H, W] float32 [0,1] -> uint8 HWC
            arr = (frames.clamp(0, 1).mul(255).round().to(torch.uint8)
                   .permute(0, 2, 3, 1).contiguous().numpy())
            gstart = ep_starts[ep]
            mm[gstart:gstart + length] = arr[:length]
        mm.flush()
        del mm
    index["complete"] = True
    index_path.write_text(json.dumps(index))
    return out_dir


class PreloadedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset whose video frames come from the RAM memmap cache.

    Only `_query_videos` is overridden; all the windowing / padding / column logic
    is inherited unchanged, so samples are identical to the video-decoding path
    (frames returned as [n, C, H, W] float32 in [0, 1]).
    """

    def _init_cache(self):
        if getattr(self, "_pf_index", None) is not None:
            return
        cdir = cache_root() / _safe(self.repo_id)
        self._pf_dir = cdir
        self._pf_index = json.loads((cdir / "index.json").read_text())
        self._pf_mm = {}  # lazily opened per worker

    def _mm(self, cam: str) -> np.memmap:
        if cam not in self._pf_mm:
            H, W = self._pf_index["shapes"][cam]
            total = self._pf_index["total_frames"]
            self._pf_mm[cam] = np.memmap(self._pf_dir / f"{cam}.u8", dtype=np.uint8,
                                         mode="r", shape=(total, H, W, 3))
        return self._pf_mm[cam]

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        self._init_cache()
        fps = self._pf_index["fps"]
        gstart = self._pf_index["ep_starts"][ep_idx]
        length = self._pf_index["ep_lengths"][ep_idx]
        item = {}
        for cam, query_ts in query_timestamps.items():
            local = np.clip(np.round(np.asarray(query_ts) * fps).astype(np.int64), 0, length - 1)
            frames = self._mm(cam)[gstart + local]                 # [n, H, W, 3] uint8
            t = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32).div_(255.0)
            t = t.permute(0, 3, 1, 2).contiguous()                 # [n, C, H, W]
            item[cam] = t.squeeze(0)
        return item


def maybe_build_cache(repo_id: str, root: str | None = None) -> bool:
    """Build the cache if OPENPI_PRELOAD_FRAMES is set. Returns whether preload is on."""
    if os.environ.get("OPENPI_PRELOAD_FRAMES", "0") != "1":
        return False
    build_frame_cache(repo_id, root)
    return True
