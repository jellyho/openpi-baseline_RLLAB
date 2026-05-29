import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves per-episode rollout videos."""

    def __init__(self, out_dir: pathlib.Path, fps: int = 25) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._fps = fps
        self._images: list[np.ndarray] = []

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        img = observation["images"]["cam_high"]  # [C, H, W]
        self._images.append(np.transpose(img, (1, 2, 0)))  # -> [H, W, C]

    @override
    def on_episode_end(self) -> None:
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"
        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(out_path, [np.asarray(x) for x in self._images], fps=self._fps)
