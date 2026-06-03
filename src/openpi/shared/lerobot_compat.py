"""LeRobot version-compat shim.

LeRobot's dataset format v3.0 (lerobot >= ~0.2) renamed the package layout from
``lerobot.common.datasets.*`` to ``lerobot.datasets.*``.  Import the dataset
classes from whichever path the installed version exposes, so openpi works with
BOTH a v2.1 lerobot (the currently pinned version) and a v3.0 lerobot (after a
``uv sync`` to a newer pin + converting the datasets with
``python -m lerobot.datasets.v30.convert_dataset_v21_to_v30``).

The public API openpi relies on is unchanged between v2.1 and v3.0:
  - ``LeRobotDataset(repo_id, root=, delta_timestamps=)``
  - ``LeRobotDatasetMetadata(repo_id, root=)``
  - the ``.meta`` / ``.fps`` / ``.features`` / ``.video_keys`` / ``.tasks`` attrs

so only the import path differs.  NOTE: a v3.0 lerobot CANNOT read a v2.1 dataset
(it raises ``BackwardCompatibilityError``); the datasets must be converted first.
"""

try:  # lerobot >= ~0.2 (dataset format v3.0): lerobot.datasets.*
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:  # lerobot v2.1 (legacy lerobot.common.datasets.* layout)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

__all__ = ["LeRobotDataset", "LeRobotDatasetMetadata"]
