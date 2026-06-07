"""Merge the three DualYam LeRobot v3.0 datasets into one combined dataset.

Combines:
  /home/yonsei_jell/insert-mouse-battery
  /home/yonsei_jell/seal-water-bottle-cap
  /home/yonsei_jell/tower-of-hanoi-game
into:
  /home/yonsei_jell/dualyam_combined   (repo_id: jellyho/dualyam_combined)

Uses lerobot's built-in aggregate_datasets, which re-indexes episodes/frames,
unifies the 3 tasks, concatenates videos, and recomputes stats.
"""

from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

HOME = Path("/home/yonsei_jell")
TASKS = ["insert-mouse-battery", "seal-water-bottle-cap", "tower-of-hanoi-game"]
AGGR_REPO_ID = "jellyho/dualyam_combined"
AGGR_ROOT = HOME / "dualyam_combined"


def main():
    roots = [HOME / t for t in TASKS]
    for r in roots:
        if not (r / "meta" / "info.json").exists():
            raise FileNotFoundError(f"missing source dataset: {r}")
    if AGGR_ROOT.exists():
        raise FileExistsError(f"destination already exists, remove it first: {AGGR_ROOT}")

    aggregate_datasets(
        repo_ids=TASKS,
        aggr_repo_id=AGGR_REPO_ID,
        roots=roots,
        aggr_root=AGGR_ROOT,
    )
    print(f"DONE -> {AGGR_ROOT}")


if __name__ == "__main__":
    main()
