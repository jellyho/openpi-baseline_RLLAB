"""Upload the re-annotated dataset to HuggingFace.

Uploads insert-mouse-battery_annotated_v2 to Gwanwoo/<HF_REPO_NAME> as a
LeRobot v3.0-compatible dataset repo. Patches meta/stats.json with the
updated reward/mc_return statistics before uploading.

Usage:
    python upload_to_hf.py [--dataset_path PATH] [--repo_id REPO_ID] [--private]

Requirements:
    huggingface-cli login  (or HF_TOKEN env var)
    huggingface_hub >= 0.22 (upload_large_folder)

HuggingFace upload strategy:
    upload_large_folder() — chunked parallel upload, auto-resumes on failure,
    no size limit, uses HF LFS for large parquets. Fastest for 200GB+.
"""

import argparse
import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH = (
    "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/"
    "insert-mouse-battery_annotated_v2"
)
DEFAULT_REPO_ID = "Gwanwoo/insert-mouse-battery_annotated_v2"


# ---------------------------------------------------------------------------
# Stats recomputation
# ---------------------------------------------------------------------------
def recompute_reward_stats(dataset_path: Path) -> dict:
    """Compute mean/std/min/max of reward and mc_return over all parquet files."""
    files = sorted((dataset_path / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_path}/data")

    rew_all, mc_all = [], []
    print(f"  recomputing stats from {len(files)} file(s)...", end=" ", flush=True)
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["reward", "mc_return"])
            rew_all.append(np.asarray(t["reward"].to_pylist(), dtype=np.float32))
            mc_all.append(np.asarray(t["mc_return"].to_pylist(), dtype=np.float32))

    rew = np.concatenate(rew_all)
    mc  = np.concatenate(mc_all)
    print("done")

    def _stats(arr):
        return {
            "mean": [float(arr.mean())],
            "std":  [float(arr.std())],
            "min":  [float(arr.min())],
            "max":  [float(arr.max())],
        }

    print(f"  reward   : mean={rew.mean():.6f}  range=[{rew.min():.4f}, {rew.max():.4f}]")
    print(f"  mc_return: mean={mc.mean():.4f}   range=[{mc.min():.4f}, {mc.max():.4f}]")
    return {"reward": _stats(rew), "mc_return": _stats(mc)}


def patch_stats_json(dataset_path: Path) -> Path:
    """Load meta/stats.json, patch reward+mc_return entries, write back.

    Returns the path to the (possibly updated) stats.json.
    """
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"  WARNING: {stats_path} not found — skipping stats patch")
        return stats_path

    with open(stats_path) as f:
        stats = json.load(f)

    new_stats = recompute_reward_stats(dataset_path)
    stats["reward"]    = new_stats["reward"]
    stats["mc_return"] = new_stats["mc_return"]

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  patched  : {stats_path}")
    return stats_path


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload(dataset_path: Path, repo_id: str, private: bool):
    api = HfApi()

    # Ensure repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"  repo     : {repo_id} (exists)")
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
        print(f"  repo     : {repo_id} (created, private={private})")

    # Force the requested visibility even if the repo already existed.
    try:
        api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
        print(f"  visibility: {'private' if private else 'public'}")
    except Exception as e:
        print(f"  WARNING: could not set visibility ({e}); set it in the HF UI if needed")

    print(f"  uploading: {dataset_path}  ({_folder_size_gb(dataset_path):.1f} GB)")
    print(f"  target   : https://huggingface.co/datasets/{repo_id}")
    print(f"  note     : upload_large_folder resumes automatically on interruption")
    print()

    api.upload_large_folder(
        folder_path=str(dataset_path),
        repo_id=repo_id,
        repo_type="dataset",
        # ignore videos (huge, not used by our loader) — comment out to include.
        # also skip the local HF download cache (incomplete stubs from the original pull).
        ignore_patterns=["videos/**", ".cache/**", "**/.cache/**"],
    )
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")


def _folder_size_gb(path: Path) -> float:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(path)
        for f in fs
    )
    return total / 1e9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload annotated dataset to HuggingFace")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH,
                        help="path to the annotated_v2 dataset root")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID,
                        help="HuggingFace repo id (user/name)")
    parser.add_argument("--public", action="store_true",
                        help="create/keep the repo public (default: private)")
    parser.add_argument("--skip_stats", action="store_true",
                        help="skip recomputing stats.json (use if stats already patched)")
    parser.add_argument("--stats_only", action="store_true",
                        help="only recompute + patch stats.json, do not upload")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run reward_annotate.py first."
        )

    private = not args.public
    print(f"=== upload_to_hf ===")
    print(f"dataset  : {dataset_path}")
    print(f"repo     : {args.repo_id}  (private={private})")
    print()

    if not args.skip_stats:
        print("[1/2] patching meta/stats.json ...")
        patch_stats_json(dataset_path)
        print()

    if args.stats_only:
        print("--stats_only: done (no upload)")
        return

    print("[2/2] uploading to HuggingFace ...")
    upload(dataset_path, args.repo_id, private)


if __name__ == "__main__":
    main()
