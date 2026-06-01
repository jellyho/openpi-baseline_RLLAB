"""
Add a Monte-Carlo return column to a LeRobot dataset (v2.1 layout).

Reward structure (LPS-RFT / RECAP):
    r_t = 0       if t = T and episode succeeded
    r_t = -C_fail if t = T and episode failed
    r_t = -1      otherwise
    G_t = sum_{k>=t} gamma^{k-t} r_k

C_fail defaults to 0.5 * max_episode_length (≈ half the task horizon).
Returns are optionally normalized to (-1, 0) by dividing by max_episode_length.

This script preserves the full LeRobot structure (per-episode parquet under
data/chunk-*/, videos/, and all meta files), only:
  1. adding an `mc_return` column to each episode parquet, and
  2. registering `mc_return` in meta/info.json features.

Usage
─────
    uv run scripts/compute_mc_returns.py \\
        --repo-id jellyho/aloha_handover_box_joint_pos_rl \\
        --output  data/aloha_handover_box_joint_pos_rl_mc \\
        --gamma   0.995
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


def episode_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted MC return for a single episode (rewards already in t order)."""
    T = len(rewards)
    g = np.zeros(T, dtype=np.float32)
    g[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        g[t] = rewards[t] + gamma * g[t + 1]
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--cfail", type=float, default=None, help="Failure penalty (default: 0.5 * max_episode_length).")
    p.add_argument("--normalize", action="store_true", default=True, help="Normalize to (-1, 0) per task.")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--reward-key", default="next.reward")
    p.add_argument("--success-key", default="next.success")
    p.add_argument("--local-root", default=None, help="Local source dataset root (skip HF download).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = pathlib.Path(args.output)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists. Pass --overwrite.")
        shutil.rmtree(out)

    # ── Locate source dataset (full v2.1 structure) ───────────────────────────
    src = pathlib.Path(args.local_root) if args.local_root else pathlib.Path(
        snapshot_download(args.repo_id, repo_type="dataset")
    )
    print(f"Source: {src}")

    # Copy the entire dataset (videos, meta, data) → output.
    print(f"Copying full dataset → {out}")
    shutil.copytree(src, out)

    # ── Discover episode parquets ─────────────────────────────────────────────
    info = json.loads((out / "meta" / "info.json").read_text())
    data_path_tmpl = info["data_path"]   # data/chunk-{...}/episode_{...}.parquet
    chunks_size = info["chunks_size"]
    n_episodes = info["total_episodes"]

    ep_paths = []
    for ep in range(n_episodes):
        rel = data_path_tmpl.format(episode_chunk=ep // chunks_size, episode_index=ep)
        ep_paths.append(out / rel)

    # ── Pass 1: episode lengths → C_fail, normalization factor ────────────────
    lengths = [len(pd.read_parquet(pth, columns=[args.reward_key])) for pth in ep_paths]
    max_len = max(lengths)
    cfail = args.cfail if args.cfail is not None else 0.5 * max_len
    print(f"Episodes: {n_episodes} | max_len: {max_len} | C_fail: {cfail:.1f} | gamma: {args.gamma}")

    # ── Pass 2: compute mc_return per episode, add column, write back ──────────
    all_returns, all_success = [], []
    for pth in ep_paths:
        df = pd.read_parquet(pth)
        rewards = df[args.reward_key].to_numpy(dtype=np.float32).copy()
        success = bool(np.asarray(df[args.success_key]).any())  # episode-level

        # Rebuild reward: -1 per step, terminal 0 (success) or -C_fail (failure).
        rewards[:] = -1.0
        rewards[-1] = 0.0 if success else -cfail

        g = episode_mc_returns(rewards, args.gamma)
        if args.normalize:
            g = np.clip(g / max_len, -1.0, 0.0)

        df["mc_return"] = g.astype(np.float32)
        df.to_parquet(pth)
        all_returns.append(g)
        all_success.append(np.full(len(g), success))

    rets = np.concatenate(all_returns)
    sus  = np.concatenate(all_success)
    print(f"mc_return: min={rets.min():.3f} max={rets.max():.3f} mean={rets.mean():.3f}")
    print(f"  success frames mean={rets[sus].mean():.3f} | failure frames mean={rets[~sus].mean():.3f}"
          f" | separation={rets[sus].mean() - rets[~sus].mean():.3f}")

    # ── Register mc_return in info.json features ───────────────────────────────
    info["features"]["mc_return"] = {"dtype": "float32", "shape": [1], "names": None}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"Done. Wrote {out} (added 'mc_return' to {n_episodes} episode parquets + info.json).")


if __name__ == "__main__":
    main()
