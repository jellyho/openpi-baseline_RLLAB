"""
Visualize critic quality on a LeRobot trajectory.

For one episode, renders an mp4 with the camera view on the left and, on the
right, the critic's per-step predicted value E[V] (from the dataset's
ground-truth action chunk) overlaid with the ground-truth Monte-Carlo return.
A moving marker tracks the current frame.

    E[V](s_t) = mean over the action horizon of the C51 expected value
                given the GT action chunk a_{t:t+H}.
    GT        = mc_return[t]  (already normalized to [-1, 0] in the dataset).

Usage:
    uv run scripts/visualize_critic.py \\
        --config   pi05_alphaflow_critic_tabletop \\
        --checkpoint checkpoints/pi05_alphaflow_critic_tabletop/<run>/<step> \\
        --repo-id  jellyho/aloha_handover_box_joint_pos_rl_mc \\
        --episode  0 \\
        --output   data/critic_vis/ep0.mp4
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.training.checkpoints as _checkpoints
import openpi.shared.download as download


def build_input_transform(train_config, checkpoint_dir):
    """Same input pipeline as create_trained_policy (repack → ... → model transforms)."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return _transforms.compose([
        *data_config.repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]), data_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Train config name (e.g. pi05_alphaflow_critic_tabletop).")
    p.add_argument("--checkpoint", required=True, help="Checkpoint dir containing params/ and assets/.")
    p.add_argument("--repo-id", required=True, help="LeRobot dataset repo id (with mc_return).")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", required=True, help="Output mp4 path.")
    p.add_argument("--cam-key", default="observation.images.back", help="Camera key to display.")
    p.add_argument("--batch", type=int, default=16, help="Frames per critic forward batch.")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--local-root", default=None)
    args = p.parse_args()

    train_config = _config.get_config(args.config)
    ckpt_dir = download.maybe_download(str(args.checkpoint))
    action_horizon = train_config.model.action_horizon

    # ── Load model with trained params ─────────────────────────────────────────
    print("Loading model...")
    model = train_config.model.load(_model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16))
    model.eval()
    if not hasattr(model, "predict_value"):
        raise ValueError(f"Model {type(model).__name__} has no predict_value (not a critic model).")

    input_transform, _ = build_input_transform(train_config, ckpt_dir)

    # ── Load the episode (with action chunks via delta_timestamps) ─────────────
    print(f"Loading dataset {args.repo_id}, episode {args.episode}...")
    meta = LeRobotDataset(args.repo_id, root=args.local_root).meta
    fps = meta.fps
    ds = LeRobotDataset(
        args.repo_id,
        root=args.local_root,
        delta_timestamps={"action": [t / fps for t in range(action_horizon)]},
    )
    # Frame indices for the requested episode.
    from_idx = ds.episode_data_index["from"][args.episode].item()
    to_idx   = ds.episode_data_index["to"][args.episode].item()
    n_frames = to_idx - from_idx
    print(f"Episode {args.episode}: {n_frames} frames")

    # ── Iterate frames, compute E[V] and collect camera images ─────────────────
    value_fn = nnx_utils.module_jit(model.predict_value)   # (obs, actions) -> [b, H]

    pred_values, gt_returns, cam_frames = [], [], []
    buf_obs, buf_act = [], []

    def flush():
        if not buf_obs:
            return
        obs = jax.tree.map(lambda *xs: jnp.stack(xs, 0), *buf_obs)   # batched Observation
        act = jnp.stack(buf_act, 0)                                 # [b, H, ad]
        v = np.asarray(value_fn(obs, act)).mean(axis=-1)            # mean over horizon → [b]
        pred_values.extend(v.tolist())
        buf_obs.clear(); buf_act.clear()

    for i in range(n_frames):
        sample = ds[from_idx + i]
        raw = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sample.items()}

        gt_returns.append(float(np.asarray(sample["mc_return"]).reshape(-1)[0]))

        # camera frame for display (CHW float [0,1] or HWC uint8 → HWC uint8)
        img = np.asarray(raw[args.cam_key])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cam_frames.append(img)

        # transform → model inputs (observation + normalized actions)
        out = input_transform(raw)
        obs_dict = {"image": out["image"], "image_mask": out["image_mask"], "state": out["state"]}
        if out.get("tokenized_prompt") is not None:
            obs_dict["tokenized_prompt"] = out["tokenized_prompt"]
            obs_dict["tokenized_prompt_mask"] = out["tokenized_prompt_mask"]
        buf_obs.append(_model.Observation.from_dict(obs_dict))
        buf_act.append(jnp.asarray(out["actions"]))

        if len(buf_obs) >= args.batch:
            flush()
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{n_frames}")
    flush()

    pred_values = np.array(pred_values)
    gt_returns  = np.array(gt_returns)
    print(f"E[V]: [{pred_values.min():.3f}, {pred_values.max():.3f}]  "
          f"GT: [{gt_returns.min():.3f}, {gt_returns.max():.3f}]  "
          f"MAE={np.mean(np.abs(pred_values - gt_returns)):.3f}")

    # ── Render mp4: camera | value plot (moving marker) ────────────────────────
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps)
    x = np.arange(n_frames)
    ymin = min(pred_values.min(), gt_returns.min()) - 0.05
    ymax = max(pred_values.max(), gt_returns.max(), 0.0) + 0.05

    for t in range(n_frames):
        fig, (ax_img, ax_v) = plt.subplots(1, 2, figsize=(11, 4.5),
                                           gridspec_kw={"width_ratios": [1, 1.3]})
        ax_img.imshow(cam_frames[t]); ax_img.axis("off")
        ax_img.set_title(f"frame {t}/{n_frames-1}")

        ax_v.plot(x, gt_returns, color="#888", lw=2, label="GT MC return")
        ax_v.plot(x[:t+1], pred_values[:t+1], color="#2e86de", lw=2, label="critic E[V]")
        ax_v.scatter([t], [pred_values[t]], color="#2e86de", zorder=5)
        ax_v.axvline(t, color="#bbb", ls="--", lw=1)
        ax_v.set_xlim(0, n_frames-1); ax_v.set_ylim(ymin, ymax)
        ax_v.set_xlabel("frame"); ax_v.set_ylabel("value")
        ax_v.set_title(f"E[V]={pred_values[t]:.3f}  GT={gt_returns[t]:.3f}")
        ax_v.legend(loc="lower right", fontsize=9); ax_v.grid(alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame)
        plt.close(fig)

    writer.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
