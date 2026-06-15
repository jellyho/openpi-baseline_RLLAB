"""
LIVE AQC-critic visualization — run the ORIGINAL LeRobot video through the MERGED bundle.

Nothing is precomputed: for each frame of a real (video) episode we build the model
Observation from the actual images/state/prompt, run the bundle's RLT model to get the
RL token z_rl + N base-action chunks, decode to raw action space, and query the prefix
critic Q(z_rl, a^(n)_{1:h}).  This matches the deployment pipeline exactly and avoids any
precomputed-column provenance mismatch.

Renders an mp4: camera (real video) on the left, and on the right:
  (top)  value curves vs GT MC return (if present):
           • V_demo     — critic value of the dataset's demo action chunk (full horizon).
           • V_adaptive — max over the N candidates AND macro-prefixes (the AQC value).
           • the FULL distribution of the N candidate values: min–max + IQR bands, median,
             and the N points scattered at the current frame.
  (bot)  h* — the adaptive commit length.
  teleop (human-in-control) frames are shaded.

NEEDS A GPU (VLA backbone + action-expert sampling per frame).  Run via srun on node01:
    srun -w node01 --gres=gpu:1 uv run scripts/visualize_aqc_live.py \\
        --bundle /data5/.../acrft/pi05_insert-mouse-battery_rlt_aqc_b2048 \\
        --dataset jellyho/insert-mouse-battery_annotated \\
        --local-root /data5/gwanwoo/rss_pft/phase1/insert-mouse-battery_annotated \\
        --episode 0 --stride 2 --output data/critic_vis/live_imb_ep0.mp4
"""

import argparse
import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.checkpoints as _checkpoints
import openpi.transforms as _transforms
from openpi.rlt_critic.inference import AQCAdaptive
from openpi.shared.lerobot_compat import LeRobotDataset


def load_bundle(bundle_dir):
    """Load the merged bundle (RLT model + prefix critic) + the raw->model input transform."""
    bundle = pathlib.Path(bundle_dir).resolve()
    manifest = json.loads((bundle / "aqc_manifest.json").read_text())
    tcfg = _config.get_config(manifest["rlt_config_name"])
    data_config = tcfg.data.create(tcfg.assets_dirs, tcfg.model)

    # norm stats — same search order as inference.create_aqc_policy.
    rlt_params = pathlib.Path(manifest["rlt_params"]).resolve()
    ns = None
    for base in (bundle / "params", rlt_params, rlt_params.parent):
        try:
            ns = _checkpoints.load_norm_stats(base, data_config.asset_id); break
        except FileNotFoundError:
            continue
    if ns is None:
        raise FileNotFoundError(f"norm stats not found for {bundle} (asset_id={data_config.asset_id})")
    data_config = dataclasses.replace(data_config, norm_stats=ns)

    ada = AQCAdaptive.load(bundle, data_config=data_config)
    # full pipeline: raw LeRobot frame -> repack -> data -> normalize -> model transforms.
    transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(ns, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    return ada, transform, manifest


def _to_np(v):
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def _cam_to_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="merged AQC bundle dir (from merge).")
    p.add_argument("--dataset", required=True, help="ORIGINAL LeRobot repo id (with videos).")
    p.add_argument("--local-root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--cam-key", default="observation.images.cam_high")
    p.add_argument("--num-samples", type=int, default=0, help="base-action candidates (0 = bundle default).")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame (faster).")
    p.add_argument("--max-frames", type=int, default=0, help="cap processed frames (0 = all).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-mc", action="store_true",
                   help="overlay the dataset GT mc_return (OFF by default: its reward scheme may "
                        "differ from the critic's training scale, e.g. after stage3b reward re-annotation).")
    args = p.parse_args()

    print("Loading bundle (RLT model + critic)...")
    ada, transform, manifest = load_bundle(args.bundle)
    H, Dr, macro_g = ada.horizon, ada.action_dim, ada.macro_group_size
    macro_H = H // macro_g
    N = int(args.num_samples or ada.num_action_samples)
    print(f"  rlt_config={manifest['rlt_config_name']} | critic_step={manifest['critic_step']} | "
          f"H={H} Dr={Dr} macro_g={macro_g} macro_H={macro_H} N={N}")

    print(f"Loading dataset {args.dataset} ep {args.episode} (with videos)...")
    # video_backend="pyav": decode via the venv's bundled FFmpeg (pyav), so it works on nodes
    # that lack a system libtorchcodec/FFmpeg (e.g. node01).
    fps = LeRobotDataset(args.dataset, root=args.local_root).meta.fps
    ds = LeRobotDataset(args.dataset, root=args.local_root, video_backend="pyav",
                        delta_timestamps={"action": [t / fps for t in range(H)]})
    if hasattr(ds, "episode_data_index"):
        a = int(ds.episode_data_index["from"][args.episode].item()); b = int(ds.episode_data_index["to"][args.episode].item())
    else:
        eps = ds.meta.episodes
        row = eps.iloc[args.episode] if hasattr(eps, "iloc") else eps[args.episode]
        a, b = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    idxs = list(range(a, b, max(1, args.stride)))
    if args.max_frames:
        idxs = idxs[: args.max_frames]
    T = len(idxs)
    print(f"  episode {args.episode}: {b-a} frames -> processing {T} (stride {args.stride})")

    rng = jax.random.key(args.seed)
    V_gt = np.full(T, np.nan, np.float32); V_adapt = np.empty(T, np.float32)
    q_cand = np.empty((T, N), np.float32); h_star = np.empty(T, np.int32); n_star = np.empty(T, np.int32)
    mc = np.full(T, np.nan, np.float32); cam = []; teleop = np.zeros(T, bool)
    has_gt = has_mc = False

    for k, i in enumerate(idxs):
        s = ds[i]
        raw = {kk: (_to_np(v) if hasattr(v, "numpy") else v) for kk, v in s.items()}
        # camera (real video frame)
        cam.append(_cam_to_uint8(raw[args.cam_key]))
        if "observation.commander_state" in s:
            teleop[k] = str(np.asarray(_to_np(s["observation.commander_state"])).reshape(-1)[0]) == "teleop"
        if "mc_return" in s:
            mc[k] = float(np.asarray(_to_np(s["mc_return"])).reshape(-1)[0]); has_mc = True

        # raw frame -> model Observation (batch dim 1)
        out = transform(raw)
        obs_d = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], {
            "image": out["image"], "image_mask": out["image_mask"], "state": out["state"],
            **({"tokenized_prompt": out["tokenized_prompt"],
                "tokenized_prompt_mask": out["tokenized_prompt_mask"]}
               if out.get("tokenized_prompt") is not None else {}),
        })
        observation = _model.Observation.from_dict(obs_d)

        # RLT model -> z_rl [1,L] + base [1,N,H,Dm]; decode -> raw [N,H,Dr]; critic -> q [N, macro_H]
        rng, r = jax.random.split(rng)
        z_rl, base = ada._propose(r, observation, N)
        base_raw = ada.decode(np.asarray(base), np.asarray(observation.state))[0]   # [N,H,Dr]
        q = ada._score(z_rl, base_raw)                                              # [N, macro_H]
        q_cand[k] = q.max(axis=1)
        flat = int(np.argmax(q)); ns_, mk = divmod(flat, macro_H)
        V_adapt[k] = q.max(); h_star[k] = (mk + 1) * macro_g; n_star[k] = ns_

        # critic value of the dataset's demo action chunk (raw 14-dim), for calibration vs mc_return.
        gt = raw.get("action")
        if gt is not None:
            gt = np.asarray(gt).reshape(H, -1)[:, :Dr][None]                        # [1,H,Dr]
            V_gt[k] = float(ada._score(z_rl, gt)[0, -1]); has_gt = True
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{T}")

    qc_lo, qc_hi = q_cand.min(1), q_cand.max(1)
    qc_p25, qc_p75 = np.percentile(q_cand, [25, 75], axis=1)
    qc_med = np.median(q_cand, axis=1)
    msg = f"V_adapt:[{V_adapt.min():.3f},{V_adapt.max():.3f}] cand=[{qc_lo.mean():.3f}..{qc_hi.mean():.3f}] h*mean={h_star.mean():.1f}"
    if has_gt and has_mc:
        ok = ~np.isnan(mc)
        msg += f" | V_gt MAE_vs_mc={np.mean(np.abs(V_gt[ok]-mc[ok])):.3f}"
    print(msg)

    # ── Render ─────────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps)
    x = np.arange(T)
    lo_all = [qc_lo.min(), V_adapt.min()] + ([np.nanmin(mc)] if has_mc else []) + ([np.nanmin(V_gt)] if has_gt else [])
    hi_all = [qc_hi.max(), V_adapt.max(), 0.0] + ([np.nanmax(mc)] if has_mc else [])
    vmin, vmax = min(lo_all) - 0.05, max(hi_all) + 0.05

    def shade(ax):
        i = 0
        while i < T:
            if teleop[i]:
                j = i
                while j < T and teleop[j]:
                    j += 1
                ax.axvspan(i, j - 1, color="#e74c3c", alpha=0.12, lw=0); i = j
            else:
                i += 1

    for t in range(T):
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[:, 0]); ax_v = fig.add_subplot(gs[0, 1]); ax_h = fig.add_subplot(gs[1, 1])

        ax_img.imshow(cam[t]); ax_img.axis("off")
        ctrl = "HUMAN (teleop)" if teleop[t] else "policy"
        ax_img.set_title(f"frame {t}/{T-1}  [{ctrl}]", color=("#e74c3c" if teleop[t] else "black"))

        shade(ax_v)
        if has_mc:
            ax_v.plot(x, mc, color="#888", lw=2, label="GT MC return")
        ax_v.fill_between(x, qc_lo, qc_hi, color="#27ae60", alpha=0.10, lw=0, label=f"{N} cand. spread")
        ax_v.fill_between(x, qc_p25, qc_p75, color="#27ae60", alpha=0.22, lw=0, label="cand. IQR")
        ax_v.plot(x, qc_med, color="#27ae60", lw=1, ls=":", alpha=0.7, label="cand. median")
        if has_gt:
            ax_v.plot(x[:t+1], V_gt[:t+1], color="#2e86de", lw=2, label="critic V (demo)")
        ax_v.plot(x[:t+1], V_adapt[:t+1], color="#1e8449", lw=2, label="V adaptive (max)")
        ax_v.scatter(np.full(N, t), q_cand[t], color="#27ae60", s=10, alpha=0.55, zorder=4, edgecolors="none")
        ax_v.scatter([t], [V_adapt[t]], color="#1e8449", zorder=5, s=24)
        ax_v.axvline(t, color="#ccc", ls="--", lw=1)
        ax_v.set_xlim(0, T-1); ax_v.set_ylim(vmin, vmax); ax_v.set_ylabel("value"); ax_v.grid(alpha=0.3)
        ax_v.legend(loc="lower right", fontsize=7, ncol=2)
        ax_v.set_title(f"V_adapt={V_adapt[t]:.3f}  cand=[{q_cand[t].min():.3f}, {q_cand[t].max():.3f}]"
                       + (f"  GT={mc[t]:.3f}" if has_mc and not np.isnan(mc[t]) else ""))

        shade(ax_h)
        ax_h.step(x[:t+1], h_star[:t+1], where="post", color="#e67e22", lw=2)
        ax_h.axvline(t, color="#ccc", ls="--", lw=1)
        ax_h.set_xlim(0, T-1); ax_h.set_ylim(0, H + macro_g * 0.5)
        ax_h.set_xlabel("frame"); ax_h.set_ylabel("h* (commit)"); ax_h.grid(alpha=0.3)
        ax_h.set_title(f"h*={h_star[t]}  (n*={n_star[t]})", fontsize=9)

        fig.tight_layout(); fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame); plt.close(fig)

    writer.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
