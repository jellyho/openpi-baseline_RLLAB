"""
Visualize the **AQC prefix-critic** on an annotated LeRobot trajectory.

Unlike the old ``visualize_critic.py`` (which ran the whole VLA critic), this reads
the PRECOMPUTED ``rl_token`` (z_rl) and ``base_action`` (N candidate chunks) columns
from an RLT-annotated dataset, so it only needs the small AQC critic — NO VLA forward.

For one episode it renders an mp4: camera view on the left, and on the right:
  (top)  value curves over the episode vs the ground-truth MC return:
           • V_gt        — critic value of the DEMO action chunk (full horizon).
                            Calibration check: should track GT mc_return.
           • V_adaptive  — max over the N base-action candidates AND macro-prefixes
                            of Q(z_rl, a^(n)_{1:h}); the AQC policy's value at s.
  (bot)  h* — the adaptive commit length (k*+1)*macro_group_size chosen by the
           joint arg-max over (candidate, prefix).  Shows how Q-chunking adapts.

Both critic values and mc_return live on the same normalized [v_min, v_max] scale.

Usage:
    uv run scripts/visualize_aqc_critic.py \\
        --critic-run-dir /data5/.../insert-mouse-battery_a201_..._s0 \\
        --dataset jellyho/insert-mouse-battery_annotated \\
        --episode 0 \\
        --output data/critic_vis/aqc_imb_ep0.mp4
"""

import argparse
import json
import pathlib

import flax.serialization as fs
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from openpi.shared.lerobot_compat import LeRobotDataset  # v2.1/v3.0-tolerant
from openpi.rlt_critic.merge import _net_hparams_from_config_dict
from openpi.rlt_critic.transformer import PrefixValue
from openpi.rlt_critic.distributional import hl_gauss_transform


def _resolve_step(run_dir: pathlib.Path, step: str) -> pathlib.Path:
    ckpts = run_dir / "checkpoints"
    avail = sorted(int(p.name.split("_")[1]) for p in ckpts.glob("step_*") if p.is_dir())
    if not avail:
        raise FileNotFoundError(f"no step_* checkpoints under {ckpts}")
    s = avail[-1] if step in ("latest", "", None) else int(step)
    if s not in avail:
        raise FileNotFoundError(f"step {s} not in {avail}")
    return ckpts / f"step_{s:08d}"


def load_critic(run_dir: pathlib.Path, step: str):
    """Load the AQC prefix-critic from a training run dir (config.json + step params).

    Returns (critic_fn, net_hp) where critic_fn(states[M,L], acts[M,H*Dr]) -> Q[M, macro_H]
    (ensemble-min of the distributional expected value).
    """
    cd = json.loads((run_dir / "config.json").read_text())
    hp = _net_hparams_from_config_dict(cd)
    net = PrefixValue(
        action_dim=hp["action_dim"], horizon=hp["horizon"],
        num_ensembles=hp["num_ensembles"], num_layers=hp["num_layers"],
        num_heads=hp["num_heads"], head_dim=hp["head_dim"], mlp_dim=hp["mlp_dim"],
        layer_norm=hp["layer_norm"], num_atoms=hp["num_atoms"],
        per_position_head=hp["per_position_head"],
        state_encoder_dims=tuple(hp["state_encoder_dims"]),
        macro_group_size=hp["macro_group_size"],
    )
    ex_obs = jnp.zeros((1, hp["latent_dim"]))
    ex_act = jnp.zeros((1, hp["horizon"] * hp["action_dim"]))
    template = net.init(jax.random.PRNGKey(0), ex_obs, ex_act)
    step_dir = _resolve_step(run_dir, step)
    params = fs.from_bytes(template, (step_dir / "params.msgpack").read_bytes())
    sigma = hp["hl_gauss_sigma_frac"] * (hp["v_max"] - hp["v_min"]) / hp["num_atoms"]
    _, from_probs = hl_gauss_transform(hp["v_min"], hp["v_max"], hp["num_atoms"], sigma)

    @jax.jit
    def critic_fn(states, acts):
        logits = net.apply(params, states, acts)          # (K, M, macro_H, atoms)
        q = from_probs(jax.nn.softmax(logits, axis=-1))    # (K, M, macro_H)
        return q.min(axis=0)                               # (M, macro_H) ensemble-min
    return critic_fn, hp, int(step_dir.name.split("_")[1])


def _episode_range(ds, episode: int):
    if hasattr(ds, "episode_data_index"):  # lerobot v2.1
        return int(ds.episode_data_index["from"][episode].item()), int(ds.episode_data_index["to"][episode].item())
    eps = ds.meta.episodes  # v3.0
    row = eps.iloc[episode] if hasattr(eps, "iloc") else eps[episode]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def _to_np(v):
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic-run-dir", required=True, help="AQC critic run dir (config.json + checkpoints/).")
    p.add_argument("--critic-step", default="latest", help="step number or 'latest'.")
    p.add_argument("--dataset", required=True, help="RLT-annotated LeRobot repo id (has rl_token + base_action).")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", required=True, help="Output mp4 path.")
    p.add_argument("--cam-key", default="observation.images.cam_high", help="Camera key to display.")
    p.add_argument("--rl-token-key", default="rl_token")
    p.add_argument("--base-action-key", default="base_action")
    p.add_argument("--num-candidates", type=int, default=0, help="Use first N base-action candidates (0 = all).")
    p.add_argument("--frame-batch", type=int, default=64, help="Frames per critic forward (memory).")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--local-root", default=None)
    args = p.parse_args()

    run_dir = pathlib.Path(args.critic_run_dir).resolve()
    # tolerate pointing at the inner checkpoints/ dir.
    if not (run_dir / "config.json").exists() and (run_dir.parent / "config.json").exists():
        run_dir = run_dir.parent

    print("Loading AQC critic...")
    critic_fn, hp, step = load_critic(run_dir, args.critic_step)
    H, Dr = hp["horizon"], hp["action_dim"]
    macro_g = hp["macro_group_size"]
    macro_H = H // macro_g
    print(f"  critic step {step} | latent_dim={hp['latent_dim']} H={H} Dr={Dr} "
          f"macro_group={macro_g} macro_H={macro_H} K={hp['num_ensembles']} atoms={hp['num_atoms']}")

    # ── Load the episode (GT action chunk via delta_timestamps; rl_token/base_action per-frame) ──
    print(f"Loading dataset {args.dataset}, episode {args.episode}...")
    fps = LeRobotDataset(args.dataset, root=args.local_root).meta.fps
    ds = LeRobotDataset(args.dataset, root=args.local_root,
                        delta_timestamps={"action": [t / fps for t in range(H)]})
    a, b = _episode_range(ds, args.episode)
    n_frames = b - a
    print(f"Episode {args.episode}: {n_frames} frames")

    # ── Gather per-frame inputs ────────────────────────────────────────────────
    z_list, base_list, gt_list, mc_list, cam_frames = [], [], [], [], []
    for i in range(n_frames):
        s = ds[a + i]
        z_list.append(_to_np(s[args.rl_token_key]).astype(np.float32).reshape(-1))          # [L]
        ba = _to_np(s[args.base_action_key]).astype(np.float32)                              # [N, H, Dr]
        base_list.append(ba)
        gt_list.append(_to_np(s["action"]).astype(np.float32).reshape(H, -1)[:, :Dr])        # [H, Dr]
        mc_list.append(float(np.asarray(_to_np(s["mc_return"])).reshape(-1)[0]))
        img = _to_np(s[args.cam_key])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cam_frames.append(img)
        if (i + 1) % 100 == 0:
            print(f"  read {i+1}/{n_frames}")

    z = np.stack(z_list)                        # [T, L]
    base = np.stack(base_list)                  # [T, N, H, Dr]
    gt = np.stack(gt_list)                       # [T, H, Dr]
    mc = np.array(mc_list)                       # [T]
    T, N = base.shape[0], base.shape[1]
    if args.num_candidates and args.num_candidates < N:
        base = base[:, : args.num_candidates]
        N = args.num_candidates
    print(f"scoring {T} frames x {N} candidates ...")

    # ── Critic forward (batched over frames) ───────────────────────────────────
    V_gt = np.empty(T, np.float32)              # critic value of the demo action (full prefix)
    V_adapt = np.empty(T, np.float32)           # best over candidates & prefixes (policy value)
    h_star = np.empty(T, np.int32)              # chosen commit length
    n_star = np.empty(T, np.int32)
    for s0 in range(0, T, args.frame_batch):
        s1 = min(s0 + args.frame_batch, T)
        bsz = s1 - s0
        # GT: one chunk per frame.
        q_gt = np.asarray(critic_fn(jnp.asarray(z[s0:s1]),
                                    jnp.asarray(gt[s0:s1].reshape(bsz, -1))))       # [bsz, macro_H]
        V_gt[s0:s1] = q_gt[:, -1]                                                   # full prefix
        # Candidates: N chunks per frame.
        st = np.repeat(z[s0:s1], N, axis=0)                                         # [bsz*N, L]
        ac = base[s0:s1].reshape(bsz * N, -1)                                       # [bsz*N, H*Dr]
        q_c = np.asarray(critic_fn(jnp.asarray(st), jnp.asarray(ac)))               # [bsz*N, macro_H]
        q_c = q_c.reshape(bsz, N, macro_H)                                          # [bsz, N, macro_H]
        flat = q_c.reshape(bsz, -1).argmax(axis=1)
        ns, mk = np.divmod(flat, macro_H)
        V_adapt[s0:s1] = q_c.reshape(bsz, -1).max(axis=1)
        h_star[s0:s1] = (mk + 1) * macro_g
        n_star[s0:s1] = ns

    print(f"V_gt:[{V_gt.min():.3f},{V_gt.max():.3f}] MAE_vs_mc={np.mean(np.abs(V_gt-mc)):.3f} | "
          f"V_adapt:[{V_adapt.min():.3f},{V_adapt.max():.3f}] | "
          f"h*: mean={h_star.mean():.1f} (range {h_star.min()}..{h_star.max()})")

    # ── Render mp4: camera | [value curves ; h*] ───────────────────────────────
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps)
    x = np.arange(T)
    vmin = min(mc.min(), V_gt.min(), V_adapt.min()) - 0.05
    vmax = max(mc.max(), V_gt.max(), V_adapt.max(), 0.0) + 0.05

    for t in range(T):
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[:, 0])
        ax_v = fig.add_subplot(gs[0, 1])
        ax_h = fig.add_subplot(gs[1, 1])

        ax_img.imshow(cam_frames[t]); ax_img.axis("off")
        ax_img.set_title(f"frame {t}/{T-1}")

        ax_v.plot(x, mc, color="#888", lw=2, label="GT MC return")
        ax_v.plot(x[:t+1], V_gt[:t+1], color="#2e86de", lw=2, label="critic V (demo action)")
        ax_v.plot(x[:t+1], V_adapt[:t+1], color="#27ae60", lw=2, label="V adaptive (best cand.)")
        ax_v.scatter([t], [V_adapt[t]], color="#27ae60", zorder=5, s=20)
        ax_v.axvline(t, color="#ccc", ls="--", lw=1)
        ax_v.set_xlim(0, T-1); ax_v.set_ylim(vmin, vmax)
        ax_v.set_ylabel("value"); ax_v.grid(alpha=0.3)
        ax_v.legend(loc="lower right", fontsize=8)
        ax_v.set_title(f"V_demo={V_gt[t]:.3f}  V_adapt={V_adapt[t]:.3f}  GT={mc[t]:.3f}")

        ax_h.step(x[:t+1], h_star[:t+1], where="post", color="#e67e22", lw=2)
        ax_h.axvline(t, color="#ccc", ls="--", lw=1)
        ax_h.set_xlim(0, T-1); ax_h.set_ylim(0, H + macro_g * 0.5)
        ax_h.set_xlabel("frame"); ax_h.set_ylabel("h* (commit)")
        ax_h.grid(alpha=0.3)
        ax_h.set_title(f"h*={h_star[t]}  (n*={n_star[t]})", fontsize=9)

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
