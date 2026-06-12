"""
Visualize the **AQC prefix-critic** on an annotated LeRobot trajectory.

Unlike the old ``visualize_critic.py`` (which ran the whole VLA critic), this reads
the PRECOMPUTED ``rl_token`` (z_rl) and ``base_action`` (N candidate chunks) columns,
so it only needs the small AQC critic — NO VLA forward.

Two input modes:
  • ``--episode-npz FILE``  — a dump from ``dump_episodes.py`` (FAST: no LeRobotDataset
                              load / video decode).  RECOMMENDED for repeated testing.
  • ``--dataset REPO --episode I`` — read straight from the LeRobot dataset (slow).

Renders an mp4: camera on the left, and on the right:
  (top)  value curves over the episode vs the ground-truth MC return:
           • V_demo      — critic value of the DEMO action chunk (full horizon).
                            Calibration: should track GT mc_return.
           • V_adaptive  — max over the N base-action candidates AND macro-prefixes
                            of Q(z_rl, a^(n)_{1:h}); the AQC policy's value at s.
         Frames where a human is in control (commander_state == 'teleop') are shaded.
  (bot)  h* — the adaptive commit length (k*+1)*macro_group_size from the joint
           arg-max over (candidate, prefix).

Usage:
    uv run scripts/visualize_aqc_critic.py \\
        --critic-run-dir /data5/.../insert-mouse-battery_a201_..._s0 \\
        --episode-npz data/critic_vis/episodes/ep00947_intervention.npz \\
        --output data/critic_vis/aqc_imb_ep947.mp4
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

from openpi.rlt_critic.merge import _net_hparams_from_config_dict
from openpi.rlt_critic.transformer import PrefixValue
from openpi.rlt_critic.distributional import hl_gauss_transform


# --------------------------------------------------------------------------- critic
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


# --------------------------------------------------------------------------- episode I/O
def _to_np(v):
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def _cam_to_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def load_from_npz(path, H, Dr):
    d = np.load(path, allow_pickle=True)
    cam = d["cam"]
    return {
        "z": d["rl_token"].astype(np.float32),
        "base": d["base_action"].astype(np.float32),
        "gt": d["gt_action"].astype(np.float32)[:, :, :Dr],
        "mc": d["mc_return"].astype(np.float32),
        "cam": [cam[i] for i in range(len(cam))],
        "commander": d["commander"] if "commander" in d else None,
        "tag": f"ep{int(d['episode_index'])} [{str(d['category'])}]",
    }


def load_from_dataset(dataset, local_root, episode, cam_key, H, Dr):
    from openpi.shared.lerobot_compat import LeRobotDataset
    fps = LeRobotDataset(dataset, root=local_root).meta.fps
    ds = LeRobotDataset(dataset, root=local_root,
                        delta_timestamps={"action": [t / fps for t in range(H)]})
    if hasattr(ds, "episode_data_index"):
        a = int(ds.episode_data_index["from"][episode].item()); b = int(ds.episode_data_index["to"][episode].item())
    else:
        eps = ds.meta.episodes
        row = eps.iloc[episode] if hasattr(eps, "iloc") else eps[episode]
        a, b = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    z, base, gt, mc, cam, cs = [], [], [], [], [], []
    for i in range(b - a):
        s = ds[a + i]
        z.append(_to_np(s["rl_token"]).astype(np.float32).reshape(-1))
        base.append(_to_np(s["base_action"]).astype(np.float32))
        gt.append(_to_np(s["action"]).astype(np.float32).reshape(H, -1)[:, :Dr])
        mc.append(float(np.asarray(_to_np(s["mc_return"])).reshape(-1)[0]))
        cs.append(str(_to_np(s.get("observation.commander_state", "")).reshape(-1)[0])
                  if "observation.commander_state" in s else "")
        cam.append(_cam_to_uint8(s[cam_key]))
        if (i + 1) % 100 == 0:
            print(f"  read {i+1}/{b-a}")
    return {"z": np.stack(z), "base": np.stack(base), "gt": np.stack(gt),
            "mc": np.array(mc, np.float32), "cam": cam, "commander": np.array(cs),
            "tag": f"ep{episode}"}


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic-run-dir", required=True, help="AQC critic run dir (config.json + checkpoints/).")
    p.add_argument("--critic-step", default="latest")
    p.add_argument("--episode-npz", default=None, help="dump from dump_episodes.py (fast path).")
    p.add_argument("--dataset", default=None, help="annotated LeRobot repo id (slow path).")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--local-root", default=None)
    p.add_argument("--cam-key", default="observation.images.cam_high")
    p.add_argument("--output", required=True, help="Output mp4 path.")
    p.add_argument("--num-candidates", type=int, default=0, help="Use first N base-action candidates (0 = all).")
    p.add_argument("--frame-batch", type=int, default=64)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()
    if not args.episode_npz and not args.dataset:
        p.error("provide --episode-npz (fast) or --dataset (+ --episode).")

    run_dir = pathlib.Path(args.critic_run_dir).resolve()
    if not (run_dir / "config.json").exists() and (run_dir.parent / "config.json").exists():
        run_dir = run_dir.parent

    print("Loading AQC critic...")
    critic_fn, hp, step = load_critic(run_dir, args.critic_step)
    H, Dr, macro_g = hp["horizon"], hp["action_dim"], hp["macro_group_size"]
    macro_H = H // macro_g
    print(f"  critic step {step} | L={hp['latent_dim']} H={H} Dr={Dr} macro_g={macro_g} "
          f"macro_H={macro_H} K={hp['num_ensembles']} atoms={hp['num_atoms']}")

    print("Loading episode...")
    if args.episode_npz:
        ep = load_from_npz(args.episode_npz, H, Dr)
    else:
        ep = load_from_dataset(args.dataset, args.local_root, args.episode, args.cam_key, H, Dr)
    z, base, gt, mc, cam, commander, tag = (
        ep["z"], ep["base"], ep["gt"], ep["mc"], ep["cam"], ep["commander"], ep["tag"])
    T, N = base.shape[0], base.shape[1]
    if args.num_candidates and args.num_candidates < N:
        base, N = base[:, : args.num_candidates], args.num_candidates
    teleop = (np.array([str(c) for c in commander]) == "teleop") if commander is not None else np.zeros(T, bool)
    print(f"{tag}: {T} frames x {N} candidates "
          f"({int(teleop.sum())} human-teleop frames)")

    # ── Critic forward (batched over frames) ───────────────────────────────────
    V_gt = np.empty(T, np.float32); V_adapt = np.empty(T, np.float32)
    h_star = np.empty(T, np.int32); n_star = np.empty(T, np.int32)
    for s0 in range(0, T, args.frame_batch):
        s1 = min(s0 + args.frame_batch, T); bsz = s1 - s0
        q_gt = np.asarray(critic_fn(jnp.asarray(z[s0:s1]), jnp.asarray(gt[s0:s1].reshape(bsz, -1))))
        V_gt[s0:s1] = q_gt[:, -1]
        st = np.repeat(z[s0:s1], N, axis=0)
        ac = base[s0:s1].reshape(bsz * N, -1)
        q_c = np.asarray(critic_fn(jnp.asarray(st), jnp.asarray(ac))).reshape(bsz, N, macro_H)
        flat = q_c.reshape(bsz, -1).argmax(axis=1)
        ns, mk = np.divmod(flat, macro_H)
        V_adapt[s0:s1] = q_c.reshape(bsz, -1).max(axis=1)
        h_star[s0:s1] = (mk + 1) * macro_g; n_star[s0:s1] = ns

    print(f"V_gt:[{V_gt.min():.3f},{V_gt.max():.3f}] MAE_vs_mc={np.mean(np.abs(V_gt-mc)):.3f} | "
          f"V_adapt:[{V_adapt.min():.3f},{V_adapt.max():.3f}] | h*: mean={h_star.mean():.1f}")

    # ── Render mp4 ─────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps)
    x = np.arange(T)
    vmin = min(mc.min(), V_gt.min(), V_adapt.min()) - 0.05
    vmax = max(mc.max(), V_gt.max(), V_adapt.max(), 0.0) + 0.05

    def shade_teleop(ax):
        # contiguous teleop spans → light red bands (human in control).
        i = 0
        while i < T:
            if teleop[i]:
                j = i
                while j < T and teleop[j]:
                    j += 1
                ax.axvspan(i, j - 1, color="#e74c3c", alpha=0.12, lw=0)
                i = j
            else:
                i += 1

    for t in range(T):
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[:, 0]); ax_v = fig.add_subplot(gs[0, 1]); ax_h = fig.add_subplot(gs[1, 1])

        ax_img.imshow(cam[t]); ax_img.axis("off")
        ctrl = "HUMAN (teleop)" if teleop[t] else "policy"
        ax_img.set_title(f"{tag}  frame {t}/{T-1}  [{ctrl}]",
                         color=("#e74c3c" if teleop[t] else "black"))

        shade_teleop(ax_v)
        ax_v.plot(x, mc, color="#888", lw=2, label="GT MC return")
        ax_v.plot(x[:t+1], V_gt[:t+1], color="#2e86de", lw=2, label="critic V (demo action)")
        ax_v.plot(x[:t+1], V_adapt[:t+1], color="#27ae60", lw=2, label="V adaptive (best cand.)")
        ax_v.scatter([t], [V_adapt[t]], color="#27ae60", zorder=5, s=20)
        ax_v.axvline(t, color="#ccc", ls="--", lw=1)
        ax_v.set_xlim(0, T-1); ax_v.set_ylim(vmin, vmax); ax_v.set_ylabel("value"); ax_v.grid(alpha=0.3)
        ax_v.legend(loc="lower right", fontsize=8)
        ax_v.set_title(f"V_demo={V_gt[t]:.3f}  V_adapt={V_adapt[t]:.3f}  GT={mc[t]:.3f}")

        shade_teleop(ax_h)
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
