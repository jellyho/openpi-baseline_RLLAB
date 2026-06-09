"""Diagnostic rollout visualization for Adaptive Q-Chunking (AQC / ACSAC).

Produces a composite video that makes the adaptive mechanism *visible*: on the left, the
environment rollout; on the right, two synchronized panels:

  * Top:    the prefix-conditioned value curve ``max_n Q(s, a^(n)_{1:h})`` for ``h = 1..H``
            at the current replanning state, with the selected chunk size ``h*`` highlighted.
            This is exactly the curve the joint arg-max selects from.
  * Bottom: the timeline of executed chunk sizes ``h*`` and the selected Q over the episode,
            so you can watch the execution horizon shrink at precise/turning phases and grow
            on straight segments.

The frames are stitched into a single video suitable for ``wandb.Video``. This is the
qualitative check from the paper (Fig. 3, "distribution of chunk size decisions") plus the
prefix-Q calibration intuition (Fig. 4), rendered per-rollout.

Dependencies: matplotlib (Agg) + PIL + numpy (all in the project env). No GPU rendering of
the panels (Agg is software), so it does not interfere with MuJoCo's EGL renderer.
"""

import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

_SELECTED = '#d62728'   # red
_CURVE = '#1f77b4'      # blue
_BARS = '#cfd4da'       # gray


def _hconcat(env_frame, panel):
    """Resize the env frame to the panel height and concatenate horizontally."""
    ph, pw = panel.shape[:2]
    ef = Image.fromarray(env_frame.astype(np.uint8)).resize((ph, ph))
    ef = np.asarray(ef)[..., :3]
    return np.concatenate([ef, panel[..., :3]], axis=1)


def _render_episode_frames(records, horizon, qlo, qhi, panel_px=(440, 400), fig_ctx=None):
    """Render composite frames for one episode's records.

    Args:
        records: list of dicts with keys frame, step, h_star, q_by_h, q_best.
        horizon: max chunk size H.
        qlo, qhi: global Q-axis limits (for stable axes across the whole video).
        panel_px: (width, height) of the matplotlib panel in pixels.
        fig_ctx: optional (fig, ax_q, ax_t, ax_t2) to reuse across episodes.

    Returns:
        list of composite frames (H, W, 3) uint8.
    """
    pw, ph = panel_px
    dpi = 100
    if fig_ctx is None:
        fig = plt.figure(figsize=(pw / dpi, ph / dpi), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.5,
                              left=0.16, right=0.86, top=0.9, bottom=0.12)
        ax_q = fig.add_subplot(gs[0])
        ax_t = fig.add_subplot(gs[1])
        ax_t2 = ax_t.twinx()
        fig_ctx = (fig, ax_q, ax_t, ax_t2)
    fig, ax_q, ax_t, ax_t2 = fig_ctx

    hs_x = np.arange(1, horizon + 1)
    steps = [r['step'] for r in records]
    hsel = [r['h_star'] for r in records]
    qsel = [r['q_best'] for r in records]
    xmax = max(steps) + 1 if steps else 1
    frames = []

    for i, r in enumerate(records):
        ax_q.clear(); ax_t.clear(); ax_t2.clear()

        # --- top: prefix-Q curve, selected h* highlighted ---
        qb = np.asarray(r['q_by_h'])
        colors = [_BARS] * horizon
        colors[r['h_star'] - 1] = _SELECTED
        ax_q.bar(hs_x, qb, color=colors, width=0.7, zorder=1)
        ax_q.plot(hs_x, qb, '-o', color=_CURVE, lw=1.5, ms=4, zorder=2)
        ax_q.axvline(r['h_star'], color=_SELECTED, ls='--', lw=1.0, alpha=0.6)
        ax_q.set_ylim(qlo, qhi)
        ax_q.set_xticks(hs_x)
        ax_q.set_xlabel('prefix length h', fontsize=8)
        ax_q.set_ylabel('max-candidate Q', fontsize=8)
        ax_q.tick_params(labelsize=7)
        ax_q.set_title(f'step {r["step"]}   selected h*={r["h_star"]}   Q={r["q_best"]:.2f}',
                       fontsize=9, color=_SELECTED)

        # --- bottom: timeline of h* (left) and selected Q (right) ---
        ax_t.plot(steps[:i + 1], hsel[:i + 1], drawstyle='steps-post',
                  color=_SELECTED, lw=1.5)
        ax_t.scatter([r['step']], [r['h_star']], color=_SELECTED, s=24, zorder=5)
        ax_t.set_ylim(0.5, horizon + 0.5)
        ax_t.set_yticks(hs_x)
        ax_t.set_xlim(0, xmax)
        ax_t.set_xlabel('env step', fontsize=8)
        ax_t.set_ylabel('chunk size h*', color=_SELECTED, fontsize=8)
        ax_t.tick_params(labelsize=7)
        ax_t.tick_params(axis='y', labelcolor=_SELECTED)

        ax_t2.plot(steps[:i + 1], qsel[:i + 1], color=_CURVE, lw=1.2)
        ax_t2.set_ylim(qlo, qhi)
        ax_t2.set_ylabel('selected Q', color=_CURVE, fontsize=8)
        ax_t2.tick_params(axis='y', labelcolor=_CURVE, labelsize=7)

        fig.canvas.draw()
        panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if panel.shape[:2] != (ph, pw):
            panel = np.asarray(Image.fromarray(panel).resize((pw, ph)))
        frames.append(_hconcat(r['frame'], panel))

    return frames, fig_ctx


def aqc_rollout_video(agent, env, action_dim, horizon, rng,
                      num_episodes=1, frame_skip=3, max_steps=1000, panel_px=(440, 400)):
    """Run adaptive rollouts and return a composite diagnostic video.

    Args:
        agent: an AQC agent exposing ``sample_actions_with_info``.
        env: evaluation environment (must support ``render()`` returning RGB frames).
        action_dim: per-step action dimension.
        horizon: maximum chunk size H.
        rng: JAX PRNG key.
        num_episodes: number of rollout episodes to include.
        frame_skip: capture one frame every ``frame_skip`` env steps.
        max_steps: safety cap on steps per episode.
        panel_px: (width, height) of the diagnostics panel.

    Returns:
        ``(video, stats)`` where ``video`` is a ``(T, C, H, W)`` uint8 array ready for
        ``wandb.Video`` and ``stats`` is a dict (mean/min/max executed chunk size).
    """
    all_episode_records = []
    all_h = []

    for _ in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        action_queue = []
        cur = None
        records = []
        step = 0
        done = False
        while not done and step < max_steps:
            if len(action_queue) == 0:
                rng, key = jax.random.split(rng)
                best_chunk, h_star, info = agent.sample_actions_with_info(observations=obs, rng=key)
                h_star = int(h_star)
                cur = dict(h_star=h_star,
                           q_by_h=np.asarray(info['q_by_h']),
                           q_best=float(info['q_best']))
                all_h.append(h_star)
                chunk = np.asarray(best_chunk).reshape(-1, action_dim)[:h_star]
                action_queue.extend(list(chunk))
            action = action_queue.pop(0)
            step_out = env.step(np.clip(action, -1, 1))
            obs, terminated, truncated = step_out[0], step_out[2], step_out[3]
            done = bool(terminated) or bool(truncated)
            step += 1
            if step % frame_skip == 0 or done:
                frame = np.asarray(env.render()).copy()
                records.append(dict(frame=frame, step=step, **cur))
        all_episode_records.append(records)

    # Global Q-axis limits across the whole video for stable, comparable axes.
    all_q = np.concatenate([r['q_by_h'] for ep in all_episode_records for r in ep]) \
        if any(all_episode_records) else np.array([0.0, 1.0])
    qlo, qhi = float(all_q.min()), float(all_q.max())
    if qhi - qlo < 1e-6:
        qhi = qlo + 1.0
    pad = 0.1 * (qhi - qlo)
    qlo, qhi = qlo - pad, qhi + pad

    frames = []
    fig_ctx = None
    for records in all_episode_records:
        if not records:
            continue
        ep_frames, fig_ctx = _render_episode_frames(records, horizon, qlo, qhi, panel_px, fig_ctx)
        frames.extend(ep_frames)
    if fig_ctx is not None:
        plt.close(fig_ctx[0])

    if not frames:
        return None, dict(mean_h=0.0, min_h=0.0, max_h=0.0)

    video = np.stack(frames, axis=0)              # (T, H, W, C)
    video = np.transpose(video, (0, 3, 1, 2))     # (T, C, H, W) for wandb.Video
    stats = dict(mean_h=float(np.mean(all_h)),
                 min_h=float(np.min(all_h)),
                 max_h=float(np.max(all_h)))
    return video.astype(np.uint8), stats
