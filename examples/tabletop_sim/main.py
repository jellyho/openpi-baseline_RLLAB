"""Tabletop-Sim inference / evaluation script for openpi policies.

Connects to a running openpi policy server and evaluates it on Tabletop-Sim
tasks.  Mirrors the structure of examples/libero/main.py.

Usage (single task):
    python main.py --task_name aloha_lift_box --num_episodes 50

Usage (all tasks):
    python main.py --task_name all --num_episodes 50
"""
import collections
import dataclasses
import logging
import pathlib

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tabletop
from tabletop.aloha_env import ALOHA_TASK_CONFIGS
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    # ---------------------------------------------------------------------------
    # Policy server
    # ---------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 25

    # ---------------------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------------------
    task_name: str = "aloha_lift_box"
    """Task name. Use 'all' to evaluate all tasks in ALOHA_TASK_CONFIGS."""
    action_space: str = "joint_pos"
    """Action space: 'joint_pos', 'ee_quat_pos', or 'ee_6d_pos'."""
    resize_size: int = 224

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    num_episodes: int = 50
    """Number of rollouts per task."""
    use_benchmark_init: bool = True
    """Use benchmark_init for reproducible initial states when available."""
    seed: int = 0

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    video_out_path: str = "data/tabletop_sim/videos"
    save_video: bool = True
    value_plot: bool = False
    """If the policy returns a critic value, overlay the per-step E[V] curve next to
    the camera (like scripts/visualize_critic.py, but no GT MC return at rollout).
    No-op for non-critic policies (they don't return a value)."""
    action_overlay: bool = False
    """If the policy returns base-policy action samples + the steered chunk (serve with
    --num-action-samples N), render a per-dim overlay video: the N random-latent samples
    (gray) vs the latent-steered chunk (red), in model space.  Debug whether steering
    pushes the action chunk off the base policy's distribution."""


def _process_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize + pad to square uint8 image."""
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def _save_value_video(cam_frames, values, fps, out_path) -> None:
    """Render camera | critic E[V] curve (moving marker) per frame → mp4.

    `values[t]` is the critic's E[V] for the chunk the frame is executing (held
    across each replan).  No ground-truth MC return at rollout, so only E[V].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    n = len(cam_frames)
    x = np.arange(n)
    fin = vals[np.isfinite(vals)]
    ymin = (fin.min() if fin.size else -1.0) - 0.05
    ymax = (max(fin.max(), 0.0) if fin.size else 0.0) + 0.05

    writer = imageio.get_writer(str(out_path), fps=fps)
    for t in range(n):
        fig, (ax_img, ax_v) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.3]}
        )
        ax_img.imshow(cam_frames[t]); ax_img.axis("off"); ax_img.set_title(f"frame {t}/{n-1}")
        ax_v.plot(x[: t + 1], vals[: t + 1], color="#2e86de", lw=2, label="critic E[V]")
        if np.isfinite(vals[t]):
            ax_v.scatter([t], [vals[t]], color="#2e86de", zorder=5)
        ax_v.axvline(t, color="#bbb", ls="--", lw=1)
        ax_v.set_xlim(0, max(n - 1, 1)); ax_v.set_ylim(ymin, ymax)
        ax_v.set_xlabel("frame"); ax_v.set_ylabel("value"); ax_v.grid(alpha=0.3)
        ax_v.set_title(f"E[V]={vals[t]:.3f}" if np.isfinite(vals[t]) else "E[V]=n/a")
        ax_v.legend(loc="lower right", fontsize=9)
        fig.tight_layout(); fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame); plt.close(fig)
    writer.close()


def _camera_projector(physics, camera_id, height, width):
    """world (x,y,z) -> (col,row) pixel for a fixed MuJoCo camera (pinhole, -z forward)."""
    cam_id = physics.model.name2id(camera_id, "camera")
    fovy = float(physics.model.cam_fovy[cam_id]) * np.pi / 180.0
    f = 0.5 * height / np.tan(fovy / 2.0)                       # focal length (px)
    cx, cy = width / 2.0, height / 2.0
    cam_pos = np.asarray(physics.data.cam_xpos[cam_id]).copy()   # camera origin (world)
    cam_mat = np.asarray(physics.data.cam_xmat[cam_id]).reshape(3, 3).copy()  # cam axes in world

    def project(world_pts):
        p = np.asarray(world_pts, dtype=float)
        pc = (p - cam_pos) @ cam_mat        # world -> camera frame  (R^T (p - t))
        depth = -pc[..., 2]                 # MuJoCo cameras look down -z
        u = cx + f * pc[..., 0] / depth
        v = cy - f * pc[..., 1] / depth     # image row grows downward
        uv = np.stack([u, v], axis=-1)
        uv[depth <= 1e-6] = np.nan          # behind the camera -> drop
        return uv

    return project


def _aloha_fk(clone, actions, left_site, right_site):
    """Joint-target actions [..., 14] -> (left_ee, right_ee) world positions [..., 3].

    Treats the joint targets as the achieved qpos (controller assumed to converge),
    forwards kinematics on a throwaway physics copy, and reads the gripper sites.
    Gripper-finger dims are left at default (they don't move the gripper site).
    """
    flat = np.asarray(actions, dtype=float).reshape(-1, actions.shape[-1])
    lout = np.empty((flat.shape[0], 3))
    rout = np.empty((flat.shape[0], 3))
    q = clone.data.qpos
    for i, a in enumerate(flat):
        q[0:6] = a[0:6]      # left arm
        q[8:14] = a[7:13]    # right arm
        clone.forward()
        lout[i] = clone.named.data.site_xpos[left_site]
        rout[i] = clone.named.data.site_xpos[right_site]
    shape = actions.shape[:-1]
    return lout.reshape(*shape, 3), rout.reshape(*shape, 3)


def _save_action_projection_video(
    raw_frames, plan_idx, step_list, plans, physics, camera_id, fps, out_path,
    left_site="left/gripper", right_site="right/gripper",
) -> None:
    """Overlay predicted gripper paths on the camera → mp4, to debug latent steering.

    For the chunk each frame is executing (held across each replan), forward-kinematics
    every action step to the left/right gripper world position and project it into the
    camera: the N base-policy samples (random sphere latents) as faint paths, and the
    latent-steered chunk as a bold path (left=blue, right=orange).  A dot marks the
    step the frame is at.  If the steered path leaves the faint cloud, steering pushes
    the action chunk off the base policy's distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = raw_frames[0].shape[:2]
    project = _camera_projector(physics, camera_id, H, W)
    clone = physics.copy(share_model=True)

    # Project each plan's gripper paths once (chunks are constant within a replan window).
    plan_px = []
    for samples, steered in plans:
        sl, sr = _aloha_fk(clone, steered, left_site, right_site)            # [ah,3] each
        entry = {"steered": (project(sl), project(sr)), "samples": None}
        if samples is not None and len(samples) > 0:
            ls, rs = _aloha_fk(clone, np.asarray(samples), left_site, right_site)  # [N,ah,3]
            entry["samples"] = (project(ls), project(rs))                   # [N,ah,2]
        plan_px.append(entry)

    n = len(raw_frames)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for t in range(n):
        fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
        ax.imshow(raw_frames[t]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        entry = plan_px[plan_idx[t]]
        step = step_list[t]
        if entry["samples"] is not None:
            for arm_px, color in zip(entry["samples"], ("#7fb3ff", "#ffb37f")):
                for k in range(arm_px.shape[0]):
                    ax.plot(arm_px[k, :, 0], arm_px[k, :, 1], color=color, lw=0.5, alpha=0.3)
        for arm_px, color in zip(entry["steered"], ("#1f6fe0", "#e0681f")):
            ax.plot(arm_px[:, 0], arm_px[:, 1], color=color, lw=2.2)
            if step is not None and step < arm_px.shape[0] and np.all(np.isfinite(arm_px[step])):
                ax.scatter([arm_px[step, 0]], [arm_px[step, 1]], color=color, s=30, zorder=5)
        ax.set_title(f"frame {t}/{n-1}  base=faint steered=bold (L blue / R orange)", fontsize=9)
        fig.tight_layout(pad=0); fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame); plt.close(fig)
    writer.close()


def eval_tabletop(args: Args) -> None:
    np.random.seed(args.seed)

    if args.task_name == "all":
        task_names = list(ALOHA_TASK_CONFIGS.keys())
    else:
        if args.task_name not in ALOHA_TASK_CONFIGS:
            raise ValueError(f"Unknown task: {args.task_name}. Available: {list(ALOHA_TASK_CONFIGS.keys())}")
        task_names = [args.task_name]

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    grand_total_episodes = 0
    grand_total_successes = 0

    for task_name in task_names:
        logging.info(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

        task_config = ALOHA_TASK_CONFIGS[task_name]
        # episode_len is in seconds; DT=0.04 => steps = seconds / 0.04
        max_steps = int(task_config["episode_len"] / 0.04)

        env = tabletop.env(task_name, args.action_space)
        task = env.task

        task_episodes = 0
        task_successes = 0

        for ep_idx in tqdm.tqdm(range(args.num_episodes), desc=task_name):
            # --- Reset / initialize episode ---
            has_benchmark = (
                args.use_benchmark_init
                and hasattr(task, "benchmark_info")
                and task.benchmark_info is not None
            )
            if has_benchmark:
                env.reset()
                timestep = task.benchmark_init(env.physics, ep_idx)
            else:
                timestep = env.reset()

            obs = timestep.observation
            action_queue: collections.deque = collections.deque()
            replay_images = []
            replay_values = []   # per-frame critic E[V] (held across each replan)
            replay_raw = []      # per-frame raw back camera frame (for projection overlay)
            replay_plan = []     # per-frame plan index (which replan chunk)
            replay_steps = []    # per-frame intra-chunk position
            plans = []           # (samples [N,ah,14] or None, steered chunk [ah,14]) per replan
            cur_value = None
            cur_plan = -1
            step_in_chunk = 0
            success = False

            for t in range(max_steps):
                # --- Build policy input ---
                img = _process_image(obs["images"]["back"], args.resize_size)
                wrist_left = _process_image(obs["images"]["wrist_left"], args.resize_size)
                wrist_right = _process_image(obs["images"]["wrist_right"], args.resize_size)

                if args.save_video:
                    replay_images.append(img)

                if not action_queue:
                    element = {
                        "state": obs["qpos"].astype(np.float32),
                        "images": {
                            "cam_high": img,
                            "cam_left_wrist": wrist_left,
                            "cam_right_wrist": wrist_right,
                        },
                        "prompt": obs["language_instruction"],
                    }
                    result = client.infer(element)
                    action_chunk = result["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"Policy only predicts {len(action_chunk)} steps "
                        f"but replan_steps={args.replan_steps}."
                    )
                    action_queue.extend(action_chunk[: args.replan_steps])
                    step_in_chunk = 0
                    if args.value_plot and result.get("value") is not None:
                        # full-chunk head ([..., -1] for multi-horizon critics)
                        cur_value = float(np.asarray(result["value"]).reshape(-1)[-1])
                    if args.action_overlay:
                        cur_plan = len(plans)
                        plans.append((result.get("action_samples"), np.asarray(action_chunk)))

                if args.save_video and args.value_plot:
                    replay_values.append(cur_value)
                if args.save_video and args.action_overlay:
                    replay_raw.append(np.asarray(obs["images"]["back"]))
                    replay_plan.append(cur_plan)
                    replay_steps.append(step_in_chunk)

                action = action_queue.popleft()
                step_in_chunk += 1
                timestep = env.step(action.tolist())
                obs = timestep.observation

                if task.reward == task.max_reward:
                    success = True
                    break

                if timestep.last():
                    break

            task_episodes += 1
            grand_total_episodes += 1
            if success:
                task_successes += 1
                grand_total_successes += 1

            # --- Save video ---
            if args.save_video and replay_images:
                suffix = "success" if success else "failure"
                video_path = out_dir / f"{task_name}_ep{ep_idx:03d}_{suffix}.mp4"
                if args.action_overlay and plans and replay_raw:
                    overlay_path = out_dir / f"{task_name}_ep{ep_idx:03d}_{suffix}_actions.mp4"
                    _save_action_projection_video(
                        replay_raw, replay_plan, replay_steps, plans,
                        env.physics, "teleoperator_pov", 25, overlay_path,
                    )
                if args.value_plot and any(v is not None for v in replay_values):
                    _save_value_video(replay_images, replay_values, 25, video_path)
                elif not args.action_overlay:
                    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=25)

            logging.info(
                f"  Episode {ep_idx+1}: {'SUCCESS' if success else 'FAILURE'} | "
                f"task SR so far: {task_successes}/{task_episodes} "
                f"({100*task_successes/task_episodes:.1f}%)"
            )

        logging.info(
            f"Task '{task_name}' final success rate: "
            f"{task_successes}/{task_episodes} ({100*task_successes/task_episodes:.1f}%)"
        )

    logging.info(
        f"\nOverall success rate: "
        f"{grand_total_successes}/{grand_total_episodes} "
        f"({100*grand_total_successes/grand_total_episodes:.1f}%)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_tabletop)
