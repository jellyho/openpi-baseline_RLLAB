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


def _process_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize + pad to square uint8 image."""
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


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
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"Policy only predicts {len(action_chunk)} steps "
                        f"but replan_steps={args.replan_steps}."
                    )
                    action_queue.extend(action_chunk[: args.replan_steps])

                action = action_queue.popleft()
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
