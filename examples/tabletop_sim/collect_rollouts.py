"""Collect TwinVLA rollouts on aloha_handover_box and save as a LeRobot dataset.

Pipeline per episode
====================
1. [ee_6d_pos env]  TwinVLA rollout  →  sequence of eef_6d_pos actions (20-dim)
2. [IK]             eef_6d_pos  →  joint_pos  via AlohaIK (for both arms)
3. [joint_pos env]  Replay joint_pos actions from the SAME initial state
4. [LeRobot]        Save qpos states, joint_pos actions, images, reward, done

Both environments are initialised with the same benchmark_init(idx) so the
replay starts from an identical object configuration.

Usage
-----
    python collect_rollouts.py \
        --checkpoint jellyho/TwinVLA-aloha_handover_box \
        --tgt_dir /path/to/output \
        --repo_id  jellyho/aloha_handover_box_joint_pos_rl \
        --num_episodes 500

Set TWINVLA_PATH env-var if TwinVLA is not already on PYTHONPATH.
"""

import os
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm

# ---------------------------------------------------------------------------
# TwinVLA path
# ---------------------------------------------------------------------------
_twinvla_root = os.environ.get("TWINVLA_PATH", str(Path(__file__).parents[4] / "TwinVLA"))
if _twinvla_root not in sys.path:
    sys.path.insert(0, _twinvla_root)

from twinvla.model.twinvla import TwinVLA  # noqa: E402

# ---------------------------------------------------------------------------
# Tabletop-Sim
# ---------------------------------------------------------------------------
import tabletop  # noqa: E402
from tabletop.aloha_ik import AlohaIK  # noqa: E402
from tabletop.utils import sixd_to_quat  # noqa: E402
from dm_env import StepType  # noqa: E402

# ---------------------------------------------------------------------------
# LeRobot
# ---------------------------------------------------------------------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import DEFAULT_FEATURES  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# LeRobot dataset features — must match jellyho/aloha_handover_box_joint_pos_rl
# ---------------------------------------------------------------------------
_VIDEO_INFO_480 = {
    "video.height": 480, "video.width": 640, "video.codec": "av1",
    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
    "video.fps": 25.0, "video.channels": 3, "has_audio": False,
}
_VIDEO_INFO_240 = {
    "video.height": 240, "video.width": 320, "video.codec": "av1",
    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
    "video.fps": 25.0, "video.channels": 3, "has_audio": False,
}
_JOINT_NAMES = [
    "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
]

DATASET_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": _JOINT_NAMES,
    },
    "observation.images.back": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": _VIDEO_INFO_480,
    },
    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
        "info": _VIDEO_INFO_240,
    },
    "observation.images.wrist_right": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
        "info": _VIDEO_INFO_240,
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": _JOINT_NAMES,
    },
    "next.reward":  {"dtype": "float32", "shape": (1,), "names": None},
    "next.success": {"dtype": "bool",    "shape": (1,), "names": None},
}
DATASET_FEATURES.update(DEFAULT_FEATURES)  # adds timestamp, frame_index, etc.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # TwinVLA checkpoint (HF Hub id or local path)
    checkpoint: str = "jellyho/TwinVLA-aloha_handover_box"
    unnorm_key: str = "aloha_handover_box"
    action_len: int = 20
    cfg_scale: float = 1.0
    f32: bool = False

    # Data collection
    task_name: str = "aloha_handover_box"
    num_episodes: int = 500
    seed: int = 0

    # LeRobot output
    tgt_dir: str = "data/tabletop_sim/rl_dataset"
    repo_id: str = "jellyho/aloha_handover_box_joint_pos_rl"
    fps: int = 25
    push_to_hub: bool = False


# ---------------------------------------------------------------------------
# IK helper
# ---------------------------------------------------------------------------
def eef6d_to_joint_pos(
    actions_6d: list[np.ndarray],
    initial_qpos: np.ndarray,
    aloha_ik: AlohaIK,
) -> list[np.ndarray]:
    """Convert a sequence of eef_6d_pos actions (20-dim) to joint_pos (14-dim).

    eef_6d_pos layout (20 dims):
      [0:3]   left  EE position
      [3:9]   left  6D rotation
      [9]     left  gripper  [-1, 1]
      [10:13] right EE position
      [13:19] right 6D rotation
      [19]    right gripper  [-1, 1]

    joint_pos layout (14 dims):
      [0:6]   left  arm joints (rad)
      [6]     left  gripper  [-1, 1]
      [7:13]  right arm joints (rad)
      [13]    right gripper  [-1, 1]
    """
    joint_pos_seq = []
    prev_qpos = initial_qpos.copy().astype(np.float64)

    for action_6d in actions_6d:
        # --- Left arm ---
        quat_left = sixd_to_quat(action_6d[3:9])
        qpos_left = aloha_ik.get_joint_pos(
            target_pos=action_6d[0:3],
            target_quat=quat_left,
            curr_qpos=prev_qpos[:6],
        )

        # --- Right arm (same IK model; both arms are kinematically identical) ---
        quat_right = sixd_to_quat(action_6d[13:19])
        qpos_right = aloha_ik.get_joint_pos(
            target_pos=action_6d[10:13],
            target_quat=quat_right,
            curr_qpos=prev_qpos[7:13],
        )

        joint_pos = np.concatenate([
            qpos_left,
            [action_6d[9]],    # left gripper, already in [-1, 1]
            qpos_right,
            [action_6d[19]],   # right gripper, already in [-1, 1]
        ]).astype(np.float32)

        joint_pos_seq.append(joint_pos)
        prev_qpos = joint_pos.astype(np.float64)

    return joint_pos_seq


# ---------------------------------------------------------------------------
# Core per-episode logic
# ---------------------------------------------------------------------------
def collect_one_episode(
    episode_idx: int,
    model: TwinVLA,
    env_6d,       # tabletop env with ee_6d_pos action space
    env_jnt,      # tabletop env with joint_pos action space
    aloha_ik: AlohaIK,
    cfg: Config,
) -> tuple[list[dict], bool]:
    """Run TwinVLA rollout, IK-convert, replay, and return frames + success flag.

    Returns
    -------
    frames : list of per-step dicts ready to be added to a LeRobot dataset
    replay_success : bool  (True = box touched basket in replay env)
    """

    # ------------------------------------------------------------------
    # Phase 1: TwinVLA rollout (ee_6d_pos)
    # ------------------------------------------------------------------
    ts = env_6d.reset()
    ts = env_6d.task.benchmark_init(env_6d.physics, episode_idx)

    actions_6d: list[np.ndarray] = []
    action_counter = 0
    action_chunk: np.ndarray | None = None

    with torch.inference_mode():
        while True:
            obs = ts.observation

            if action_counter == 0:
                action_chunk = model.predict_action(
                    unnorm_key=cfg.unnorm_key,
                    instruction=obs["language_instruction"],
                    image=obs["images"]["back"],
                    image_wrist_r=obs["images"]["wrist_right"],
                    image_wrist_l=obs["images"]["wrist_left"],
                    proprio=obs["ee_6d_pos"],
                    cfg=cfg.cfg_scale,
                )

            action = action_chunk[action_counter]
            actions_6d.append(action.copy())
            ts = env_6d.step(action)
            action_counter += 1

            if action_counter == cfg.action_len:
                action_counter = 0
            if ts.reward == env_6d.task.max_reward or ts.step_type == StepType.LAST:
                break

    # ------------------------------------------------------------------
    # Phase 2: IK  eef_6d_pos → joint_pos
    # ------------------------------------------------------------------
    # Use the initial qpos from the replay env (same benchmark init)
    env_jnt.reset()
    ts_jnt = env_jnt.task.benchmark_init(env_jnt.physics, episode_idx)
    initial_qpos = ts_jnt.observation["qpos"].copy()  # 14-dim

    joint_pos_actions = eef6d_to_joint_pos(actions_6d, initial_qpos, aloha_ik)

    # ------------------------------------------------------------------
    # Phase 3: Replay joint_pos actions
    # ------------------------------------------------------------------
    frames: list[dict] = []
    n = len(joint_pos_actions)

    for i, action in enumerate(joint_pos_actions):
        obs = ts_jnt.observation
        reward_val = float(ts_jnt.reward) if ts_jnt.reward is not None else 0.0

        frames.append({
            "observation.state": obs["qpos"].astype(np.float32),
            "observation.images.back":        obs["images"]["back"],
            "observation.images.wrist_left":  obs["images"]["wrist_left"],
            "observation.images.wrist_right": obs["images"]["wrist_right"],
            "action": action,
            "task": obs["language_instruction"],
            # next.reward / next.success will be filled after the step below
        })

        ts_jnt = env_jnt.step(action)

        # Record reward/success AFTER the step (i.e. "next" state)
        next_reward = float(ts_jnt.reward) if ts_jnt.reward is not None else 0.0
        next_success = (env_jnt.task.reward == env_jnt.task.max_reward)
        frames[-1]["next.reward"]  = np.array([next_reward],  dtype=np.float32)
        frames[-1]["next.success"] = np.array([next_success], dtype=bool)

    replay_success = (env_jnt.task.reward == env_jnt.task.max_reward)
    return frames, replay_success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: Config) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load TwinVLA
    dtype = torch.float32 if cfg.f32 else torch.bfloat16
    print(f"Loading TwinVLA from {cfg.checkpoint} ...")
    model = TwinVLA(pretrained_path=cfg.checkpoint, dtype=dtype)
    model.eval()

    # IK solver (shared across episodes)
    aloha_ik = AlohaIK()

    # Two environments: one for rollout, one for replay
    env_6d  = tabletop.env(cfg.task_name, "ee_6d_pos")
    env_jnt = tabletop.env(cfg.task_name, "joint_pos")

    # Create LeRobot dataset
    tgt_dir = Path(cfg.tgt_dir)
    if tgt_dir.exists():
        ans = input(f"{tgt_dir} already exists. Delete and recreate? (y/n): ")
        if ans.lower() == "y":
            import shutil
            shutil.rmtree(tgt_dir)
        else:
            raise FileExistsError(f"{tgt_dir} already exists.")

    ds = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        root=str(tgt_dir),
        features=DATASET_FEATURES,
        fps=cfg.fps,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=8,
    )

    # Collect episodes
    successes = 0
    for ep_idx in tqdm.tqdm(range(cfg.num_episodes), desc="Episodes"):
        try:
            frames, success = collect_one_episode(
                episode_idx=ep_idx,
                model=model,
                env_6d=env_6d,
                env_jnt=env_jnt,
                aloha_ik=aloha_ik,
                cfg=cfg,
            )
        except Exception as e:
            print(f"  Episode {ep_idx} failed: {e}")
            continue

        for frame in frames:
            ds.add_frame(frame)
        ds.save_episode()

        if success:
            successes += 1

        if (ep_idx + 1) % 50 == 0 or ep_idx == cfg.num_episodes - 1:
            sr = successes / (ep_idx + 1)
            print(f"  [{ep_idx+1}/{cfg.num_episodes}] replay success rate: {sr:.1%}")

    ds.finalize()
    print(f"\nDone. {successes}/{cfg.num_episodes} episodes succeeded in replay.")

    if cfg.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {cfg.repo_id} ...")
        ds.push_to_hub()


if __name__ == "__main__":
    import draccus
    main(draccus.parse(Config))
