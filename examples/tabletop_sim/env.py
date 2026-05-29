import numpy as np
import tabletop
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

RESIZE_SIZE = 224


class TabletopSimEnvironment(_environment.Environment):
    """Environment wrapper for Tabletop-Sim (dual-arm Aloha in MuJoCo).

    Observation format passed to the policy:
      - state:  float32 [14] -- qpos: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                                gripper values are in [-1, 1] (tabletop normalized space)
      - images: dict -- cam_high, cam_left_wrist, cam_right_wrist  each uint8 [C, H, W]
      - prompt: str  -- task language instruction

    Action format received from the policy:
      - actions: float32 [14] -- absolute joint positions in the same space as state
    """

    def __init__(
        self,
        task_name: str,
        action_space: str = "joint_pos",
        benchmark_idx: int | None = None,
        seed: int = 0,
    ) -> None:
        np.random.seed(seed)
        self._env = tabletop.env(task_name, action_space)
        self._benchmark_idx = benchmark_idx
        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

    @override
    def reset(self) -> None:
        task = self._env.task
        has_benchmark = (
            self._benchmark_idx is not None
            and hasattr(task, "benchmark_info")
            and task.benchmark_info is not None
        )
        if has_benchmark:
            self._env.reset()
            timestep = task.benchmark_init(self._env.physics, self._benchmark_idx)
        else:
            timestep = self._env.reset()
        self._last_obs = self._convert_observation(timestep.observation)
        self._done = False
        self._episode_reward = 0.0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation not set. Call reset() first.")
        return self._last_obs

    @override
    def apply_action(self, action: dict) -> None:
        timestep = self._env.step(action["actions"])
        self._last_obs = self._convert_observation(timestep.observation)
        self._done = timestep.last()
        reward = timestep.reward or 0.0
        self._episode_reward = max(self._episode_reward, reward)

    @property
    def is_success(self) -> bool:
        task = self._env.task
        return task.reward == task.max_reward

    def _convert_observation(self, obs: dict) -> dict:
        def process_image(img: np.ndarray) -> np.ndarray:
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
            return np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return {
            "state": obs["qpos"].astype(np.float32),
            "images": {
                "cam_high": process_image(obs["images"]["back"]),
                "cam_left_wrist": process_image(obs["images"]["wrist_left"]),
                "cam_right_wrist": process_image(obs["images"]["wrist_right"]),
            },
            "prompt": obs["language_instruction"],
        }
