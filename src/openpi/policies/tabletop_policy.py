"""Policy transforms for the Tabletop-Sim environment.

The Tabletop-Sim uses the dual-arm Aloha robot in MuJoCo (via dm_control).

State / action space (joint_pos mode, 14 dims):
  [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  - Joint angles are in radians.
  - Gripper values are in [-1, 1] (tabletop's own normalized space, NOT the
    Aloha linear-space used by the real robot).

Because the gripper normalization convention differs from the real Aloha robot,
we use adapt_to_pi=False so no gripper-space conversion is performed.

Camera names:
  cam_high        -- main overhead/back camera (480x640 -> resized to 224x224)
  cam_left_wrist  -- left wrist camera
  cam_right_wrist -- right wrist camera
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_tabletop_example() -> dict:
    """Creates a random input example for the tabletop policy (for testing)."""
    return {
        "state": np.random.rand(14).astype(np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the box",
    }


@dataclasses.dataclass(frozen=True)
class TabletopInputs(transforms.DataTransformFn):
    """Input transform for the Tabletop-Sim policy.

    Expected inputs (from env or lerobot dataset after repack):
      - images: dict[name, img] where img is CHW uint8 or float32.
                name must be a subset of: cam_high, cam_left_wrist, cam_right_wrist.
      - state:   float32 [14]  -- qpos in tabletop normalized space
      - actions: float32 [T, 14] -- only present during training
      - prompt:  str

    Padding to action_dim is handled downstream by PadStatesAndActions in model_transforms.
    """

    model_type: _model.ModelType = _model.ModelType.PI0
    # If True, the images/state come as [current, next] windows (LPS-RFT chunked
    # TD); split them into the current obs + a model-ready next obs (+ `done`).
    # The next state is s_{t+H} (full-chunk end) — also used as the single backup
    # for multi-horizon Q-chunking (multi-horizon prediction, single-state backup).
    load_next_obs: bool = False

    EXPECTED_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(f"Unexpected camera keys: {unknown}")

        def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:  # CHW -> HWC
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        # Camera-name → model image slot mapping.
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                extra_slots = {
                    "left_wrist_0_rgb": "cam_left_wrist",
                    "right_wrist_0_rgb": "cam_right_wrist",
                }
            case _model.ModelType.PI0_FAST:
                extra_slots = {
                    "base_1_rgb": "cam_left_wrist",
                    "wrist_0_rgb": "cam_right_wrist",
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        def build_image_dict(cam_imgs: dict) -> tuple[dict, dict]:
            base = to_hwc_uint8(cam_imgs["cam_high"])
            imgs = {"base_0_rgb": base}
            masks = {"base_0_rgb": np.True_}
            for dest, source in extra_slots.items():
                if source in cam_imgs:
                    imgs[dest] = to_hwc_uint8(cam_imgs[source])
                    masks[dest] = np.True_
                else:
                    imgs[dest] = np.zeros_like(base)
                    masks[dest] = np.False_
            return imgs, masks

        # The next-obs window only exists during TRAINING (the data loader supplies
        # it).  At inference the env gives a single observation (no `next_is_pad`),
        # so fall through to the single-obs path below.
        if self.load_next_obs and "next_is_pad" in data:
            # images[cam] = [current, next] (2, C, H, W); state = [current, next] (2, S).
            cur_imgs  = {k: np.asarray(v)[0] for k, v in in_images.items()}
            next_imgs = {k: np.asarray(v)[1] for k, v in in_images.items()}
            images, image_masks = build_image_dict(cur_imgs)
            next_image, next_image_mask = build_image_dict(next_imgs)
            state_window = np.asarray(data["state"])             # (2, S)
            inputs = {
                "image": images,
                "image_mask": image_masks,
                "state": state_window[0],
                "next_image": next_image,
                "next_image_mask": next_image_mask,
                "next_state": state_window[1],
                # done: next frame (offset H) is past the episode end → no bootstrap.
                "done": np.asarray(data["next_is_pad"])[1],
            }
        else:
            images, image_masks = build_image_dict(dict(in_images))
            inputs = {
                "image": images,
                "image_mask": image_masks,
                "state": np.asarray(data["state"]),
            }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # MC return (scalar, CalQL anchor) + reward window [H] (chunked-TD target).
        if "mc_return" in data:
            inputs["mc_return"] = np.asarray(data["mc_return"])
        if "reward" in data:
            inputs["reward"] = np.asarray(data["reward"])

        return inputs


@dataclasses.dataclass(frozen=True)
class TabletopOutputs(transforms.DataTransformFn):
    """Output transform for the Tabletop-Sim policy.

    Returns the first 14 action dimensions (joint_pos format).
    No gripper-space conversion is applied (tabletop uses its own [-1, 1] space).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
