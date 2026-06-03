import dataclasses

import einops
import numpy as np
from typing import ClassVar
from openpi import transforms
from openpi.models import model as _model


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Piper transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Piper code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Piper.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Piper code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)

#######################################
# YAM robot set-up.                   #
#######################################
@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs for the World Engine's Yam policy.
    The difference between Yam & Piper, is their 3rd joint is inversed.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Piper space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # LPS-RFT chunked TD: images/state come as [current, next] windows; split into
    # the current obs + a model-ready next obs s_{t+H} (+ `done`).
    load_next_obs: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def _image_dict(self, in_images: dict) -> tuple[dict, dict]:
        """Map decoded cameras (hwc uint8) to model image slots + masks."""
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        base_image = in_images["cam_high"]
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                extra_image_names = {"left_wrist_0_rgb": "cam_left_wrist", "right_wrist_0_rgb": "cam_right_wrist"}
            case _model.ModelType.PI0_FAST:
                extra_image_names = {"base_1_rgb": "cam_left_wrist", "wrist_0_rgb": "cam_right_wrist"}
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_
        return images, image_masks

    def _decode_obs(self, state: np.ndarray, images: dict) -> tuple[np.ndarray, dict]:
        """Decode one observation: yam state (joint flips / gripper) + images (hwc uint8)."""
        d = _decode_yam({"state": state, "images": images}, adapt_to_pi=self.adapt_to_pi)
        return d["state"], d["images"]

    def __call__(self, data: dict) -> dict:
        if self.load_next_obs:
            # state = [current, next] (2, 14); images[cam] = [current, next] (2, C, H, W).
            state_win = np.asarray(data["state"])
            imgs = {k: np.asarray(v) for k, v in data["images"].items()}
            cur_state, cur_imgs = self._decode_obs(state_win[0], {k: v[0] for k, v in imgs.items()})
            nxt_state, nxt_imgs = self._decode_obs(state_win[1], {k: v[1] for k, v in imgs.items()})
            images, image_masks = self._image_dict(cur_imgs)
            next_image, next_image_mask = self._image_dict(nxt_imgs)
            inputs = {
                "image": images,
                "image_mask": image_masks,
                "state": transforms.pad_to_dim(cur_state, self.action_dim),
                "next_image": next_image,
                "next_image_mask": next_image_mask,
                "next_state": transforms.pad_to_dim(nxt_state, self.action_dim),
                # done: next frame (offset H) is past the episode end → no bootstrap.
                "done": np.asarray(data["next_is_pad"])[1],
            }
        else:
            state, in_images = self._decode_obs(np.asarray(data["state"]), data["images"])
            images, image_masks = self._image_dict(in_images)
            inputs = {
                "image": images,
                "image_mask": image_masks,
                "state": transforms.pad_to_dim(state, self.action_dim),
            }

        # Actions are only available during training.
        if "actions" in data:
            actions = _encode_yam_actions_inv(np.asarray(data["actions"]), adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # MC return (scalar, CalQL anchor) + reward window [H] (chunked-TD target).
        if "mc_return" in data:
            inputs["mc_return"] = np.asarray(data["mc_return"])
        if "reward" in data:
            inputs["reward"] = np.asarray(data["reward"])

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs for the World Engine's Yam policy."""

    # If true, this will convert the joint and gripper values from the standard Yam space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_yam_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _yam_joint_flip_mask() -> np.ndarray:
    """Used to convert between yam and pi joint angles. (Yam is the same as Arx)"""
    return np.array([1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])


def _decode_yam(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_yam_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_yam_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _yam_joint_flip_mask() * state[: _yam_joint_flip_mask().shape[0]]
        # Reverse the gripper transformation that is being applied by the Piper runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_yam_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _yam_joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_yam_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _yam_joint_flip_mask() * actions[:, : len(_yam_joint_flip_mask())]
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions