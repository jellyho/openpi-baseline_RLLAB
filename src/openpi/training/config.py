"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_alphaflow as pi0_alphaflow
import openpi.models.pi0_alphaflow_critic as pi0_alphaflow_critic
import openpi.models.pi0_lps_rft as pi0_lps_rft
import openpi.models.pi0_rlt as pi0_rlt
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.tabletop_policy as tabletop_policy
import openpi.policies.yam_policy as yam_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # LPS-RFT chunked-TD: if True the loader additionally fetches a reward window
    # `reward[t:t+H]` and the next observation (`rl_obs_keys` at offset H = s_{t+H})
    # so compute_loss can build  y = R_chunk + γ^H·V(s')  with done masking.
    # `rl_obs_keys` are the RAW LeRobot column names loaded at offsets [0, H].
    load_rl_windows: bool = False
    rl_obs_keys: Sequence[str] = ()
    # Step offsets (in env steps) at which to load the obs window for the TD
    # bootstrap.  Index 0 is the current state; the rest are next states.  Single
    # chunk-TD uses (0, H); multi-horizon Q-chunking uses (0,)+td_horizons so the
    # critic can bootstrap V(s_{t+k}) at each designated horizon k.  Empty → (0, H).
    rl_obs_offsets: Sequence[int] = ()

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    local_files_path: str | None = None

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class DualYamDataConfig(DataConfigFactory):
    """Data class for dual-arm yam system."""

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = ""
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    # RFT data: load mc_return (critic CalQL anchor); and the reward window + next
    # observation s_{t+H} + done (LPS-RFT chunked-TD target).  Requires the dataset
    # to have `mc_return` / `reward` columns (see scripts/compute_mc_returns.py).
    include_mc_return: bool = False
    include_next_obs: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack raw LeRobot columns into the common format; add RFT columns on demand.
        repack_map = {
            "images": {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }
        if self.include_mc_return:
            repack_map["mc_return"] = "mc_return"
        if self.include_next_obs:
            # reward[t:t+H] window; observation.state_is_pad marks whether the next
            # frame (offset H) is past the episode end → `done`.
            repack_map["reward"] = "reward"
            repack_map["next_is_pad"] = "observation.state_is_pad"
        repack_transforms = _transforms.Group(inputs=[_transforms.RepackTransform(repack_map)])

        # Prepare data for policy training (uint8 images + masks; next-obs split for RFT).
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(
                action_dim=model_config.action_dim,
                adapt_to_pi=self.adapt_to_pi,
                model_type=model_config.model_type,
                load_next_obs=self.include_next_obs,
            )],
            outputs=[yam_policy.YamOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            # Left and right arm joints use delta actions, grippers use absolute actions
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)], # Convert to delta actions
                outputs=[_transforms.AbsoluteActions(delta_action_mask)], # Convert back to absolute actions during inference
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            load_rl_windows=self.include_next_obs,
            rl_obs_keys=(
                "observation.state",
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ) if self.include_next_obs else (),
            rl_obs_offsets=(0, model_config.action_horizon) if self.include_next_obs else (),
        )

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTabletopDataConfig(DataConfigFactory):
    """Data config for the Tabletop-Sim (dual-arm Aloha in MuJoCo).

    The dataset is expected to be in LeRobot format with these keys:
      observation.images.agentview    -- main overhead camera
      observation.images.wrist_left   -- left wrist camera
      observation.images.wrist_right  -- right wrist camera
      observation.state.joint_pos     -- qpos [14]
      action.joint_pos                -- target joint positions [14]
      task                            -- language instruction

    All gripper values are in tabletop's [-1, 1] normalized space.
    We therefore set adapt_to_pi=False (no Aloha-specific gripper conversion).
    """

    # Convert arm joints to delta actions; keep grippers absolute.
    # Set to False when dataset already stores absolute joint positions (tabletop default).
    use_delta_joint_actions: bool = False

    # If True, also load the 'mc_return' column (required for critic training).
    # The dataset must contain an 'mc_return' column (see scripts/compute_mc_returns.py).
    include_mc_return: bool = False

    # If True, also load the chunked-TD windows for LPS-RFT: a reward window
    # reward[t:t+H] and the next observation s_{t+H} (state/images), plus a `done`
    # flag (whether s_{t+H} is past the episode end).  Requires include_mc_return.
    include_next_obs: bool = False

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Bootstrap obs offsets: a single next state at the full-chunk end (0, H).
        # Multi-horizon Q-chunking does multi-horizon PREDICTION but a single-state
        # BACKUP (same target broadcast to all heads), so it needs only s_{t+H}.
        rl_obs_offsets = (0, model_config.action_horizon) if self.include_next_obs else ()
        # Build repack mapping; optionally include mc_return for critic training.
        repack_map = {
            "images": {
                "cam_high": "observation.images.back",
                "cam_left_wrist": "observation.images.wrist_left",
                "cam_right_wrist": "observation.images.wrist_right",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }
        if self.include_mc_return:
            repack_map["mc_return"] = "mc_return"
        if self.include_next_obs:
            # reward[t:t+H] window for the chunked-TD target; and observation.state's
            # is_pad marks whether the next frame (offset H) is past the episode end → `done`.
            repack_map["reward"] = "reward"
            repack_map["next_is_pad"] = "observation.state_is_pad"
        repack_transforms = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        )

        data_transforms = _transforms.Group(
            inputs=[tabletop_policy.TabletopInputs(
                model_type=model_config.model_type,
                load_next_obs=self.include_next_obs,
            )],
            outputs=[tabletop_policy.TabletopOutputs()],
        )
        if self.use_delta_joint_actions:
            # Arms: delta actions.  Grippers: absolute.
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            load_rl_windows=self.include_next_obs,
            rl_obs_keys=(
                "observation.state",
                "observation.images.back",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            ) if self.include_next_obs else (),
            rl_obs_offsets=rl_obs_offsets,
            prompt_from_task=True,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 32
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ---------------------------------------------------------------------------
# Shared config building blocks — keep _CONFIGS compact so only the per-config
# DIFFERENCES stay inline.  Each call returns a fresh frozen dataclass, identical
# to writing the literal out (verified behaviour-preserving).
# ---------------------------------------------------------------------------
_PI05_BASE_ASSETS = AssetsConfig(
    assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets", asset_id="trossen"
)
_PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"


def _tabletop_data(repo_id, *, include_mc_return=False, include_next_obs=False):
    """LeRobotTabletopDataConfig on pi05_base assets (prompt-from-task, abs actions)."""
    return LeRobotTabletopDataConfig(
        repo_id=repo_id,
        assets=_PI05_BASE_ASSETS,
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=False,
        include_mc_return=include_mc_return,
        include_next_obs=include_next_obs,
    )


def _critic_model(*, warmup_ratio, transition_ratio, **kw):
    """Pi0WithCriticConfig with the tuned AlphaFlow recipe (flow_ratio=0.25, λ 0.5/0.5)."""
    return pi0_alphaflow_critic.Pi0WithCriticConfig(
        pi05=True,
        num_train_steps=30_000,
        flow_ratio=0.25,
        lambda_fm=0.5,
        lambda_mf=0.5,
        warmup_ratio=warmup_ratio,
        transition_ratio=transition_ratio,
        **kw,
    )


def _dualyam_data(task, *, include_mc_return=False, include_next_obs=False):
    """DualYamDataConfig for a Challenge expert-data task (pi-adapted, delta actions).

    include_mc_return / include_next_obs add the RFT columns (critic CalQL anchor /
    LPS-RFT chunked-TD windows); the dataset must then carry mc_return / reward.
    """
    return DualYamDataConfig(
        repo_id=f"jellyho/{task}_rl_224",
        base_config=DataConfig(
            prompt_from_task=True,
            # local_files_path=f"/home/yonsei_jell/{task}",
            # local_files_path=f"/data5/jellyho/PFR_RSS/dataset/phase1_merged/{task}",
        ),
        use_delta_joint_actions=True,
        adapt_to_pi=True,
        include_mc_return=include_mc_return,
        include_next_obs=include_next_obs,
    )


def _challenge_rft(task):
    """RFT stage-1 (alphaflow+critic) and stage-2 (LPS-RFT) configs for a Challenge task.

    Mirrors the tabletop phase-1/phase-2 pipeline on DualYam data.  Requires the
    task's dataset to carry mc_return / reward columns (run scripts/compute_mc_returns.py).
    Stage-2's weight_loader points at stage-1's output — update the step if needed.
    """
    return [
        TrainConfig(
            name=f"pi05_rft_phase1_{task}",
            model=_critic_model(warmup_ratio=0.25, transition_ratio=0.75),
            data=_dualyam_data(task, include_mc_return=True),
            weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
            lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
            num_train_steps=100_000,
            batch_size=256,
            num_workers=64,
            save_interval=25_000,
        ),
        TrainConfig(
            name=f"pi05_rft_phase2_{task}",
            model=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True),   # crossq default False
            data=_dualyam_data(task, include_mc_return=True, include_next_obs=True),
            weight_loader=weight_loaders.AlphaFlowWeightLoader(
                f"checkpoints/pi05_rft_phase1_{task}/pi05_rft_phase1_{task}/99999/params"
            ),
            freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
            lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
            num_train_steps=200_000,
            batch_size=1024,
            num_workers=64,
            save_interval=25_000,
        ),
        TrainConfig(
            name=f"pi05_rft_phase2_{task}_mh",
            model=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True, td_horizons=(5, 10, 25, 50)),
            data=_dualyam_data(task, include_mc_return=True, include_next_obs=True),
            weight_loader=weight_loaders.AlphaFlowWeightLoader(
                f"checkpoints/pi05_rft_phase1_{task}/pi05_rft_phase1_{task}/99999/params"
            ),
            freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
            lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
            num_train_steps=200_000,
            batch_size=1024,
            num_workers=64,
            save_interval=25_000,
        ),
    ]


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # ============================================================================
    # Tabletop RFT pipeline (LPS-RFT) — two stages, all on scripts/train.py.
    #   0. Flow-matching baseline    pi05_tabletop / pi05_tabletop_bc (Pi0Config, FM;
    #                                in the Tabletop-Sim section below)
    #   1. Phase 1 (alphaflow+critic) pi05_rft_phase1_{rl,rl_mh,bc}  (from pi05_base:
    #                                alpha-flow distillation + C51 critic)
    #   2. Phase 2 (LPS-RFT)         pi05_rft_phase2_rl*             (from phase-1:
    #                                freeze VLM+action, train critic + latent actor)
    #
    # AlphaFlowWeightLoader loads a base/parent checkpoint and keeps new params
    # (r_mlp, critic/latent experts) at init.  AlphaFlow recipe = tuned
    # (flow_ratio=0.25, lambda 0.5/0.5, large_span_warmup_gate).  Data: rl_orig / bc_orig.
    # ============================================================================
    #
    # --- 1. Flow-matching baseline (plain pi05 FM; no alpha-flow, no critic) -----
    # Defined below in the "Fine-tuning Tabletop-Sim" section (kept there because
    # README/serve examples reference them):
    #   pi05_tabletop      (rl_orig)  <- cat-1 rl: the task-adapted FM policy that
    #   pi05_tabletop_bc   (bc_orig)     cat-3 phase-1 rectifies.
    #
    # --- Phase 1: AlphaFlow distillation + C51 critic (joint, from pi05_base) ----
    # Single-stage: alpha-flow action distillation + critic together, from pi05_base
    # (NOT task-adapted → keeps the 25/50/25 FM warmup).  This is the phase-1
    # checkpoint that LPS-RFT phase-2 loads.  rl = single full-chunk head (50);
    # rl_mh = multi-horizon {5,10,25,50} (per-horizon Q logged as critic/q_h{k}_mean
    # + value_mae_h{k}); bc keeps the default horizons.
    TrainConfig(
        name="pi05_rft_phase1_rl",
        model=_critic_model(warmup_ratio=0.25, transition_ratio=0.75, critic_horizons=(50,)),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=7500,
        batch_size=128,
        num_workers=64,
        save_interval=10000,
    ),
    TrainConfig(
        name="pi05_rft_phase1_rl_mh",
        model=_critic_model(warmup_ratio=0.25, transition_ratio=0.75, critic_horizons=(5, 10, 25, 50)),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=7500,
        batch_size=128,
        num_workers=64,
        save_interval=10000,
    ),
    # Phase-1 WITH the learned KV-compression bottleneck ("middle transformer"):
    # the compressor + action expert + critic co-train (phase-1 is full FT) so the
    # action expert learns to decode from the N=compress_tokens compressed prefix.
    # This is the alignment stage; LPS-RFT phase-2 then freezes (compressor + action)
    # and trains only critic + latent.  (global batch 64, 15k steps.)
    TrainConfig(
        name="pi05_rft_phase1_compressed",
        model=pi0_alphaflow_critic.Pi0WithCriticCompressedConfig(
            pi05=True,
            num_train_steps=30_000,
            flow_ratio=0.25,
            lambda_fm=0.5,
            lambda_mf=0.5,
            warmup_ratio=0.25,
            transition_ratio=0.75,
            critic_horizons=(5, 10, 25, 50),
            compress_tokens=4,
        ),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=30_000,
        batch_size=64,
        num_workers=16,
        save_interval=5_000,
    ),
    # Same as pi05_rft_phase1_compressed but N=2 tokens and global batch 32
    # (fair-comparison variant).
    TrainConfig(
        name="pi05_rft_phase1_compressed_n2",
        model=pi0_alphaflow_critic.Pi0WithCriticCompressedConfig(
            pi05=True,
            num_train_steps=30_000,
            flow_ratio=0.25,
            lambda_fm=0.5,
            lambda_mf=0.5,
            warmup_ratio=0.25,
            transition_ratio=0.75,
            critic_horizons=(5, 10, 25, 50),
            compress_tokens=2,
        ),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=16,
        save_interval=5_000,
    ),
    TrainConfig(
        name="pi05_rft_phase1_bc",
        model=_critic_model(warmup_ratio=0.25, transition_ratio=0.75),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_bc_224", include_mc_return=True),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=7500,
        batch_size=128,
        num_workers=64,
        save_interval=10000,
    ),
    # --- Phase 2: LPS-RFT — offline RL via Latent Policy Steering ---------------
    # Loads a phase-1 (alphaflow+critic) checkpoint — VLM + action + critic — adds a
    # latent actor, freezes VLM + action expert, trains critic + latent actor.
    # Update the weight_loader path to your phase-1 run/step.
    TrainConfig(
        name="pi05_rft_phase2_rl",
        model=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=1024,
        num_workers=64,
        save_interval=25_000,
    ),
    # Multi-horizon Q-chunking: PREDICTION at chunk lengths {5,10,25,50} (per-horizon
    # head), single-state BACKUP from s_{t+H} broadcast to all heads.  Same data as
    # single-Q (one next state at H); per-horizon Q logged as critic/q_data_h{k}_mean.
    TrainConfig(
        name="pi05_rft_phase2_rl_mh",
        model=pi0_lps_rft.Pi0LPSRFTConfig(
            pi05=True, td_horizons=(5, 10, 25, 50)
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(
            "/data5/jellyho/PFR_RSS/openpi-baseline_RLLAB/checkpoints/pi05_alphaflow_critic_rl_mh/pi05_alphaflow_critic_rl_mh/7499/params"
        ),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        save_interval=25_000,
    ),
    # Same as phase2_rl_mh + LPSD: BC term pulling the steered action toward a base
    # generation decode(s, e) (lpsd_alpha · mean‖a_actor − sg(decode(e))‖²), so DDPG
    # can't steer the latent off-distribution (the exploding last/middle chunk steps).
    TrainConfig(
        name="pi05_rft_phase2_rl_mh_lpsd",
        model=pi0_lps_rft.Pi0LPSRFTConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), lpsd_alpha=1.0
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(
            "/data5/jellyho/PFR_RSS/openpi-baseline_RLLAB/checkpoints/pi05_alphaflow_critic_rl_mh/pi05_alphaflow_critic_rl_mh/7499/params"
        ),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        save_interval=25_000,
    ),
    # OOD-steering fixes test: decode clip [-1,1] (default on) + critic ensemble of 4
    # heads (TD target & DDPG actor use min over heads → clipped-double-Q pessimism).
    # Loads base params (critic + latent train from scratch in phase-2, like phase2_rl).
    TrainConfig(
        name="pi05_rft_phase2_rl_mh_ens",
        model=pi0_lps_rft.Pi0LPSRFTConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), n_critics=4
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        save_interval=5000,
    ),
    # All three fixes together: clip + critic ensemble (pessimism) + LPSD (BC to base gen).
    TrainConfig(
        name="pi05_rft_phase2_rl_mh_ens_lpsd",
        model=pi0_lps_rft.Pi0LPSRFTConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), n_critics=4, lpsd_alpha=1.0
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        save_interval=25_000,
    ),
    # Ensemble + EMA target critic (Polyak τ=0.005) for a stabler/faster chunked-TD
    # bootstrap (v_next from the trailing target instead of the online critic).
    TrainConfig(
        name="pi05_rft_phase2_rl_mh_ens_target",
        model=pi0_lps_rft.Pi0LPSRFTConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), n_critics=4, target_tau=0.005
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTConfig(pi05=True).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        save_interval=25_000,
    ),
    # LPS-RFT phase-2 with a learned KV-compression bottleneck ("middle transformer"):
    # a per-layer attention-pool compresses the ~800-token frozen prefix to
    # compress_tokens (N) summary tokens; action/critic/latent attend ONLY to those.
    # The compressor is trainable (added to the freeze filter's trainable set).
    TrainConfig(
        name="pi05_rft_phase2_compressed",
        model=pi0_lps_rft.Pi0LPSRFTCompressedConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), compress_tokens=4
        ),
        data=_tabletop_data(
            "jellyho/aloha_handover_box_joint_pos_rl_224", include_mc_return=True, include_next_obs=True
        ),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=pi0_lps_rft.Pi0LPSRFTCompressedConfig(
            pi05=True, td_horizons=(5, 10, 25, 50), compress_tokens=4
        ).get_freeze_filter(),
        lr_schedule=_optimizer.ConstantSchedule(lr=5e-5),
        num_train_steps=15_000,
        batch_size=64,
        num_workers=16,
        save_interval=5_000,
    ),
    TrainConfig(
        name="pi05_tabletop_bc",
        model=pi0_config.Pi0Config(pi05=True),
        data=_tabletop_data("jellyho/aloha_handover_box_joint_pos_bc"),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE_PARAMS),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=16,
        save_interval=10_000,
    ),
    # Challenge Baseline Examples
    TrainConfig(
        name="pi05_insert-mouse-battery_bc",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("insert-mouse-battery"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000, # 200k is about 3 epochs.
        batch_size=64,
        num_workers=32,
        save_interval=40_000
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap_bc",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("seal-water-bottle-cap"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=64,
        num_workers=32,
        save_interval=40_000
    ),
    TrainConfig(
        name="pi05_tower-of-hanoi-game_bc",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("tower-of-hanoi-game"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=64,
        num_workers=32,
        save_interval=40_000
    ),
    # BC on the three DualYam tasks merged into one dataset
    # (/home/yonsei_jell/dualyam_combined; insert-mouse-battery + seal-water-bottle-cap
    #  + tower-of-hanoi-game, 3205 episodes / 3 tasks).  prompt_from_task keeps each
    # episode's own task string, so a single policy learns all three.
    TrainConfig(
        name="pi05_generalist_bc_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="jellyho/phase1_combined",
            base_config=DataConfig(
                prompt_from_task=True,
                local_files_path="/home/yonsei_jell/dualyam_combined",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/NHNHOME/WORKSPACE/0526040008_A/jellyho/ckpts_rss/pi05_rss2026_multitask/199999/params"),
        num_train_steps=200_000,
        batch_size=128,
        num_workers=32,
        save_interval=25_000,
    ),
    # Fine-tune pi05 BC from a SAVED checkpoint (not pi05_base): inits from the trained
    # pi05_tabletop_bc/my_run/29999 params (fresh run, step 0, fresh optimizer).  Point
    # `weight_loader` at any `<checkpoints>/<config>/<exp>/<step>/params`, and `data` at
    # the dataset to fine-tune on.
    TrainConfig(
        name="pi05_seal-water-bottle-cap_bc_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("seal-water-bottle-cap"),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data5/jellyho/PFR_RSS/checkpoints/rss_ckpt/pi05_seal-water-bottle-cap/199999/params"
        ),
        num_train_steps=100_000,
        batch_size=256,
        num_workers=16,
        save_interval=25_000,
    ),
    TrainConfig(
        name="pi05_insert-mouse-battery_bc_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("insert-mouse-battery"),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data5/jellyho/PFR_RSS/checkpoints/rss_ckpt/pi05_insert-mouse-battery/199999/params"
        ),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=16,
        save_interval=25_000,
    ),
    TrainConfig(
        name="pi05_tower-of-hanoi-game_bc_ft",
        model=pi0_config.Pi0Config(pi05=True),
        data=_dualyam_data("tower-of-hanoi-game"),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data5/jellyho/PFR_RSS/checkpoints/rss_ckpt/pi05_tower-of-hanoi-game/199999/params"
        ),
        num_train_steps=100_000,
        batch_size=128,
        num_workers=16,
        save_interval=25_000,
    ),
    
    # ── RL Token bottleneck (arXiv:2604.23073) ───────────────────────────────
    # Train the encoder–decoder RL-token bottleneck on top of a FROZEN, task-
    # finetuned pi05 policy.  Only the rlt_* params train (VLA + action expert
    # frozen); the objective is autoregressive reconstruction of the VLA's prefix
    # embeddings (+ proprio) through the compact RL token.  AlphaFlowWeightLoader
    # loads the overlapping pi05 weights and keeps the new rlt_* params at init.
    TrainConfig(
        name="pi05_seal-water-bottle-cap_rlt",
        model=pi0_rlt.Pi0RLTConfig(pi05=True),
        data=_dualyam_data("seal-water-bottle-cap"),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(
            "/data5/jellyho/PFR_RSS/openpi-baseline_RLLAB/checkpoints/pi05_seal-water-bottle-cap_bc_ft/pi05_seal-water-bottle-cap_bc_ft/99999/params"
        ),
        lr_schedule=_optimizer.ConstantSchedule(lr=1e-4),
        num_train_steps=200_000,
        batch_size=128,
        num_workers=32,
        save_interval=20_000,
    ),
    TrainConfig(
        name="pi05_insert-mouse-battery_rlt",
        model=pi0_rlt.Pi0RLTConfig(pi05=True),
        data=_dualyam_data("insert-mouse-battery"),
        weight_loader=weight_loaders.AlphaFlowWeightLoader(
            "/home/yonsei_jell/openpi-baseline_RLLAB/checkpoints/pi05_insert-mouse-battery_bc_ft/pi05_insert-mouse-battery_bc_ft/99999/params"
        ),
        lr_schedule=_optimizer.ConstantSchedule(lr=1e-4),
        num_train_steps=100_000,
        batch_size=256,
        num_workers=64,
        save_interval=20_000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
