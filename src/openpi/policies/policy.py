from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        num_action_samples: int = 0,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # Optional extras (JAX critic models), both for eval / debug overlays:
        #   predict_value(obs, action)        -> E[V](s, sampled_action)
        #   sample_random_actions(rng, obs)   -> base-policy chunk from a random sphere latent
        self._predict_value = None
        self._num_action_samples = num_action_samples
        self._sample_random = None
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            if hasattr(model, "predict_value"):
                self._predict_value = nnx_utils.module_jit(model.predict_value)
            if num_action_samples > 0 and hasattr(model, "sample_random_actions"):
                self._sample_random = nnx_utils.module_jit(model.sample_random_actions)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        random_rng = None
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device, random_rng = jax.random.split(self._rng, 3)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        # critic E[V](s, sampled chunk).  Kept OUT of `outputs` so the output
        # transform (which rebuilds a fresh {"actions": ...} dict) can't drop it;
        # re-attached after.  [b, K] for multi-horizon critics → full-chunk head.
        value = (
            self._predict_value(observation, outputs["actions"])
            if self._predict_value is not None else None
        )
        # Optional base-policy action samples at the current state (N random sphere
        # latents) — for debug overlays comparing the steered chunk against the cloud
        # the policy *could* generate.  Same model (normalized) space as the steered
        # chunk; kept OUT of `outputs` (see value note) and re-attached after.
        action_samples = None
        if self._sample_random is not None:
            obs_n = jax.tree.map(
                lambda x: jnp.repeat(x, self._num_action_samples, axis=0), observation
            )
            action_samples = self._sample_random(random_rng, obs_n)  # [N, ah, ad]
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            if value is not None:
                value = np.asarray(value[0, ...].detach().cpu())
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            if value is not None:
                value = np.asarray(value[0, ...])
            if action_samples is not None:
                action_samples = np.asarray(action_samples)  # [N, ah, ad]

        # Un-normalize each base-policy sample through the SAME output transform as the
        # executed chunk, so action_samples land in real (e.g. joint) space — matching
        # outputs["actions"] (the steered chunk) for image-space FK / projection.
        action_samples_real = None
        if action_samples is not None:
            action_samples_real = np.stack(
                [self._output_transform({**outputs, "actions": a})["actions"] for a in action_samples],
                axis=0,
            )

        outputs = self._output_transform(outputs)
        if value is not None:
            outputs["value"] = value
        if action_samples_real is not None:
            # [N, ah, action_dim], same real space as outputs["actions"] (steered).
            outputs["action_samples"] = action_samples_real
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
