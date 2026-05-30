"""
Alpha-Flow fine-tuning script for Pi05.

Fine-tunes a pretrained Pi05 checkpoint with the alpha-Flow curriculum
to produce a 1-NFE policy.

Training has three phases (paper Algorithm 1):
    Phase 1 (alpha = 1)   → Trajectory Flow Matching warmup
    Phase 2 (0 < alpha < 1) → Discrete alpha-Flow transition
    Phase 3 (alpha = 0)   → Exact MeanFlow via JVP

Phases 1-2 share the same JIT-compiled train step (discrete).
Phase 3 uses a separate JIT-compiled train step (JVP) to avoid paying
the JVP cost during phases 1-2.  Switching is done at the Python level.

Usage:
    uv run scripts/train_alphaflow.py <config_name> \\
        --pretrained-checkpoint <path/to/pi05/params> \\
        --exp-name <run_name>

    # Resume
    uv run scripts/train_alphaflow.py <config_name> \\
        --pretrained-checkpoint <path/to/pi05/params> \\
        --exp-name <run_name> --resume
"""

import argparse
import dataclasses
import functools
import logging
import platform
import sys

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from openpi.models.pi0_alphaflow import (
    Pi0AlphaFlowConfig,
    alpha_schedule_python,
)


# ---------------------------------------------------------------------------
# Logging / WandB
# ---------------------------------------------------------------------------

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# ---------------------------------------------------------------------------
# Weight loader: Pi05 checkpoint → Pi0AlphaFlow (r_proj stays at zero-init)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AlphaFlowWeightLoader(_weight_loaders.WeightLoader):
    """
    Loads a Pi05 checkpoint into Pi0AlphaFlow.

    Params present in the checkpoint are copied over.
    New params (r_proj) keep their zero-initialized values by returning a
    jax.ShapeDtypeStruct placeholder — the same pattern used for LoRA weights.
    """
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(
            _weight_loaders.download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        flat_ref    = traverse_util.flatten_dict(params, sep="/")
        flat_loaded = traverse_util.flatten_dict(loaded, sep="/")

        result = {}
        for k, v in flat_loaded.items():
            if k in flat_ref:
                ref_dtype = flat_ref[k].dtype
                result[k] = v.astype(ref_dtype) if v.dtype != ref_dtype else v

        new_keys = set(flat_ref) - set(result)
        if new_keys:
            logging.info(
                "AlphaFlowWeightLoader: %d new param(s) kept at zero-init: %s",
                len(new_keys), sorted(new_keys),
            )
        for k in new_keys:
            result[k] = flat_ref[k]   # ShapeDtypeStruct → filtered out later

        return traverse_util.unflatten_dict(result, sep="/")


# ---------------------------------------------------------------------------
# Train state init
# ---------------------------------------------------------------------------

def _load_weights_and_validate(loader, params_shape):
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items()
         if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, object]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


# ---------------------------------------------------------------------------
# Shared gradient update logic
# ---------------------------------------------------------------------------

def _apply_grads(config, state, model, grads, loss, alpha_val):
    """Apply gradients and return (new_state, info)."""
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "alpha": alpha_val,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# ---------------------------------------------------------------------------
# Train step — Phase 1 & 2: discrete alpha-Flow  (alpha > 0)
#
# alpha is passed as a plain jnp.float32 scalar from the Python training loop.
# JIT traces once for the (float32,) shape — no re-compilation across steps.
# ---------------------------------------------------------------------------

@at.typecheck
def train_step_discrete(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    alpha,   # jnp.float32 scalar, shape ()
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, obs, actions):
        return jnp.mean(model.compute_alphaflow_loss(rng, obs, actions, alpha, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    obs, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, obs, actions)

    return _apply_grads(config, state, model, grads, loss, alpha)


# ---------------------------------------------------------------------------
# Train step — Phase 3: exact MeanFlow via JVP  (alpha = 0)
#
# Compiled separately so the JVP overhead only affects phase 3.
# ---------------------------------------------------------------------------

@at.typecheck
def train_step_jvp(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, obs, actions):
        return jnp.mean(model.compute_jvp_meanflow_loss(rng, obs, actions, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    obs, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, obs, actions)

    return _apply_grads(config, state, model, grads, loss, jnp.float32(0.0))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(config: _config.TrainConfig, pretrained_checkpoint: str):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding       = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter   = iter(data_loader)
    batch       = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    if not resuming:
        config = dataclasses.replace(
            config,
            weight_loader=AlphaFlowWeightLoader(params_path=pretrained_checkpoint),
        )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    af_cfg: Pi0AlphaFlowConfig = config.model  # type: ignore[assignment]

    # Convert ratio-based schedule to absolute steps.
    warmup_end     = int(af_cfg.warmup_ratio     * config.num_train_steps)
    transition_end = int(af_cfg.transition_ratio * config.num_train_steps)
    logging.info(
        "Alpha-Flow schedule: warmup_end=%d, transition_end=%d (of %d total steps)",
        warmup_end, transition_end, config.num_train_steps,
    )

    # Two separately compiled train steps.
    # discrete: used for alpha > 0 (phases 1 & 2); alpha is a traced jnp.float32.
    # jvp:      used for alpha = 0 (phase 3); no alpha arg, different loss.
    ptrain_step_discrete = jax.jit(
        functools.partial(train_step_discrete, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    ptrain_step_jvp = jax.jit(
        functools.partial(train_step_jvp, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # Python-level alpha — determines which compiled step to run.
        # Avoids paying JVP cost during phases 1 & 2.
        alpha_py = alpha_schedule_python(
            step,
            warmup_end=warmup_end,
            transition_end=transition_end,
            gamma=af_cfg.alpha_gamma,
            eta=af_cfg.alpha_min,
        )

        with sharding.set_mesh(mesh):
            if alpha_py == 0.0:
                # Phase 3: exact MeanFlow via JVP
                train_state, info = ptrain_step_jvp(train_rng, train_state, batch)
            else:
                # Phase 1 & 2: discrete alpha-Flow
                train_state, info = ptrain_step_discrete(
                    train_rng, train_state, batch, jnp.float32(alpha_py)
                )

        infos.append(info)
        if step % config.log_interval == 0:
            stacked  = common_utils.stack_forest(infos)
            reduced  = jax.device_get(jax.tree.map(jnp.mean, stacked))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pretrained-checkpoint", required=True,
                        help="Path to the Pi05 checkpoint params directory.")
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    train_config = _config.cli()

    main(train_config, pretrained_checkpoint=known.pretrained_checkpoint)
