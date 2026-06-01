import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

    # Number of sampling steps (NFE) passed to model.sample_actions.
    # Alpha-flow: 1 = 1-NFE MeanFlow (production), N>1 = N-step integration.
    num_steps: int | None = None
    # Alpha-flow integration mode: "mean" (MeanFlow, r=t_next) or "fm"
    # (instantaneous velocity, r=t — for base-flow eval before MeanFlow training).
    nfe_mode: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Optional path to a directory containing pre-computed norm stats (e.g. ./assets/pi05_tabletop).
    # Useful when serving a base model checkpoint (e.g. pi05_base) with norm stats computed from a
    # different dataset, without needing a fully trained fine-tuned checkpoint.
    # The directory must contain <asset_id>/norm_stats.json (same layout as assets/ produced by
    # compute_norm_stats.py). If not set, norm stats are loaded from inside the checkpoint directory.
    norm_stats_dir: str | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    norm_stats = None
    if args.norm_stats_dir is not None:
        train_config = _config.get_config(args.policy.config) if isinstance(args.policy, Checkpoint) else None
        if train_config is not None:
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
            norm_stats = _checkpoints.load_norm_stats(args.norm_stats_dir, data_config.asset_id)
            logging.info("Loaded norm stats from %s (asset_id=%s)", args.norm_stats_dir, data_config.asset_id)

    match args.policy:
        case Checkpoint():
            sample_kwargs = {}
            if args.policy.num_steps is not None:
                sample_kwargs["num_steps"] = args.policy.num_steps
            if args.policy.nfe_mode is not None:
                sample_kwargs["nfe_mode"] = args.policy.nfe_mode
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                norm_stats=norm_stats,
                sample_kwargs=sample_kwargs or None,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
