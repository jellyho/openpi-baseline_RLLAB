"""Stage 3 — adaptive-Q-chunking inference for a merged RLT+critic bundle.

Loads a bundle built by ``openpi.rlt_critic.merge`` and exposes a single ``sample_actions``
that does the ACSAC adaptive replanning at deployment:

    1. RLT model samples N base-action chunks + the RL token z_rl for the current obs
       (Pi0RLTJoint: one backbone forward; vanilla Pi0RLT: token + sampling separately).
    2. decode the N chunks (normalized model space) -> raw action space (the SAME space the
       critic trained on: the stored ``base_action`` column).
    3. the prefix-critic scores every (candidate n, prefix length h): Q(z_rl, a^(n)_{1:h}),
       ensemble-min over K.
    4. joint arg-max over (n, h) -> (n*, h*): pick the best chunk AND how many steps to commit.
    5. return that chunk under one of two execution modes:
         - 'truncate'      : execute only the first h* steps, then replan (chunk[:h*]).
         - 'absolute_hold' : keep the full-H chunk but hold the h*-th (absolute) action for the
                             tail (chunk[h*:] = chunk[h*-1]) — effectively executing h* steps
                             when the policy outputs absolute joint targets.

NON-BREAKING: this is a standalone wrapper. It does not modify ``Pi0``/``Pi0RLT``/``Policy``
or any existing checkpoint; it composes them. Pass an ``AQCAdaptive`` instance to a
``Policy`` (it has ``sample_actions`` / ``predict_value``), or call it directly.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import flax.serialization as fs
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.rlt_critic.distributional import hl_gauss_transform
from openpi.rlt_critic.transformer import PrefixValue


def _build_action_decoder(data_config):
    """normalized-model-space -> raw-action-space converter for sampled chunks.

    Mirrors ``scripts/compute_rl_tokens._build_action_decoder`` so inference decodes base
    actions to the EXACT space the stored ``base_action`` (hence the critic) used:
    Unnormalize -> AbsoluteActions (state-dependent) -> remaining row-wise output transforms
    (e.g. YamOutputs: slice to 14 dims). Returns ``decode(sampled[B,N,H,Dm], state[B,Dm])``.
    """
    if data_config.norm_stats is None:
        return None
    unnorm = _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    outs = list(data_config.data_transforms.outputs)

    def decode(sampled: np.ndarray, state: np.ndarray) -> np.ndarray:
        b, n, h, dm = sampled.shape
        acts = sampled.reshape(b * n, h, dm).astype(np.float32)
        st = np.repeat(state.astype(np.float32), n, axis=0)
        d = unnorm({"state": st, "actions": acts})
        i = 0
        while i < len(outs) and isinstance(outs[i], _transforms.AbsoluteActions):
            d = outs[i](d)
            i += 1
        flat = d["actions"].reshape(-1, dm)
        for t in outs[i:]:
            flat = t({"actions": flat})["actions"]
        return flat.reshape(b, n, h, -1)

    return decode


class AQCAdaptive:
    """Adaptive-Q-chunking policy: RLT actor + prefix critic, joint (n*, h*) selection."""

    def __init__(self, *, model, critic_net, critic_params, from_probs,
                 macro_group_size, horizon, action_dim, num_action_samples,
                 num_flow_steps, decode, rlt_config_name):
        self.model = model
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)          # raw critic action dim (e.g. 14)
        self.macro_group_size = int(macro_group_size)
        self.macro_H = self.horizon // self.macro_group_size
        self.num_action_samples = int(num_action_samples)
        self.num_flow_steps = int(num_flow_steps)
        self.decode = decode                        # may be None (no norm_stats)
        self.rlt_config_name = rlt_config_name
        self._joint = hasattr(model, "extract_token_and_base_actions")

        # jit the (frozen) RLT extractors.
        if self._joint:
            self._extract_both = jax.jit(
                lambda rng, obs, n: model.extract_token_and_base_actions(
                    rng, obs, num_samples=n, num_steps=self.num_flow_steps),
                static_argnums=(2,))
        else:
            self._extract_token = jax.jit(lambda obs: model.extract_rl_token(obs))
            self._sample_base = jax.jit(
                lambda rng, obs, n: model.sample_base_actions(
                    rng, obs, num_samples=n, num_steps=self.num_flow_steps),
                static_argnums=(2,))

        # jit the critic forward -> expected scalar prefix values (always distributional here).
        def _prefix_values(obs2d, act2d):
            logits = critic_net.apply(critic_params, obs2d, act2d)   # (K, M, macro_H, atoms)
            return from_probs(jax.nn.softmax(logits, axis=-1))       # (K, M, macro_H)
        self._critic = jax.jit(_prefix_values)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, bundle_dir, data_config=None):
        bundle = pathlib.Path(bundle_dir).resolve()
        manifest = json.loads((bundle / "aqc_manifest.json").read_text())
        net_hp = json.loads((bundle / "critic" / "net.json").read_text())

        # RLT model (nnx, orbax params).
        tcfg = _config.get_config(manifest["rlt_config_name"])
        params = _model.restore_params(bundle / "params")
        model = tcfg.model.load(params)
        model.eval()

        # critic net (linen) + params (msgpack) + HL-Gauss transform.
        net = PrefixValue(
            action_dim=net_hp["action_dim"], horizon=net_hp["horizon"],
            num_ensembles=net_hp["num_ensembles"], num_layers=net_hp["num_layers"],
            num_heads=net_hp["num_heads"], head_dim=net_hp["head_dim"],
            mlp_dim=net_hp["mlp_dim"], layer_norm=net_hp["layer_norm"],
            num_atoms=net_hp["num_atoms"], per_position_head=net_hp["per_position_head"],
            state_encoder_dims=tuple(net_hp["state_encoder_dims"]),
            macro_group_size=net_hp["macro_group_size"])
        ex_obs = jnp.zeros((1, net_hp["latent_dim"]))
        ex_act = jnp.zeros((1, net_hp["horizon"] * net_hp["action_dim"]))
        template = net.init(jax.random.PRNGKey(0), ex_obs, ex_act)
        critic_params = fs.from_bytes(template, (bundle / "critic" / "params.msgpack").read_bytes())
        sigma = net_hp["hl_gauss_sigma_frac"] * (net_hp["v_max"] - net_hp["v_min"]) / net_hp["num_atoms"]
        _, from_probs = hl_gauss_transform(net_hp["v_min"], net_hp["v_max"], net_hp["num_atoms"], sigma)

        # action decoder (normalized -> raw); built from the RLT config's data pipeline if not given.
        if data_config is None:
            data_config = tcfg.data.create(tcfg.assets_dirs, tcfg.model)
        decode = _build_action_decoder(data_config)
        if decode is None:
            print("[AQCAdaptive] WARNING: no norm_stats -> base actions cannot be decoded to the "
                  "critic's raw space. sample_actions will fall back to a shape-only passthrough "
                  "(slice first action_dim) -- VALUES ARE WRONG, for smoke tests only.")

        return cls(model=model, critic_net=net, critic_params=critic_params, from_probs=from_probs,
                   macro_group_size=net_hp["macro_group_size"], horizon=net_hp["horizon"],
                   action_dim=net_hp["action_dim"], num_action_samples=manifest["num_action_samples"],
                   num_flow_steps=manifest["num_flow_steps"], decode=decode,
                   rlt_config_name=manifest["rlt_config_name"])

    # ------------------------------------------------------------------ core
    def _propose(self, rng, observation, num_samples):
        """RLT forward -> (z_rl [1,L], base [1,N,H,Dm-model])."""
        if self._joint:
            z_rl, base = self._extract_both(rng, observation, num_samples)
        else:
            z_rl = self._extract_token(observation)
            base = self._sample_base(rng, observation, num_samples)
        return z_rl, base

    def _score(self, z_rl, base_raw):
        """Critic prefix-Q for every candidate; ensemble-min. base_raw [N,H,Dr] -> q [N, macro_H]."""
        n = base_raw.shape[0]
        states = jnp.broadcast_to(jnp.asarray(z_rl[0]), (n, z_rl.shape[-1]))      # (N, L)
        acts = jnp.asarray(base_raw).reshape(n, -1)                              # (N, H*Dr)
        q = self._critic(states, acts)                                           # (K, N, macro_H)
        return np.asarray(q.min(axis=0))                                         # (N, macro_H)

    def sample_actions(self, rng, observation, *, exec_mode="truncate", num_samples=None):
        """Adaptive (n*, h*) selection. Returns a dict with the chosen chunk + diagnostics.

        exec_mode:
          'truncate'      -> 'actions' is the first h* steps  [h*, Dm]  (variable length).
          'absolute_hold' -> 'actions' is full-H with the tail held at the h*-th action
                             [H, Dm]  (execute h* effective steps with absolute targets).
        """
        n = int(num_samples or self.num_action_samples)
        z_rl, base = self._propose(rng, observation, n)               # base [1,N,H,Dm]
        base_np = np.asarray(base)
        state_np = np.asarray(observation.state)

        if self.decode is not None:
            base_raw = self.decode(base_np, state_np)[0]              # [N,H,Dr]
        else:                                                        # smoke-only fallback
            base_raw = base_np[0, :, :, : self.action_dim]

        q = self._score(z_rl, base_raw)                              # [N, macro_H]
        flat = int(np.argmax(q))
        n_star, mh = divmod(flat, self.macro_H)
        h_star = (mh + 1) * self.macro_group_size                    # steps to execute (1..H)

        # Return the chosen chunk in RAW action space (decoded above) — i.e. the absolute joint
        # targets the robot consumes. exec_mode shapes the tail:
        chosen = np.asarray(base_raw[n_star])                        # [H, Dr] raw
        if exec_mode == "truncate":
            executed = chosen[:h_star]                               # [h*, Dr] -> execute, then replan
        elif exec_mode == "absolute_hold":
            executed = chosen.copy()
            executed[h_star:] = chosen[h_star - 1]                   # [H, Dr] -> hold the h*-th absolute target
        else:
            raise ValueError(f"unknown exec_mode {exec_mode!r}")

        return {
            "actions": executed,                       # raw absolute joint targets (exec_mode applied)
            "full_chunk": chosen,                      # [H, Dr] raw chosen candidate (untrimmed)
            "normalized_chunk": np.asarray(base_np[0, n_star]),  # [H, Dm] model space
            "h_star": int(h_star),
            "n_star": int(n_star),
            "q_by_h": q.max(axis=0),                   # best candidate value at each prefix length
            "q_best": float(q[n_star, mh]),
        }

    def predict_value(self, z_rl, base_raw):
        """Prefix-Q [N, macro_H] for given candidates (z_rl [1,L], base_raw [N,H,Dr])."""
        return self._score(z_rl, np.asarray(base_raw))


# ===========================================================================
# Deployable policy: raw obs dict -> adaptive action (openpi serving interface)
# ===========================================================================

class AQCPolicy(_base_policy.BasePolicy):
    """BasePolicy wrapper: applies the RLT config's INPUT transforms to a raw obs dict,
    runs the adaptive (n*, h*) selection, and returns the executed chunk in raw action space.

    ``infer(obs)`` -> ``{actions, h_star, n_star, q_by_h, policy_timing}``. The runtime should
    execute ``h_star`` steps (``truncate``) — or all H with the tail held (``absolute_hold``) —
    then call ``infer`` again to replan. Drop-in for the openpi websocket policy server.
    """

    def __init__(self, ada: "AQCAdaptive", input_transform, *,
                 exec_mode: str = "truncate", num_samples=None, rng=None, metadata=None):
        self._ada = ada
        self._input = input_transform
        self._exec_mode = exec_mode
        self._num_samples = num_samples
        self._rng = rng if rng is not None else jax.random.key(0)
        self._metadata = metadata or {}

    @property
    def metadata(self):
        return self._metadata

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        inputs = jax.tree.map(lambda x: x, obs)            # copy (transforms may mutate)
        inputs = self._input(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)  # add batch dim
        self._rng, rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        t0 = time.monotonic()
        out = self._ada.sample_actions(rng, observation, exec_mode=self._exec_mode,
                                       num_samples=self._num_samples)
        dt_ms = (time.monotonic() - t0) * 1000.0
        return {
            "actions": np.asarray(out["actions"]),         # raw absolute joint targets
            "h_star": out["h_star"],
            "n_star": out["n_star"],
            "q_by_h": np.asarray(out["q_by_h"]),
            "policy_timing": {"infer_ms": dt_ms},
        }


def create_aqc_policy(
    bundle_dir,
    *,
    exec_mode: str = "truncate",
    default_prompt: str | None = None,
    norm_stats=None,
    num_samples: int | None = None,
    rng=None,
) -> AQCPolicy:
    """Build a deployable adaptive-Q-chunking policy from a merged bundle.

    Mirrors ``policy_config.create_trained_policy``: the INPUT transforms come from the RLT
    config's data pipeline (repack/data/normalize/model transforms); the OUTPUT decode is
    folded into ``AQCAdaptive`` (its ``decode`` already maps sampled chunks to raw action
    space, which is what the critic scored and what the robot executes). norm_stats are loaded
    from the RLT checkpoint that produced the bundle, so deployment uses the SAME stats as
    training/annotation.
    """
    bundle = pathlib.Path(bundle_dir).resolve()
    manifest = json.loads((bundle / "aqc_manifest.json").read_text())
    tcfg = _config.get_config(manifest["rlt_config_name"])
    data_config = tcfg.data.create(tcfg.assets_dirs, tcfg.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("asset_id required to load norm stats for the AQC policy")
        # The bundle's params/ is self-contained: for a step-dir RLT it symlinks
        # <step>/params (carrying <asset_id>/norm_stats.json from save_state); for a flat
        # orbax store it symlinks the store itself (norm_stats.json at its root). Either
        # way load_norm_stats finds it under bundle/params. Fall back to the RLT
        # checkpoint dir + its parent for older/edge layouts.
        rlt_params = pathlib.Path(manifest["rlt_params"]).resolve()
        for base in (bundle / "params", rlt_params, rlt_params.parent):
            try:
                norm_stats = _checkpoints.load_norm_stats(base, data_config.asset_id)
                break
            except FileNotFoundError:
                continue
        if norm_stats is None:
            raise FileNotFoundError(
                f"norm stats not found for AQC bundle {bundle} (asset_id={data_config.asset_id}); "
                f"searched bundle/params, {rlt_params}, {rlt_params.parent}")
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    ada = AQCAdaptive.load(bundle, data_config=data_config)
    input_transform = _transforms.compose([
        _transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    return AQCPolicy(ada, input_transform, exec_mode=exec_mode, num_samples=num_samples,
                     rng=rng, metadata=tcfg.policy_metadata)
