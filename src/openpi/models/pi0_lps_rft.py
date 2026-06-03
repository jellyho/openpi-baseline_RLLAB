"""
Pi0 LPS-RFT — Latent Policy Steering for offline reinforced fine-tuning.

Extends Pi0WithCritic with a third task expert: a **latent actor** that, given
the state, outputs a deterministic latent z on the sphere.  The frozen 1-NFE
action expert decodes a = decode(s, z), and the critic scores Q(s, a).  Training
steers the frozen policy toward high-value actions by moving only the latent
actor (and the critic), keeping the VLM backbone and the action decoder frozen.

Architecture
────────────
    PaliGemma (VLM, FROZEN)
    ├─ Action Expert (FROZEN)  ── 1-NFE decoder: a = decode(s, z)
    ├─ Critic Expert (train)   ── Q(s, a) distributional value (C51 / HL-Gauss)
    └─ Latent Actor  (train)   ── z(s): deterministic latent on the sphere

Gemma `configs` list has 4 entries: [paligemma, action, critic, latent]
(index 0 is the VLM backbone itself).

Inheritance
───────────
    Pi0AlphaFlow → Pi0WithCritic → Pi0LPSRFT

Loss (chunked-TD critic + latent actor)
───────────────────────────────────────
    critic: HL-Gauss C51 cross-entropy vs a CalQL-anchored, done-masked chunked
            TD target  y = done ? G_t : max(R_chunk + γ^H·Q(s', a'), G_t).
            R_chunk = Σ_i γ^i·reward[t+i] (dataset reward window); Q(s', a') is
            scored in the SAME batch (CrossQ, no target net) and stop-gradient'd;
            mc_return G_t is used only as the CalQL anchor / terminal target.
    actor : maximize E[V](s, decode(s, z(s)))  →  steers the latent toward
            high-value regions (critic params detached, DDPG-style).
"""

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0_alphaflow_critic import (
    Pi0WithCritic, Pi0WithCriticConfig, critic_loss_hl_gauss, expected_value,
)
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.shared.nnx_utils as nnx_utils
from openpi.shared import array_typing as at


# Selects the critic parameters (expert _2 + critic projections) — used to
# detach them in the DDPG actor term.
_CRITIC_FILTER = nnx.Any(
    nnx_utils.PathRegex(".*_2.*"),
    nnx_utils.PathRegex(".*critic_.*"),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0LPSRFTConfig(Pi0WithCriticConfig):
    """
    Config for Pi0LPSRFT.

    Extends Pi0WithCriticConfig with a latent-actor expert.  The VLM backbone
    and the action expert are frozen; only the critic and latent actor train.
    Initialize from a pi05_alphaflow_critic checkpoint (the latent expert starts
    from scratch).
    """

    # Latent actor backbone — must share head_dim / num_heads / num_kv_heads / depth
    # with the other experts (shared attention).  The latent actor just maps a query
    # token → latent z, so a small expert suffices.  gemma_50m (~48M) by default;
    # gemma_30m (~28M) is also available.
    latent_expert_variant: _gemma.Variant = "gemma_50m"

    # Loss weights.
    actor_loss_weight:  float = 1.0
    # critic_loss_weight inherited (default 1.0).

    # Discount factor (stored for the v2 chunked-TD term).
    gamma: float = 0.995

    # ── RL ablation toggles (config-isolatable) ───────────────────────────────
    # CrossQ joint batch: process current s and next s' in ONE forward (concat →
    # split).  Mathematically identical to two separate forwards (no BatchNorm),
    # but ~2x cheaper to compile/run.  Default False = separate forwards (baseline).
    crossq_joint_batch:   bool = False
    # Normalize the DDPG actor loss by its own (detached) magnitude → scale ±1,
    # decoupling the actor step from the absolute Q-value scale.  False = raw -Q.
    normalize_actor_loss: bool = True

    # Multi-horizon PREDICTION, single-state BACKUP.  The critic predicts a value
    # at several chunk lengths k (number of committed actions, 1..action_horizon):
    # Q_k(s, a_{0:k}).  The TD backup is computed ONCE from the state after the
    # FULL chunk (s_{t+H}) — y = max(R_chunk + γ^H·V(s_{t+H}), G_t) — and the SAME
    # target supervises every horizon head (all estimate V(s_t)).  So this needs
    # only the single next state at H (same data as the single-Q path).
    # Empty tuple = single chunk-level Q (current behaviour).  The last horizon
    # must equal action_horizon (the full chunk); the actor maximizes that Q.
    td_horizons: tuple[int, ...] = ()

    # Which critic head(s) the DDPG actor maximizes (multi-horizon only).  Each
    # value must be one of td_horizons.  The actor loss is -mean over the listed
    # heads' Q (so a single head → maximize that head; several → maximize their
    # mean).  Empty → the full chunk (last td_horizon = action_horizon), i.e. the
    # deployed quantity (current behaviour).
    actor_horizons: tuple[int, ...] = ()

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0LPSRFT":
        return Pi0LPSRFT(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze everything except the critic and latent-actor params.

        Trainable: critic expert (`_2`) + critic projections, latent expert
        (`_3`) + latent projections.  Frozen: SigLIP, PaliGemma (expert 0),
        action expert (`_1`) + action projections.
        """
        trainable = nnx.Any(
            nnx_utils.PathRegex(".*_2.*"),          # critic expert (gemma suffix _2)
            nnx_utils.PathRegex(".*_3.*"),          # latent expert (gemma suffix _3)
            nnx_utils.PathRegex(".*critic_.*"),     # critic_in_proj / critic_out_proj
            nnx_utils.PathRegex(".*latent_.*"),     # latent_query / latent_out_proj
        )
        return nnx.Not(trainable)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Pi0LPSRFT(Pi0WithCritic):
    """
    Pi0WithCritic + latent actor (4-expert Gemma).

    Overrides
    ─────────
    __init__              — 4-expert Gemma + latent query/projection
    _experts_prefix/action/_adarms_action — length-4 expert token lists
    _critic_logits        — length-4 critic token list (overrides parent's len-3)
    compute_loss          — offline RL: MC critic + latent-actor value steering

    Inherits unchanged
    ──────────────────
    _embed_prefix_kv, _action_velocity, sample_actions_1nfe, embed_critic_suffix,
    predict_value, critic utilities.
    """

    def __init__(self, config: Pi0LPSRFTConfig, rngs: nnx.Rngs):
        # Rebuild everything with 4 experts (bypass parent 3-expert __init__).
        _model.BaseModel.__init__(
            self, config.action_dim, config.action_horizon, config.max_token_len
        )
        self.pi05 = True

        paligemma_cfg     = _gemma.get_config(config.paligemma_variant)
        action_expert_cfg = _gemma.get_config(config.action_expert_variant)
        critic_expert_cfg = _gemma.get_config(config.critic_expert_variant)
        latent_expert_cfg = _gemma.get_config(config.latent_expert_variant)

        # ── 4-expert Gemma ──────────────────────────────────────────────────
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_cfg, action_expert_cfg, critic_expert_cfg, latent_expert_cfg],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True, False, False],  # only action expert uses adaRMS
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_cfg.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ── Action expert (frozen at train time) ────────────────────────────
        self.action_in_proj  = nnx.Linear(config.action_dim, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_in     = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_out    = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_cfg.width, config.action_dim, rngs=rngs)
        # r conditioning: full 2-layer MLP (mirrors time_mlp), output layer zero-init.
        self.r_mlp_in  = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.r_mlp_out = nnx.Linear(
            action_expert_cfg.width, action_expert_cfg.width, rngs=rngs,
            kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros,
        )

        # ── Critic expert ────────────────────────────────────────────────────
        self.critic_in_proj  = nnx.Linear(config.action_dim, critic_expert_cfg.width, rngs=rngs)
        self.critic_out_proj = nnx.Linear(critic_expert_cfg.width, config.n_bins, rngs=rngs)

        # ── Latent actor expert ──────────────────────────────────────────────
        # Learned query tokens (one per action-horizon step) that attend to the
        # prefix and produce the latent.  latent_out_proj maps to action_dim so
        # the latent z has the same shape as the decoder's noise input.
        self.latent_query = nnx.Param(
            jax.random.normal(rngs.params(), (config.action_horizon, latent_expert_cfg.width)) * 0.02
        )
        self.latent_out_proj = nnx.Linear(latent_expert_cfg.width, config.action_dim, rngs=rngs)

        # C51 / schedule params.
        self.n_bins             = config.n_bins
        self.v_min              = config.v_min
        self.v_max              = config.v_max
        self.hl_gauss_sigma     = config.hl_gauss_sigma
        self.critic_loss_weight = config.critic_loss_weight
        self.actor_loss_weight  = config.actor_loss_weight
        self.gamma              = config.gamma
        self.crossq_joint_batch   = config.crossq_joint_batch
        self.normalize_actor_loss = config.normalize_actor_loss
        # Multi-horizon Q-chunking.  Empty → single chunk-level Q.  Horizons are
        # numbers of committed actions (1..H); the critic token index is k-1.
        self.td_horizons = tuple(config.td_horizons)
        if self.td_horizons:
            assert self.td_horizons[-1] == config.action_horizon, (
                f"last td_horizon must equal action_horizon ({config.action_horizon}), "
                f"got {self.td_horizons}"
            )
            # Stored as plain int tuples (NOT jnp/np arrays): nnx rejects raw array
            # leaves on a Module ("Arrays leaves are not supported"); wrap with
            # jnp.asarray(...) at the use sites instead.
            self._horizon_idx = tuple(k - 1 for k in self.td_horizons)
            # Heads the actor maximizes.  Default = full chunk (last horizon).
            self.actor_horizons = tuple(config.actor_horizons) or (config.action_horizon,)
            assert all(h in self.td_horizons for h in self.actor_horizons), (
                f"actor_horizons {self.actor_horizons} must all be in td_horizons {self.td_horizons}"
            )
            self._actor_horizon_idx = tuple(h - 1 for h in self.actor_horizons)   # critic token index (k-1)
            self._actor_td_pos = tuple(self.td_horizons.index(h) for h in self.actor_horizons)  # pos within td_horizons

        # alpha schedule + step counter (unused for LPS loss but kept for parity).
        self._warmup_end     = int(config.warmup_ratio     * config.num_train_steps)
        self._transition_end = int(config.transition_ratio * config.num_train_steps)
        self._alpha_gamma    = config.alpha_gamma
        self._alpha_min      = config.alpha_min
        self._alpha_eta      = config.alpha_eta
        self._mf_loss_weight = config.mf_loss_weight
        self.sphere_latent   = config.sphere_latent
        self.time_sampler    = config.time_sampler
        self.use_jvp         = config.use_jvp
        self.jvp_fp32        = config.jvp_fp32
        self._mf_reweight    = config.mf_reweight
        self._reweight_kappa = config.reweight_kappa
        self._large_span_ratio = config.large_span_ratio
        self._large_span_warmup_gate = config.large_span_warmup_gate
        self._flow_ratio     = config.flow_ratio
        self._lambda_fm      = config.lambda_fm
        self._lambda_mf      = config.lambda_mf
        self.delta_conditioning = config.delta_conditioning
        self.train_step      = nnx.Variable(jnp.array(0, dtype=jnp.int32))

        self.deterministic = True

    # ------------------------------------------------------------------
    # Expert-layout hooks (4 experts: paligemma, action, critic, latent)
    # ------------------------------------------------------------------

    def _experts_prefix(self, prefix_tokens):
        return [prefix_tokens, None, None, None]

    def _experts_action(self, action_tokens):
        return [None, action_tokens, None, None]

    def _adarms_action(self, adarms_cond):
        return [None, adarms_cond, None, None]

    # _action_out_index() == 1 (inherited)

    def _critic_logits(self, kv_cache, prefix_mask, actions, token_indices=None):
        """C51 logits.  The critic is causal, so token i sees a_{0:i+1}.

        token_indices=None → last token = whole chunk Q(s, a_{0:H})  → [b, n_bins].
        token_indices=array(k-1) → per-horizon Q_k(s, a_{0:k})        → [b, n_h, n_bins].
        """
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)   # critic ⊥ frozen backbone
        critic_tokens, critic_mask, critic_ar = self.embed_critic_suffix(actions)
        full_attn, positions = self._suffix_attn_and_positions(prefix_mask, critic_mask, critic_ar)
        outs, _ = self.PaliGemma.llm(
            [None, None, critic_tokens, None],
            mask=full_attn, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, None, None, None],
        )
        feats = outs[2][:, -1] if token_indices is None else outs[2][:, token_indices]
        return self.critic_out_proj(feats)               # [b, n_bins] or [b, n_h, n_bins]

    def _critic_value_detached(self, kv_cache, prefix_mask, actions, token_indices=None):
        """E[V](s, a) with critic PARAMETERS detached (DDPG actor term).

        Gradient flows to the action (and thus the latent actor) via dQ/da, but
        the critic parameters are treated as constants — they are updated only by
        the critic TD loss, never inflated by the actor's value-maximization.

        token_indices=None → full-chunk value [b];  array → per-head values
        [b, n] at those critic token positions (multi-horizon actor).
        """
        gdef, critic_params, rest = nnx.split(self, _CRITIC_FILTER, ...)
        critic_params = jax.tree.map(jax.lax.stop_gradient, critic_params)
        model_sg = nnx.merge(gdef, critic_params, rest)
        logits = model_sg._critic_logits(kv_cache, prefix_mask, actions, token_indices)
        return expected_value(
            jax.nn.softmax(logits, axis=-1), self.v_min, self.v_max, self.n_bins
        )                                          # [b]  or  [b, n]

    # ------------------------------------------------------------------
    # Latent actor
    # ------------------------------------------------------------------

    def _to_sphere(self, z):
        """Project z to the hypersphere of radius sqrt(ah*ad) (LPS latent prior)."""
        b = z.shape[0]
        flat = z.reshape(b, -1)
        norm = jnp.sqrt(jnp.sum(jnp.square(flat), axis=-1, keepdims=True) + 1e-6)
        scale = jnp.sqrt(jnp.float32(flat.shape[-1]))
        return (flat / norm * scale).reshape(z.shape)

    def _latent_action(self, kv_cache, prefix_mask, batch_size):
        """Deterministic latent z(s) on the sphere — shape [b, ah, ad].

        The learned query tokens attend to the (frozen) prefix KV; the latent
        expert produces a per-step embedding → action_dim → sphere.
        """
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)   # latent ⊥ frozen backbone
        q = jnp.broadcast_to(self.latent_query.value[None], (batch_size, *self.latent_query.value.shape))
        mask = jnp.ones(q.shape[:2], dtype=jnp.bool_)
        ar   = jnp.array([True] + [False] * (self.action_horizon - 1))   # bidirectional within chunk
        full_attn, positions = self._suffix_attn_and_positions(prefix_mask, mask, ar)
        outs, _ = self.PaliGemma.llm(
            [None, None, None, q],
            mask=full_attn, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, None, None, None],
        )
        z = self.latent_out_proj(outs[3])         # [b, ah, ad]
        return self._to_sphere(z)

    def _decode(self, kv_cache, prefix_mask, observation, z):
        """Frozen 1-NFE decode: a = z - u(z, t=1, r=0).

        Gradient flows through z into the latent actor; the action expert params
        are frozen (excluded from the trainable filter), so they are not updated.
        No clip (matches pi0 / alpha-flow inference).
        """
        b = z.shape[0]
        t = jnp.ones(b)
        r = jnp.zeros(b)
        u = self._action_velocity(kv_cache, prefix_mask, observation, z, t, r)
        return z - u

    def steer_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **_unused,
    ) -> _model.Actions:
        """Inference: steered 1-NFE action a = decode(s, z(s))."""
        observation = _model.preprocess_observation(None, observation, train=False)
        kv, pm = self._embed_prefix_kv(observation)
        z = self._latent_action(kv, pm, observation.state.shape[0])
        return self._decode(kv, pm, observation, z)

    @override
    def sample_actions(self, rng, observation, *, num_steps: int = 1, noise=None):  # noqa: ARG002
        """LPS inference uses the steered latent (ignores noise/num_steps)."""
        return self.steer_actions(rng, observation)

    # ------------------------------------------------------------------
    # Next-state prefix (built from the carried next-state fields)
    # ------------------------------------------------------------------

    def _next_observation(self, observation: _model.Observation) -> _model.Observation:
        """Assemble the next-state Observation from the carried next-* fields.

        Next-state images/state are already model-ready (the data pipeline
        resized/normalized them).  The prompt is the same task → reuse current's.
        """
        return _model.Observation(
            images=observation.next_images,
            image_masks=observation.next_image_masks,
            state=observation.next_state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )

    @staticmethod
    def _cat_obs(o1: _model.Observation, o2: _model.Observation) -> _model.Observation:
        """Concatenate two observations along the batch axis (CrossQ joint batch).

        Only the prefix fields (images / state / prompt) are concatenated — that
        is all `_embed_prefix_kv` consumes.  Processing [s; s'] in one forward is
        mathematically identical to two separate forwards (no BatchNorm; RMSNorm
        and attention are per-sample), just ~2x cheaper to compile and run.
        """
        cat  = lambda a, b: jnp.concatenate([a, b], axis=0)
        catd = lambda d1, d2: {k: cat(d1[k], d2[k]) for k in d1}
        return _model.Observation(
            images=catd(o1.images, o2.images),
            image_masks=catd(o1.image_masks, o2.image_masks),
            state=cat(o1.state, o2.state),
            tokenized_prompt=None if o1.tokenized_prompt is None
                else cat(o1.tokenized_prompt, o2.tokenized_prompt),
            tokenized_prompt_mask=None if o1.tokenized_prompt_mask is None
                else cat(o1.tokenized_prompt_mask, o2.tokenized_prompt_mask),
        )

    # ------------------------------------------------------------------
    # Multi-horizon Q-chunking loss (td_horizons set)
    # ------------------------------------------------------------------

    def _loss_multi_horizon(self, observation, actions, mc_t, reward_win, done, b, H):
        """Multi-horizon PREDICTION, single-state BACKUP.

        Prediction: the causal critic exposes a value at every chunk length
        k ∈ td_horizons — Q_k(s, a_{0:k}) = the (k-1)-th critic token → [b, n_h].

        Backup: ONE TD target, computed from the state after the FULL chunk
        (s_{t+H}) exactly as in the single-Q path —
            y = max(R_chunk + γ^H·V(s_{t+H}),  G_t),  done-masked  → [b]
        — and the SAME y supervises every horizon head (all heads estimate V(s_t);
        they differ only in how much committed action context a_{0:k} they see).
        The actor maximizes the full-chunk Q (last horizon = the deployed quantity).
        """
        n_h    = len(self.td_horizons)
        done   = done.astype(jnp.float32)                          # [b]
        gammaH = self.gamma ** H
        next_observation = self._next_observation(observation)

        if self.crossq_joint_batch:
            # Joint PREFIX forward over [s; s'] = [2b] (shares the frozen VLM).  The
            # critic forwards stay separate: data uses the per-horizon tokens, next
            # uses the single full-chunk token.
            joint_obs = self._cat_obs(observation, next_observation)
            kvj, pmj  = self._embed_prefix_kv(joint_obs)
            # kv_cache layout is [l, b, t, k, h] → the batch axis is 1, NOT 0.
            kv  = jax.tree.map(lambda x: x[:, :b], kvj)
            kvn = jax.tree.map(lambda x: x[:, b:], kvj)
            pm, pmn = pmj[:b], pmj[b:]
        else:
            kv,  pm  = self._embed_prefix_kv(observation)
            kvn, pmn = self._embed_prefix_kv(next_observation)

        # Backup: full-chunk value V(s_{t+H}) at the single next state.
        z_next = self._latent_action(kvn, pmn, b)
        a_next = self._decode(kvn, pmn, next_observation, z_next)
        logits_next = jax.lax.stop_gradient(self._critic_logits(kvn, pmn, a_next))   # [b, n_bins]
        v_next = expected_value(
            jax.nn.softmax(logits_next, axis=-1), self.v_min, self.v_max, self.n_bins
        )                                                          # [b]  full-chunk V(s')

        discounts = self.gamma ** jnp.arange(H, dtype=jnp.float32)  # [H]
        r_chunk = jnp.sum(reward_win * discounts, axis=-1)         # [b]
        y_td    = r_chunk + gammaH * v_next                       # [b]
        y_boot  = jnp.maximum(y_td, mc_t)                         # CalQL MC anchor
        y       = jnp.where(done > 0.0, mc_t, y_boot)             # done → MC return
        y       = jax.lax.stop_gradient(jnp.clip(y, self.v_min, self.v_max))  # [b]  single target

        # Prediction: per-horizon data Q  Q_k(s, a_data_{0:k}).
        logits_data = self._critic_logits(kv, pm, actions, jnp.asarray(self._horizon_idx))        # [b, n_h, n_bins]
        # Broadcast the SAME target y to every horizon head.
        critic_loss = critic_loss_hl_gauss(
            logits_data.reshape(b * n_h, self.n_bins), jnp.repeat(y, n_h),
            self.v_min, self.v_max, self.n_bins, self.hl_gauss_sigma,
        ).reshape(b, n_h).mean(axis=-1)                           # [b]  mean over horizons

        # ── Actor (DDPG): maximize the mean Q over the configured actor_horizons ──
        z_cur   = self._latent_action(kv, pm, b)
        a_actor = self._decode(kv, pm, observation, z_cur)
        q_actor = self._critic_value_detached(kv, pm, a_actor, jnp.asarray(self._actor_horizon_idx))  # [b, n_a]
        value_actor = jnp.mean(q_actor, axis=-1)                   # [b]  mean over actor heads
        actor_loss  = -value_actor
        if self.normalize_actor_loss:
            actor_scale = jnp.abs(jnp.mean(jax.lax.stop_gradient(actor_loss))) + 1e-8
            actor_loss  = actor_loss / actor_scale

        total = self.critic_loss_weight * critic_loss + self.actor_loss_weight * actor_loss

        # ── diagnostics ───────────────────────────────────────────────────────
        q_data = expected_value(
            jax.nn.softmax(jax.lax.stop_gradient(logits_data), axis=-1),
            self.v_min, self.v_max, self.n_bins,
        )                                                          # [b, n_h]  Q_k(s, a_data)
        # advantage vs the dataset Q at the SAME head(s) the actor optimizes.
        q_data_actor = jnp.mean(q_data[:, jnp.asarray(self._actor_td_pos)], axis=-1)   # [b]
        advantage = value_actor - q_data_actor                     # steered − data (actor heads)
        z_flat = z_cur.reshape(b, -1)

        def _mmm(prefix, x):
            return {f"{prefix}_mean": jnp.mean(x), f"{prefix}_min": jnp.min(x), f"{prefix}_max": jnp.max(x)}

        aux = {
            "loss/critic":  jnp.mean(critic_loss),
            "loss/actor":   jnp.mean(actor_loss),
            "loss/total":   jnp.mean(total),
            **_mmm("critic/q_data", q_data),                       # over all horizons
            "critic/q_data_full_mean": jnp.mean(q_data[:, -1]),    # full-chunk Q
            "critic/value_mae":   jnp.mean(jnp.abs(q_data[:, -1] - mc_t)),
            **_mmm("critic/mc_return", mc_t),
            **_mmm("td/target", y),
            **_mmm("td/q_next", v_next),
            "td/r_chunk_mean":    jnp.mean(r_chunk),
            "td/mc_anchor_frac":  jnp.mean((mc_t >= y_td).astype(jnp.float32)),
            "td/done_frac":       jnp.mean(done),
            **_mmm("actor/q_steered", value_actor),
            **_mmm("actor/advantage", advantage),
            "latent/z_abs_mean":  jnp.mean(jnp.abs(z_flat)),
            "latent/z_norm_mean": jnp.mean(jnp.linalg.norm(z_flat, axis=-1)),
            "latent/z_batch_std": jnp.mean(jnp.std(z_flat, axis=0)),
        }
        # per-horizon data-Q spread (how prediction varies with committed context)
        for i, k in enumerate(self.td_horizons):
            aux[f"critic/q_data_h{k}_mean"] = jnp.mean(q_data[:, i])
        return total, aux

    # ------------------------------------------------------------------
    # compute_loss: offline RL — CrossQ chunked TD (MC-anchored) + DDPG actor
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        """Returns (per-sample combined loss [b], aux dict).

        Critic: chunked TD target  y = max(R_chunk + γ^H·Q(s', a'),  mc_return)
                (CalQL MC anchor; no target network).  The next-state value
                Q(s', a') is computed in the SAME forward batch as Q(s, a_data)
                (CrossQ joint batch) and stop-gradient'd for the target.
        Actor : DDPG — maximize E[V](s, decode(s, z(s))) with critic params
                detached, so only the latent actor moves toward high value.
        """
        preprocess_rng, _ = jax.random.split(rng)
        mc_t       = observation.mc_return    # mc_return[t]  (CalQL anchor only)
        reward_win = observation.reward       # reward[t:t+H]  [b, H]
        done       = observation.done.astype(jnp.float32)  # [b]  next state past episode end
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        b = actions.shape[0]
        H = self.action_horizon
        gammaH = self.gamma ** H

        # Multi-horizon Q-chunking: per-horizon n-step TD (next-states at each
        # td_horizon).  Self-contained path; single chunk-level Q below otherwise.
        if self.td_horizons:
            return self._loss_multi_horizon(observation, actions, mc_t, reward_win, done, b, H)

        next_observation = self._next_observation(observation)

        if self.crossq_joint_batch:
            # CrossQ joint batch: prefix forward (and the data/next critic forward)
            # run ONCE over [s; s'] = [2b] instead of twice.  No BatchNorm → bit-
            # identical to the separate-forward path, just ~2x cheaper.
            joint_obs = self._cat_obs(observation, next_observation)
            kvj, pmj  = self._embed_prefix_kv(joint_obs)                 # prefix forward ×1 (2b)
            # kv_cache layout is [l, b, t, k, h] → the batch axis is 1, NOT 0.
            kv  = jax.tree.map(lambda x: x[:, :b], kvj)
            kvn = jax.tree.map(lambda x: x[:, b:], kvj)
            pm, pmn = pmj[:b], pmj[b:]

            z_next = self._latent_action(kvn, pmn, b)
            a_next = self._decode(kvn, pmn, next_observation, z_next)

            actions_joint = jnp.concatenate([actions, a_next], axis=0)   # [2b, ah, ad]
            logits_joint  = self._critic_logits(kvj, pmj, actions_joint) # critic forward ×1 (2b)
            logits_data = logits_joint[:b]
            logits_next = jax.lax.stop_gradient(logits_joint[b:])
        else:
            # Baseline: separate forwards for current and next.
            kv,  pm  = self._embed_prefix_kv(observation)
            kvn, pmn = self._embed_prefix_kv(next_observation)

            z_next = self._latent_action(kvn, pmn, b)
            a_next = self._decode(kvn, pmn, next_observation, z_next)

            logits_data = self._critic_logits(kv,  pm,  actions)        # [b, n_bins]
            logits_next = jax.lax.stop_gradient(self._critic_logits(kvn, pmn, a_next))

        v_next = expected_value(
            jax.nn.softmax(logits_next, axis=-1), self.v_min, self.v_max, self.n_bins
        )                                                               # [b]  Q(s', a')  chunk value

        # ── Chunked TD target, MC-anchored (CalQL), done-masked ───────────────
        # R_chunk = Σ_{i=0}^{H-1} γ^i · reward[t+i]  (discounted dataset reward over
        # the chunk; the failure penalty is baked into `reward`).  mc_return is used
        # ONLY as the CalQL anchor (not to build R_chunk).
        discounts = self.gamma ** jnp.arange(H, dtype=jnp.float32)      # [H]
        r_chunk = jnp.sum(reward_win * discounts, axis=-1)             # [b]
        y_td    = r_chunk + gammaH * v_next                           # [b]
        y_boot  = jnp.maximum(y_td, mc_t)                            # CalQL MC anchor
        # done (s_{t+H} past the episode end): no valid bootstrap → use the MC
        # return G_t directly (= truncated chunk return, terminal reward included).
        y       = jnp.where(done > 0.0, mc_t, y_boot)               # [b]
        y       = jax.lax.stop_gradient(jnp.clip(y, self.v_min, self.v_max))  # [b]
        critic_loss = critic_loss_hl_gauss(
            logits_data, y, self.v_min, self.v_max, self.n_bins, self.hl_gauss_sigma
        )                                                               # [b]

        # ── Actor (DDPG): maximize value, critic params detached ──────────────
        z_cur   = self._latent_action(kv, pm, b)
        a_actor = self._decode(kv, pm, observation, z_cur)             # frozen decoder
        value_actor = self._critic_value_detached(kv, pm, a_actor)     # [b], critic θ const
        actor_loss  = -value_actor                                     # [b]
        if self.normalize_actor_loss:
            # Divide the DDPG loss by its own (detached) magnitude → mean scale ±1,
            # so the actor gradient size is independent of the absolute Q-value
            # scale (decouples actor_loss_weight from how big Q is; per-sample
            # relative weighting preserved).
            actor_scale = jnp.abs(jnp.mean(jax.lax.stop_gradient(actor_loss))) + 1e-8
            actor_loss  = actor_loss / actor_scale                     # [b], mean |·| ≈ 1

        total = self.critic_loss_weight * critic_loss + self.actor_loss_weight * actor_loss

        # ── diagnostics (rich monitoring) ─────────────────────────────────────
        pred_value_data = expected_value(
            jax.nn.softmax(jax.lax.stop_gradient(logits_data), axis=-1),
            self.v_min, self.v_max, self.n_bins,
        )                                                               # [b]  Q(s, a_data)
        advantage = value_actor - pred_value_data                       # [b]  steered − data
        probs_data = jax.nn.softmax(jax.lax.stop_gradient(logits_data), axis=-1)
        critic_entropy = -jnp.sum(probs_data * jnp.log(probs_data + 1e-8), axis=-1)  # [b]
        z_flat = z_cur.reshape(b, -1)                                   # latent actor output

        def _mmm(prefix, x):
            return {f"{prefix}_mean": jnp.mean(x), f"{prefix}_min": jnp.min(x), f"{prefix}_max": jnp.max(x)}

        aux = {
            "loss/critic":  jnp.mean(critic_loss),
            "loss/actor":   jnp.mean(actor_loss),
            "loss/total":   jnp.mean(total),
            # critic Q(s, a_data)  — value the critic assigns to dataset actions
            **_mmm("critic/q_data", pred_value_data),
            "critic/q_data_std":  jnp.std(pred_value_data),       # ~0 ⇒ value collapse
            "critic/value_mae":   jnp.mean(jnp.abs(pred_value_data - mc_t)),
            "critic/entropy":     jnp.mean(critic_entropy),       # C51 dist sharpness
            # MC return target scale (data) + TD target
            **_mmm("critic/mc_return", mc_t),
            **_mmm("td/target", y),
            **_mmm("td/q_next", v_next),                          # V(s', a')
            "td/r_chunk_mean":    jnp.mean(r_chunk),
            "td/mc_anchor_frac":  jnp.mean((mc_t >= y_td).astype(jnp.float32)),
            "td/done_frac":       jnp.mean(done),
            # actor: steered value Q(s, decode(s, z(s))) and advantage over data
            **_mmm("actor/q_steered", value_actor),
            **_mmm("actor/advantage", advantage),                # >0 ⇒ actor steers up
            # latent actor health
            "latent/z_abs_mean":  jnp.mean(jnp.abs(z_flat)),
            "latent/z_norm_mean": jnp.mean(jnp.linalg.norm(z_flat, axis=-1)),
            "latent/z_batch_std": jnp.mean(jnp.std(z_flat, axis=0)),  # ~0 ⇒ latent collapse
        }
        return total, aux
