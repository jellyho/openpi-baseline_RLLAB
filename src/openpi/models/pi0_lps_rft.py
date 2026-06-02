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

    # Latent actor backbone — must share head_dim / num_heads / num_kv_heads
    # with the other experts.  gemma_100m satisfies this.
    latent_expert_variant: _gemma.Variant = "gemma_100m"

    # Loss weights.
    actor_loss_weight:  float = 1.0
    # critic_loss_weight inherited (default 1.0).

    # Discount factor (stored for the v2 chunked-TD term).
    gamma: float = 0.995

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

    def _critic_logits(self, kv_cache, prefix_mask, actions):
        """Per-token C51 logits [b, ah, n_bins] (4-expert critic token list)."""
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)   # critic ⊥ frozen backbone
        critic_tokens, critic_mask, critic_ar = self.embed_critic_suffix(actions)
        full_attn, positions = self._suffix_attn_and_positions(prefix_mask, critic_mask, critic_ar)
        outs, _ = self.PaliGemma.llm(
            [None, None, critic_tokens, None],
            mask=full_attn, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, None, None, None],
        )
        return self.critic_out_proj(outs[2])      # [b, ah, n_bins]

    def _critic_value_detached(self, kv_cache, prefix_mask, actions):
        """E[V](s, a) with critic PARAMETERS detached (DDPG actor term).

        Gradient flows to the action (and thus the latent actor) via dQ/da, but
        the critic parameters are treated as constants — they are updated only by
        the critic TD loss, never inflated by the actor's value-maximization.
        """
        gdef, critic_params, rest = nnx.split(self, _CRITIC_FILTER, ...)
        critic_params = jax.tree.map(jax.lax.stop_gradient, critic_params)
        model_sg = nnx.merge(gdef, critic_params, rest)
        logits = model_sg._critic_logits(kv_cache, prefix_mask, actions)
        return expected_value(
            jax.nn.softmax(logits, axis=-1), self.v_min, self.v_max, self.n_bins
        )                                          # [b, ah]

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
        """Returns (per-token combined loss [*b, ah], aux dict).

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
        next_observation = self._next_observation(observation)
        b = actions.shape[0]
        H = self.action_horizon
        gammaH = self.gamma ** H

        # Prefix backbones (frozen): current and next, each ONCE.
        kv,  pm  = self._embed_prefix_kv(observation)
        kvn, pmn = self._embed_prefix_kv(next_observation)

        # ── Next action a' = decode(s', z(s')) (for the bootstrap) ────────────
        z_next  = self._latent_action(kvn, pmn, b)
        a_next  = self._decode(kvn, pmn, next_observation, z_next)

        # ── Critic forwards: Q(s,a_data) [grad] and Q(s',a') [stop-grad] ──────
        # (CrossQ: both pairs scored by the same critic; the next pair is the
        #  bootstrap target and is detached.)
        logits_data = self._critic_logits(kv,  pm,  actions)            # [b, ah, n_bins]
        logits_next = jax.lax.stop_gradient(self._critic_logits(kvn, pmn, a_next))
        v_next = expected_value(
            jax.nn.softmax(logits_next, axis=-1), self.v_min, self.v_max, self.n_bins
        )                                                               # [b, ah]  (per-token)
        v_next = jnp.mean(v_next, axis=-1)                              # [b]  Q(s', a')

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
        y       = jax.lax.stop_gradient(jnp.clip(y, self.v_min, self.v_max))
        y_exp   = jnp.broadcast_to(y[:, None], (b, H))                 # [b, ah]
        critic_loss = critic_loss_hl_gauss(
            logits_data, y_exp, self.v_min, self.v_max, self.n_bins, self.hl_gauss_sigma
        )                                                               # [b, ah]

        # ── Actor (DDPG): maximize value, critic params detached ──────────────
        z_cur   = self._latent_action(kv, pm, b)
        a_actor = self._decode(kv, pm, observation, z_cur)             # frozen decoder
        value_actor = self._critic_value_detached(kv, pm, a_actor)     # [b, ah], critic θ const
        actor_loss  = -value_actor

        total = self.critic_loss_weight * critic_loss + self.actor_loss_weight * actor_loss

        # diagnostics
        pred_value_data = expected_value(
            jax.nn.softmax(jax.lax.stop_gradient(logits_data), axis=-1),
            self.v_min, self.v_max, self.n_bins,
        )
        mc_exp = jnp.broadcast_to(mc_t[:, None], (b, H))
        aux = {
            "loss/critic":         jnp.mean(critic_loss),
            "loss/actor":          jnp.mean(actor_loss),
            "td/target":           jnp.mean(y),
            "td/r_chunk":          jnp.mean(r_chunk),
            "td/q_next":           jnp.mean(v_next),
            "td/mc_anchor_frac":   jnp.mean((mc_t >= y_td).astype(jnp.float32)),
            "td/done_frac":        jnp.mean(done),
            "critic/value_data":   jnp.mean(pred_value_data),
            "critic/value_mae":    jnp.mean(jnp.abs(pred_value_data - mc_exp)),
            "actor/value_steered": jnp.mean(value_actor),
            "actor/advantage":     jnp.mean(value_actor) - jnp.mean(pred_value_data),
        }
        return total, aux
