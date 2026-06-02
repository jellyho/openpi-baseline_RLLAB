"""
Pi0 with Critic Expert — extends Pi0AlphaFlow.

Architecture
────────────
    PaliGemma (image + language prefix)
    ├─ Action Expert   ← Pi0AlphaFlow (flow-matching + alpha-flow)
    └─ Critic Expert   ← new, distributional Q-value head (C51 / HL-Gauss)

The three experts share one joint Gemma transformer.
Action and Critic experts are isolated from each other via the attention mask;
both attend to the PaliGemma prefix but not to each other.

Passing None for an unused expert causes the Gemma module to skip its
QKV / FFN computation entirely (no wasted FLOPs).

Inheritance
───────────
    _model.BaseModel
    └─ Pi0                      (2-expert Gemma, standard FM)
       └─ Pi0AlphaFlow          (adds r_proj + alpha-flow losses)
          └─ Pi0WithCritic      (3-expert Gemma + C51 critic)

Pi0WithCritic overrides __init__ to create the 3-expert Gemma, and
overrides the three forward-pass methods that are hardcoded for 2 experts
(_forward_velocity, compute_loss, sample_actions).  Everything else
(alpha-flow losses, 1-NFE sampling, etc.) is inherited unchanged.

C51 / HL-Gauss
──────────────
The critic outputs a categorical distribution over n_bins (default 101)
evenly spaced between v_min and v_max.  Training targets come from
Monte-Carlo returns collected during BC / offline RL.
"""

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.models.pi0_alphaflow import Pi0AlphaFlow, Pi0AlphaFlowConfig
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.shared.nnx_utils as nnx_utils
from openpi.shared import array_typing as at


# ---------------------------------------------------------------------------
# HL-Gauss / C51 utilities
# ---------------------------------------------------------------------------

def make_bin_centers(v_min: float, v_max: float, n_bins: int) -> jax.Array:
    return jnp.linspace(v_min, v_max, n_bins)


def scalar_to_hl_gauss(
    value,
    v_min: float,
    v_max: float,
    n_bins: int = 101,
    sigma: float = 0.5,
) -> jax.Array:
    """
    Convert scalar return(s) to an HL-Gauss target distribution.

    Places a Gaussian kernel (std = sigma * bin_width) centred at `value`
    over evenly-spaced bin centres, then normalises to a probability vector.

    value: [...] scalar returns
    Returns: [..., n_bins] probability distribution
    """
    bin_width = (v_max - v_min) / (n_bins - 1)
    centers   = make_bin_centers(v_min, v_max, n_bins)   # (n_bins,)
    diff      = centers - value[..., None]               # [..., n_bins]
    probs     = jnp.exp(-0.5 * (diff / (sigma * bin_width)) ** 2)
    return probs / probs.sum(axis=-1, keepdims=True)


def expected_value(probs, v_min: float, v_max: float, n_bins: int = 101) -> jax.Array:
    """E[V] = Σ p_i * z_i from a C51 distribution."""
    centers = make_bin_centers(v_min, v_max, n_bins)
    return jnp.sum(probs * centers, axis=-1)


def critic_loss_hl_gauss(
    logits,
    returns,
    v_min: float,
    v_max: float,
    n_bins: int = 101,
    sigma:  float = 0.5,
) -> jax.Array:
    """
    C51 cross-entropy loss with HL-Gauss target.

    logits:  [..., n_bins]  raw logits (before softmax)
    returns: [...]          scalar MC return targets
    Returns: [...] per-sample CE loss
    """
    target    = scalar_to_hl_gauss(returns, v_min, v_max, n_bins, sigma)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target * log_probs, axis=-1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0WithCriticConfig(Pi0AlphaFlowConfig):
    """
    Config for Pi0WithCritic.

    Extends Pi0AlphaFlowConfig with critic-expert and C51 hyperparameters.
    pi05=True is required (inherited from Pi0AlphaFlowConfig).

    Critic expert uses gemma_100m (~100 M params) by default, which is
    significantly smaller than the action expert (gemma_300m, 311 M params).
    """

    # Critic expert backbone — must share head_dim / num_heads / num_kv_heads
    # with the action expert.  gemma_100m satisfies this constraint.
    critic_expert_variant: _gemma.Variant = "gemma_100m"

    # C51 hyperparameters
    # Dataset MC returns are normalized to [-1, 0], so v_min=-1, v_max=0.
    n_bins:          int   = 101
    v_min:           float = -1.0
    v_max:           float = 0.0
    hl_gauss_sigma:  float = 0.5   # Gaussian width in units of bin_width

    # Weight of the critic loss relative to the alpha-flow loss in compute_loss.
    critic_loss_weight: float = 1.0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0WithCritic":
        return Pi0WithCritic(self, rngs=nnx.Rngs(rng))

    def get_rectify_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the VLM (SigLIP + PaliGemma prefix expert); train everything else
        — the action expert (`_1`) + r-conditioning + action/state projections, and
        the critic expert (`_2`) + critic projections.

        For 2-stage RFT phase-1 ("rectify"): init from a task-adapted flow-matching
        checkpoint, distill it to a 1-NFE mean-velocity policy (action expert) while
        warming up the critic, without disturbing the frozen perception backbone.
        Mirrors Pi0LPSRFTConfig.get_freeze_filter but keeps the action expert trainable.
        """
        # NB: qualify the expert selectors with ".*llm.*" — a bare ".*_1.*" also
        # matches SigLIP submodules (LayerNorm_1, Dense_1) and would leak the
        # vision tower into the trainable set.  ".*llm.*_1.*" is the same selector
        # pi0_config.py uses for the action expert.
        trainable = nnx.Any(
            nnx_utils.PathRegex(".*llm.*_1.*"),     # action expert (gemma suffix _1)
            nnx_utils.PathRegex(".*llm.*_2.*"),     # critic expert (gemma suffix _2)
            nnx_utils.PathRegex(".*critic_.*"),     # critic_in_proj / critic_out_proj
            nnx_utils.PathRegex(".*r_mlp.*"),       # r-conditioning (alpha-flow)
            nnx_utils.PathRegex(".*action_.*"),     # action_in_proj / action_out_proj
            nnx_utils.PathRegex(".*time_mlp.*"),    # time embedding MLP
            nnx_utils.PathRegex(".*state_proj.*"),  # robot-state projection
        )
        return nnx.Not(trainable)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Pi0WithCritic(Pi0AlphaFlow):
    """
    Pi0AlphaFlow extended with a Critic Expert for distributional value
    estimation (C51 / HL-Gauss).

    Overrides
    ─────────
    __init__                   — creates a 3-expert Gemma instead of 2-expert
    _forward_velocity_with_prefix — routes [prefix, action, None] through the 3-expert llm
    compute_loss               — alpha-flow (parent logic) + C51 critic, shared prefix
    sample_actions             — delegates to sample_actions_1nfe (1-NFE, the alpha-flow goal)

    Inherits unchanged from Pi0AlphaFlow
    ─────────────────────────────────────
    embed_prefix, embed_suffix, embed_suffix_with_r  (Pi0 / Pi0AlphaFlow)
    r_proj, _sample_sphere_noise, train_step counter, _alpha_now
    _alphaflow_loss_with_prefix (discrete/JVP branches), sample_actions_1nfe
    """

    def __init__(self, config: Pi0WithCriticConfig, rngs: nnx.Rngs):
        # Bypass Pi0AlphaFlow.__init__ (which creates a 2-expert Gemma) and
        # also bypass Pi0.__init__ — we rebuild everything with 3 experts.
        _model.BaseModel.__init__(
            self, config.action_dim, config.action_horizon, config.max_token_len
        )
        self.pi05 = True

        paligemma_cfg     = _gemma.get_config(config.paligemma_variant)
        action_expert_cfg = _gemma.get_config(config.action_expert_variant)
        critic_expert_cfg = _gemma.get_config(config.critic_expert_variant)

        # ── 3-expert Gemma ──────────────────────────────────────────────────
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_cfg, action_expert_cfg, critic_expert_cfg],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True, False],  # only action expert uses adaRMS
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
        img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ── Action expert (Pi05 / Pi0AlphaFlow) ─────────────────────────────
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_in    = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
        self.time_mlp_out   = nnx.Linear(action_expert_cfg.width, action_expert_cfg.width, rngs=rngs)
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

        # C51 params stored as plain attributes (not nnx params)
        self.n_bins             = config.n_bins
        self.v_min              = config.v_min
        self.v_max              = config.v_max
        self.hl_gauss_sigma     = config.hl_gauss_sigma
        self.critic_loss_weight = config.critic_loss_weight

        # Alpha schedule + step counter (we bypassed Pi0AlphaFlow.__init__, so
        # replicate its schedule setup here).
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
    # Expert-layout hooks (3 experts: paligemma, action, critic)
    # ------------------------------------------------------------------

    def _experts_prefix(self, prefix_tokens):
        return [prefix_tokens, None, None]

    def _experts_action(self, action_tokens):
        return [None, action_tokens, None]

    def _adarms_action(self, adarms_cond):
        return [None, adarms_cond, None]

    # _action_out_index() stays 1 (action expert) — inherited from Pi0AlphaFlow.

    # ------------------------------------------------------------------
    # Critic forward (suffix-only, reuses the cached prefix KV)
    # ------------------------------------------------------------------

    def _critic_logits(self, kv_cache, prefix_mask, actions):
        """Single chunk-level C51 logits [b, n_bins], reusing the shared prefix KV.

        The critic tokens are causally masked (token i sees a_{0..i}), so the LAST
        token has attended to the entire action chunk — its output is Q(s, a_{0:H}),
        the value of the whole chunk.  We read that token and project it to one C51
        distribution.  (Per-timestep / multi-horizon value heads — 5/10/25/50 — are
        a planned extension; for now there is a single Q per (state, chunk).)

        The prefix KV is **stop-gradient'd**: the randomly-initialised critic
        head must not backprop into the shared PaliGemma backbone, otherwise its
        large early gradients corrupt the pretrained features the action expert
        depends on (the action / alpha-flow loss then stalls).  The critic still
        trains its own expert + projection on top of the (frozen) prefix features
        — matching the paper's design of a value head that does not perturb the
        policy backbone.
        """
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        critic_tokens, critic_mask, critic_ar_mask = self.embed_critic_suffix(actions)
        full_attn, positions = self._suffix_attn_and_positions(prefix_mask, critic_mask, critic_ar_mask)
        outs, _ = self.PaliGemma.llm(
            [None, None, critic_tokens],
            mask=full_attn,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, None, None],
        )
        critic_out = outs[2][:, -1]                      # [b, critic_w]  last token = whole chunk
        return self.critic_out_proj(critic_out)          # [b, n_bins]

    # ------------------------------------------------------------------
    # compute_loss: alpha-flow (reused from parent) + critic, shared prefix KV
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
        """Joint objective: alpha-flow action loss + C51 critic loss.

        Returns (per-token combined loss [*b, ah], aux metrics dict).

        The prefix backbone runs ONCE; its KV cache is reused by the action
        expert (alpha-flow u_pred / u_next) and the critic expert.  All
        alpha-flow logic — schedule, discrete/JVP branching — is inherited from
        Pi0AlphaFlow (which routes through the 3-expert hooks above).
        """
        preprocess_rng, flow_rng = jax.random.split(rng)
        mc_returns  = observation.mc_return
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Prefix backbone ONCE → shared KV cache.
        prefix_kv = self._embed_prefix_kv(observation)       # (kv_cache, prefix_mask)
        kv_cache, prefix_mask = prefix_kv

        # Action expert: alpha-flow loss (parent logic, 3-expert hooks).
        alpha   = self._alpha_now()
        af_loss, af_raw_l2 = self._alphaflow_loss_with_prefix(flow_rng, observation, actions, prefix_kv, alpha)

        # Critic expert: single chunk-level C51 loss (reuses the same prefix KV).
        logits      = self._critic_logits(kv_cache, prefix_mask, actions)  # [b, n_bins]
        critic_loss = critic_loss_hl_gauss(
            logits, mc_returns, self.v_min, self.v_max, self.n_bins, self.hl_gauss_sigma
        )                                                                 # [b]

        # Advance the shared alpha schedule once per train step.
        self.train_step.value = self.train_step.value + 1

        # Critic diagnostics: predicted E[V] vs MC return.
        probs       = jax.nn.softmax(jax.lax.stop_gradient(logits), axis=-1)
        pred_value  = expected_value(probs, self.v_min, self.v_max, self.n_bins)  # [b]
        value_mae   = jnp.mean(jnp.abs(pred_value - mc_returns))

        phase = jnp.where(alpha >= 1.0, 1.0, jnp.where(alpha <= 0.0, 0.0, 0.5))
        aux = {
            "alpha":              alpha,
            "phase":              phase,
            "loss/alphaflow":     jnp.mean(af_loss),
            "loss/l2_raw":        jnp.mean(af_raw_l2),   # plain MSE — same scale as pi05 FM
            "loss/critic":        jnp.mean(critic_loss),
            "critic/value_mean":  jnp.mean(pred_value),
            "critic/value_mae":   value_mae,
            "critic/mc_return_mean": jnp.mean(mc_returns),
        }
        # af_loss is per action token [b, ah]; broadcast the scalar chunk critic
        # loss [b] across tokens so jnp.mean weights it as one term per sample.
        return af_loss + self.critic_loss_weight * critic_loss[:, None], aux

    # ------------------------------------------------------------------
    # Override: alpha-flow sampling.
    # ------------------------------------------------------------------

    # sample_actions is inherited from Pi0AlphaFlow (honours num_steps:
    # 1 → 1-NFE, N → N-step), routed through the 3-expert hooks.

    # ------------------------------------------------------------------
    # Critic suffix embedding
    # ------------------------------------------------------------------

    def embed_critic_suffix(
        self,
        actions: _model.Actions,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """
        Embed clean dataset actions as critic expert tokens.

        The critic receives GROUND-TRUTH actions (not noisy ones) to evaluate
        Q(s, a_data) on the offline dataset.
        """
        critic_tokens = self.critic_in_proj(actions)                    # [b, ah, critic_w]
        input_mask    = jnp.ones(critic_tokens.shape[:2], dtype=jnp.bool_)
        # Causal masking: every critic token has ar=True so cumsum increments at each position.
        # Token i can attend to prefix (cumsum=0) and critic tokens 0..i (cumsum<=i+1).
        ar_mask       = jnp.ones(self.action_horizon, dtype=jnp.bool_)
        return critic_tokens, input_mask, ar_mask

    # ------------------------------------------------------------------
    # Critic forward pass
    # ------------------------------------------------------------------

    def compute_critic_logits(
        self,
        observation: _model.Observation,
        actions:     _model.Actions,
    ) -> at.Float[at.Array, "b n_bins"]:
        """
        Compute the single chunk-level C51 logits for Q(obs, a_{0:H}).

        The critic tokens are causally masked, so the last token has seen the whole
        chunk; its C51 distribution over [v_min, v_max] is the chunk value.

        Returns: [b, n_bins] logits (before softmax).
        """
        kv_cache, prefix_mask = self._embed_prefix_kv(observation)
        return self._critic_logits(kv_cache, prefix_mask, actions)   # [b, n_bins]

    # ------------------------------------------------------------------
    # Critic loss (supervised by MC returns)
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        rng:         at.KeyArrayLike,
        observation: _model.Observation,
        actions:     _model.Actions,
        mc_returns:  at.Float[at.Array, " b"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """
        HL-Gauss C51 cross-entropy loss for the single chunk-level value.

        mc_returns: [b] scalar G_t for each sample.
        Returns:    [b] per-sample CE loss.
        """
        preprocess_rng, _ = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        logits = self.compute_critic_logits(observation, actions)  # [b, n_bins]

        return critic_loss_hl_gauss(
            logits, mc_returns,
            self.v_min, self.v_max, self.n_bins, self.hl_gauss_sigma,
        )                                                           # [b]

    # ------------------------------------------------------------------
    # Value inference
    # ------------------------------------------------------------------

    def predict_value(
        self,
        observation: _model.Observation,
        actions:     _model.Actions,
    ) -> at.Float[at.Array, " b"]:
        """
        Return the chunk-level E[V] = Q(s, a_{0:H}) from the C51 distribution.

        Returns [b] — one expected value per sample (the whole action chunk).
        """
        logits = self.compute_critic_logits(observation, actions)  # [b, n_bins]
        probs  = jax.nn.softmax(logits, axis=-1)
        return expected_value(probs, self.v_min, self.v_max, self.n_bins)  # [b]
