"""
Alpha-Flow fine-tuning for Pi05.

Creates a 1-NFE policy by fine-tuning a pretrained Pi05 model with the
discrete alpha-Flow curriculum objective.

Usage:
    # Fine-tune
    uv run scripts/train_alphaflow.py <config_name> --exp-name <run_name>

    # 1-step inference
    actions = model.sample_actions_1nfe(rng, observation)

Ref: "AlphaFlow: Understanding and Improving MeanFlow Models" (arXiv 2510.20771)
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
from openpi.models import model as _model
from openpi.shared import array_typing as at


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

def alpha_schedule(step, warmup_end: int, transition_end: int, gamma: float = 25.0, eta: float = 5e-3):
    """
    Sigmoid curriculum schedule.  All operations are JAX-traceable (JIT-safe).

    Returns a JAX scalar alpha:
        step <= warmup_end      → 1.0  (TFM warmup)
        warmup_end … transition_end  → sigmoid 1 → 0
        step >= transition_end  → 0.0  (exact MeanFlow JVP phase)

    Values near boundaries are snapped:
        raw > 1 - eta  → 1.0
        raw < eta      → 0.0  (triggers JVP branch)
    """
    step_f = step.astype(jnp.float32)
    midpoint = jnp.float32((warmup_end + transition_end) / 2.0)
    width    = jnp.float32(float(transition_end - warmup_end))

    x = gamma * (step_f - midpoint) / width
    raw = 1.0 - jax.nn.sigmoid(x)
    raw = jnp.where(raw > 1.0 - eta, 1.0, raw)
    raw = jnp.where(raw < eta,       0.0, raw)

    alpha = jnp.where(step_f <= jnp.float32(warmup_end),    1.0, raw)
    alpha = jnp.where(step_f >= jnp.float32(transition_end), 0.0, alpha)
    return alpha


def alpha_schedule_python(step: int, warmup_end: int, transition_end: int,
                          gamma: float = 25.0, eta: float = 5e-3) -> float:
    """
    Python-level version of alpha_schedule (returns a plain float).
    Used in the training loop for Python-level branching between
    discrete alpha-flow and JVP MeanFlow train steps.
    """
    import math
    if step <= warmup_end:
        return 1.0
    if step >= transition_end:
        return 0.0
    midpoint = (warmup_end + transition_end) / 2.0
    width = float(transition_end - warmup_end)
    x = gamma * (step - midpoint) / width
    raw = 1.0 - 1.0 / (1.0 + math.exp(-x))
    if raw > 1.0 - eta:
        return 1.0
    if raw < eta:
        return 0.0
    return float(raw)


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def _adaptive_l2_loss(error, weight_scale=1.0, c: float = 1e-3):
    """
    Adaptive L2 loss matching the official alpha-flow implementation (p=1).

    Official formula (loss.py line 592):
        weight = weight_scale / (loss_unscaled.detach() + eps)
        loss   = weight * loss_unscaled

    error: (..., action_dim)
    Returns (...) — mean squared error over action_dim, adaptively weighted.

    weight_scale:
        discrete MF samples → alpha   (decreases as curriculum progresses)
        JVP / FM samples    → 1.0
    """
    sq_err = jnp.mean(jnp.square(error), axis=-1)          # (...) = (B, ah)
    w = weight_scale / (sq_err + c)                         # p=1 (official)
    w = jax.lax.stop_gradient(w)
    return w * sq_err


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Pi0AlphaFlow(Pi0):
    """
    Pi05 extended with alpha-Flow fine-tuning support.

    Changes vs. the base Pi0 class:
    - Adds `r_proj`: a **zero-initialized** linear that adds target-timestep r
      conditioning to the adaRMS time embedding.  At initialization its output
      is 0, so the pretrained behaviour is preserved.
    - `embed_suffix_with_r`: thin wrapper that calls the parent `embed_suffix`
      and then adds the r residual.
    - `compute_alphaflow_loss`: discrete alpha-Flow training objective.
    - `sample_actions_1nfe`: single-step inference (t=1 → r=0).
    """

    def __init__(self, config: "Pi0AlphaFlowConfig", rngs: nnx.Rngs):
        assert config.pi05, "Pi0AlphaFlow requires pi05=True"
        super().__init__(config, rngs)

        # Zero-init so pretrained adaRMS conditioning is unchanged at the start
        # of fine-tuning.  We only need r (not t-r separately): the model
        # already receives t, so it can derive the interval length t-r itself.
        action_expert_cfg = _gemma_width(config)
        self.r_proj = nnx.Linear(
            action_expert_cfg,
            action_expert_cfg,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )

    # ------------------------------------------------------------------
    # r conditioning
    # ------------------------------------------------------------------

    def embed_suffix_with_r(self, obs, noisy_actions, timestep, r):
        """Embed suffix with additional target-timestep r conditioning."""
        tokens, mask, ar_mask, adarms_cond = self.embed_suffix(obs, noisy_actions, timestep)
        # adarms_cond is the time embedding produced by the base class (pi05 path)
        r_emb   = posemb_sincos(r, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        r_extra = self.r_proj(r_emb)
        return tokens, mask, ar_mask, adarms_cond + r_extra

    # ------------------------------------------------------------------
    # Forward pass helper
    # ------------------------------------------------------------------

    def _forward_velocity(self, observation, noisy_actions, t, r):
        """
        Full prefix+suffix forward pass, returns velocity prediction.

        Args:
            observation:    model Observation
            noisy_actions:  [b, ah, ad]  noisy action tensor z_t
            t:              [b]          current noise timestep
            r:              [b]          target timestep (r <= t)
        Returns:
            [b, ah, ad]  predicted mean velocity u_theta(z_t, r, t)
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_r(
            observation, noisy_actions, t, r
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask  = make_attn_mask(input_mask, ar_mask)
        positions  = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    # ------------------------------------------------------------------
    # Sphere noise sampling  (same prior as LPS)
    # ------------------------------------------------------------------

    def _sample_sphere_noise(self, rng, shape):
        """
        Sample from the hypersphere prior used in LPS.

        Each sample z has shape (B, action_horizon, action_dim).
        We normalize over all non-batch dims so every sample lies on a
        sphere of radius sqrt(action_horizon * action_dim), matching the
        expected L2 norm of a standard Gaussian in that space.
        """
        e = jax.random.normal(rng, shape)
        flat = e.reshape(shape[0], -1)                             # (B, H*D)
        norm = jnp.sqrt(jnp.sum(jnp.square(flat), axis=-1, keepdims=True) + 1e-6)
        scale = jnp.sqrt(jnp.float32(flat.shape[-1]))             # sqrt(H * D)
        return (flat / norm * scale).reshape(shape)

    # ------------------------------------------------------------------
    # Alpha-Flow loss
    # ------------------------------------------------------------------

    def compute_alphaflow_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        alpha,
        *,
        train: bool = False,
        flow_ratio: float = 0.25,
        utgt_clamp: float = 10.0,
    ):
        """
        Discrete alpha-Flow training objective (matches official loss.py).

        Batch split (official "ratio_fm" pattern):
          FM border (flow_ratio):   r = t  → target = v_t  (pure TFM supervision)
          MF samples (1-flow_ratio): r < t  → target = alpha*v_t + (1-alpha)*u_next

        Adaptive weighting (official p=1):
          FM:  weight = 1.0  / (sq_err + eps)
          MF:  weight = alpha / (sq_err + eps)

        Args:
            alpha:       JAX float32 scalar from alpha_schedule().
            flow_ratio:  fraction of batch reserved as FM border (r=t) samples.
            utgt_clamp:  clip target for stability (official: cfg.clamp_utgt).
        """
        preprocess_rng, noise_rng, t_rng, r_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b = actions.shape[0]
        batch_shape = actions.shape[:-2]                              # (b,)
        noise = self._sample_sphere_noise(noise_rng, actions.shape)  # (b, ah, ad)

        # --- Sample (t, r) ---
        n_fm = int(b * flow_ratio)   # FM border samples (r = t)

        t1 = jax.nn.sigmoid(jax.random.normal(t_rng, batch_shape) * 1.0 - 0.4)
        t2 = jax.nn.sigmoid(jax.random.normal(r_rng, batch_shape) * 1.0 - 0.4)
        t_all = jnp.maximum(t1, t2)
        r_mf  = jnp.minimum(t1, t2)

        # First n_fm samples: r = t (FM border);  rest: r < t (MF, min-max)
        t = t_all                                                    # (b,)
        r = jnp.concatenate([t_all[:n_fm], r_mf[n_fm:]], axis=0)   # (b,)

        t_e = t[..., None, None]
        s   = alpha * r + (1.0 - alpha) * t
        s_e = s[..., None, None]

        # Straight-line flow
        z_t = t_e * noise + (1.0 - t_e) * actions
        v_t = noise - actions

        # Intermediate state: z_s = z_t - (t-s)*v_t
        z_s = z_t - (t_e - s_e) * v_t

        # Forward passes
        u_pred = self._forward_velocity(observation, z_t, t, r)
        u_next = jax.lax.stop_gradient(
            self._forward_velocity(observation, z_s, s, r)
        )

        # Target with clipping (official: torch.clip)
        u_tgt = jax.lax.stop_gradient(
            jnp.clip(alpha * v_t + (1.0 - alpha) * u_next, -utgt_clamp, utgt_clamp)
        )

        err = u_pred - u_tgt
        # FM: weight_scale=1, MF: weight_scale=alpha
        loss_fm = _adaptive_l2_loss(err[:n_fm],  weight_scale=1.0)
        loss_mf = _adaptive_l2_loss(err[n_fm:],  weight_scale=alpha)
        return jnp.concatenate([loss_fm, loss_mf], axis=0)          # (b, ah)

    # ------------------------------------------------------------------
    # Exact MeanFlow loss via JVP  (alpha = 0 phase)
    # ------------------------------------------------------------------

    def compute_jvp_meanflow_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        flow_ratio: float = 0.25,
    ):
        """
        Exact MeanFlow training objective via JVP (alpha = 0 phase).

        Mirrors the official implementation: when alpha=0, dt = alpha*(t-r) = 0
        for ALL samples, so ALL go through the JVP branch.  The 25% FM border
        ratio is still maintained:

          r = t  (flow_ratio):   u_tgt = v_t - (t-r)*dudt = v_t  (TFM, t-r=0)
          r < t  (1-flow_ratio): u_tgt = v_t - (t-r)*dudt         (exact MeanFlow)

        Both groups get weight_scale = 1.0 (no alpha weighting in JVP phase).
        """
        preprocess_rng, noise_rng, t_rng, r_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b = actions.shape[0]
        batch_shape = actions.shape[:-2]
        noise = self._sample_sphere_noise(noise_rng, actions.shape)

        # Same (t, r) split as discrete loss
        n_fm = int(b * flow_ratio)
        t1 = jax.nn.sigmoid(jax.random.normal(t_rng, batch_shape) * 1.0 - 0.4)
        t2 = jax.nn.sigmoid(jax.random.normal(r_rng, batch_shape) * 1.0 - 0.4)
        t_all = jnp.maximum(t1, t2)
        r_mf  = jnp.minimum(t1, t2)

        t = t_all
        r = jnp.concatenate([t_all[:n_fm], r_mf[n_fm:]], axis=0)  # FM: r=t; MF: r<t

        t_e = t[..., None, None]
        r_e = r[..., None, None]

        z_t = t_e * noise + (1.0 - t_e) * actions
        v_t = noise - actions

        # JVP along ODE trajectory: dz/dt = v_t, dr/dt = 0, dt/dt = 1
        def fn(z_val, t_val, r_val):
            return self._forward_velocity(observation, z_val, t_val, r_val)

        u_pred, dudt = jax.jvp(
            fn,
            (z_t, t, r),
            (v_t, jnp.ones_like(t), jnp.zeros_like(r)),
        )

        # u_tgt = v_t - (t-r)*dudt
        #   r = t  → (t-r) = 0  → u_tgt = v_t  (FM supervision)
        #   r < t  → exact MeanFlow target
        u_tgt = jax.lax.stop_gradient(v_t - (t_e - r_e) * dudt)

        return _adaptive_l2_loss(u_pred - u_tgt, weight_scale=1.0)

    # ------------------------------------------------------------------
    # 1-NFE sampling
    # ------------------------------------------------------------------

    def sample_actions_1nfe(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        One-step (1-NFE) action sampling using the trained alpha-Flow policy.

        The model predicts the mean velocity u over the full interval [r=0, t=1].
        The clean action is recovered as:
            z_0 = z_1 - (t - r) * u = noise - 1.0 * u_pred
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self._sample_sphere_noise(rng, (batch_size, self.action_horizon, self.action_dim))

        t = jnp.ones(batch_size)   # t = 1  (pure noise)
        r = jnp.zeros(batch_size)  # r = 0  (target: clean action)

        u = self._forward_velocity(observation, noise, t, r)
        # z_0 = z_1 - (t - r) * u  =  noise - 1.0 * u
        return jnp.clip(noise - u, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0AlphaFlowConfig(pi0_config.Pi0Config):
    """
    Config for Pi0AlphaFlow.

    Inherits all Pi0Config fields.  Set pi05=True (required).

    Schedule is defined as fractions of num_train_steps so it adapts
    automatically when you change the training budget:

        [0,           warmup_ratio)       → alpha = 1.0  (TFM warmup)
        [warmup_ratio, transition_ratio)  → alpha sigmoid 1 → 0
        [transition_ratio, 1.0]           → alpha = 0.0  (JVP MeanFlow)

    Default: 20 % warmup · 60 % transition · 20 % JVP
    """

    # --- alpha-flow schedule (fractions of num_train_steps) ---
    warmup_ratio:     float = 0.4   # fraction at which alpha starts decreasing
    transition_ratio: float = 0.6   # fraction at which alpha reaches 0 (JVP starts)
    alpha_gamma:      float = 25.0  # sigmoid temperature (sharpness of transition)
    alpha_min:        float = 5e-3  # snap-to-boundary threshold (eta in paper)

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0AlphaFlowConfig requires pi05=True")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0AlphaFlow":
        return Pi0AlphaFlow(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gemma_width(config: pi0_config.Pi0Config) -> int:
    """Return the action-expert hidden width."""
    import openpi.models.gemma as _gemma
    return _gemma.get_config(config.action_expert_variant).width
