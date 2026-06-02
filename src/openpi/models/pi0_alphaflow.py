"""
Alpha-Flow fine-tuning for Pi05.

Creates a 1-NFE policy by fine-tuning a pretrained Pi05 model with the
discrete alpha-Flow curriculum objective.  The full objective lives in
`compute_loss`, so the standard scripts/train.py drives training with no
special handling — the alpha schedule is advanced by an internal step counter.

Usage:
    ./train.sh pi05_alphaflow_tabletop_bc_orig <gpus> <batch/gpu> <steps>

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

def alpha_schedule(step, warmup_end: int, transition_end: int, gamma: float = 25.0,
                   eta: float = 5e-3, floor: float = 0.0):
    """
    Scaled-sigmoid curriculum schedule (AlphaFlow / AlphaFlowTSE).  JIT-safe.

    Three phases, defined by step fractions so they map to clean % of training:
        step <  warmup_end                  → 1.0          (FM warmup)
        warmup_end ≤ step < transition_end  → 1 → end_val  (smooth transition)
        step ≥ transition_end               → end_val      (floor)

    The transition uses the *scaled* sigmoid  α = 1 + (end_val-1)·σ(γ·progress),
    progress = (step-mid)/width ∈ [-0.5, 0.5].  This reaches end_val *exactly* at
    transition_end (unlike a 1→0 sigmoid hard-clamped at the floor, which would
    hit the floor early and plateau).  So the % of steps spent in each phase is
    exactly: warmup_ratio FM, (transition_ratio-warmup_ratio) transition,
    (1-transition_ratio) floor.

    floor=0.0 (use_jvp=True):  end_val=0 and the near-zero tail snaps to exactly
        0 (below eta) so the JVP MeanFlow branch triggers at the end.
    floor>0.0 (use_jvp=False): end_val=floor; α descends to the floor and stays
        there (discrete-only MeanFlow — stable, the paper's recommendation).

    Degenerate (transition_end ≤ warmup_end, e.g. both 0): no curriculum →
        α == end_val from step 0 ("MeanFlow from scratch" ablation).
    """
    end_val = float(floor) if floor > 0.0 else 0.0
    step_f = step.astype(jnp.float32)

    # No curriculum: alpha is constant at end_val for every step.
    if transition_end <= warmup_end:
        return jnp.full_like(step_f, end_val)

    midpoint = jnp.float32((warmup_end + transition_end) / 2.0)
    width    = jnp.float32(float(transition_end - warmup_end))

    progress = (step_f - midpoint) / width                       # [-0.5, 0.5]
    raw = 1.0 + (end_val - 1.0) * jax.nn.sigmoid(gamma * progress)  # 1 → end_val
    if floor <= 0.0:
        raw = jnp.where(raw < eta, 0.0, raw)   # snap near-zero tail to 0 → JVP branch

    alpha = jnp.where(step_f <  jnp.float32(warmup_end),     1.0, raw)
    alpha = jnp.where(step_f >= jnp.float32(transition_end), end_val, alpha)
    return alpha


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


def _bounded_l2_loss(error, alpha, kappa: float = 1.0, eps: float = 1e-3):
    """Bounded alpha^-1 reweighting from AlphaFlowTSE (2603.10701, Eq.18):

        l_bnd(D; a) = sg[ kappa / (m(D) + a*kappa + eps) ] * m(D),   m(D)=mean(D^2)

    vs the plain `_adaptive_l2_loss(weight_scale=alpha)` = alpha/(m+c)*m, which
    SUPPRESSES the loss as alpha->0 (the small-alpha mean-velocity samples we
    actually want to learn).  This bounded form instead AMPLIFIES informative
    (small-alpha) samples — weight -> kappa/(m+eps) ≈ full FM strength as a->0 —
    while the a*kappa term in the denominator caps the weight so it never blows
    up.  alpha only modulates the saturation point, not a multiplicative gate.
    """
    m = jnp.mean(jnp.square(error), axis=-1)               # (...) = (B, ah)
    w = kappa / (m + alpha * kappa + eps)
    w = jax.lax.stop_gradient(w)
    return w * m


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Pi0AlphaFlow(Pi0):
    """
    Pi05 extended with alpha-Flow fine-tuning support.

    Changes vs. the base Pi0 class:
    - Adds `r_proj`: a **zero-initialized** linear that adds target-timestep r
      conditioning to the adaRMS time embedding.  At init its output is 0, so
      the pretrained behaviour is preserved.
    - `embed_suffix_with_r`: wraps the parent `embed_suffix` and adds the r residual.
    - `compute_loss`: full alpha-Flow objective.  An internal `train_step` counter
      drives the alpha schedule; `lax.cond` picks discrete vs JVP MeanFlow at
      runtime (only one branch executes — no JVP cost during phases 1-2).
    - Shared prefix KV: the backbone runs over the image+language prefix ONCE
      per step; the action suffix passes reuse the cached KV.
    - `sample_actions_1nfe` / `sample_actions_nfe`: 1-step / N-step inference.
    """

    def __init__(self, config: "Pi0AlphaFlowConfig", rngs: nnx.Rngs):
        assert config.pi05, "Pi0AlphaFlow requires pi05=True"
        super().__init__(config, rngs)

        # Target-timestep (r) conditioning.  The official alpha-flow embeds the
        # current time σ=t AND the target time σ_next=r with TWO equal-capacity
        # TimestepEmbedders and sums them — strong r-conditioning is essential for
        # the mean velocity (which depends on the interval [r, t]).  We mirror that:
        # r goes through a full 2-layer MLP (same as the pretrained t time_mlp),
        # and its OUTPUT layer is zero-init so the conditioning equals the
        # pretrained t-only behaviour at the start of fine-tuning (the hidden
        # layer still has full, non-zero capacity to learn r-dependence).
        action_expert_cfg = _gemma_width(config)
        self.r_mlp_in = nnx.Linear(action_expert_cfg, action_expert_cfg, rngs=rngs)
        self.r_mlp_out = nnx.Linear(
            action_expert_cfg,
            action_expert_cfg,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )

        # Alpha schedule: ratio → absolute steps (so train.py needs no changes).
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

        # Internal training-step counter (non-trainable nnx state, threaded
        # through jit/optimizer like BatchNorm stats).  Incremented once per
        # compute_loss call so the alpha schedule advances automatically.
        self.train_step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    # ------------------------------------------------------------------
    # r conditioning
    # ------------------------------------------------------------------

    def embed_suffix_with_r(self, obs, noisy_actions, timestep, r):
        """Embed suffix with additional target-timestep r conditioning.

        Mirrors the pretrained t time_mlp (posemb → in → swish → out → swish)
        for r, and sums it onto the adaRMS cond:  cond = time_mlp(t) + r_mlp(r).
        r_mlp_out is zero-init, so swish(0)=0 → r contributes nothing at the
        start (pretrained behaviour preserved) but has full capacity to learn.
        """
        tokens, mask, ar_mask, adarms_cond = self.embed_suffix(obs, noisy_actions, timestep)
        # delta_conditioning: embed the interval length Δ (AlphaFlowTSE) instead of
        # r directly.  Under our t≥r convention the (non-negative) interval length
        # is t-r, matching the paper's Δ=r-t≥0 (their t<r).  At init r_mlp_out is
        # zero so both choices are identical; they diverge only as r_mlp learns.
        r_in = (timestep - r) if self.delta_conditioning else r
        r_emb = posemb_sincos(r_in, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        r_emb = self.r_mlp_in(r_emb)
        r_emb = nnx.swish(r_emb)
        r_emb = self.r_mlp_out(r_emb)
        r_emb = nnx.swish(r_emb)
        return tokens, mask, ar_mask, adarms_cond + r_emb

    # ------------------------------------------------------------------
    # Sphere noise sampling  (same prior as LPS)
    # ------------------------------------------------------------------

    def _sample_sphere_noise(self, rng, shape):
        """
        Sample from the hypersphere prior used in LPS.

        Each sample lies on a sphere of radius sqrt(action_horizon * action_dim),
        matching the expected L2 norm of a standard Gaussian in that space.
        """
        e = jax.random.normal(rng, shape)
        flat = e.reshape(shape[0], -1)                             # (B, H*D)
        norm = jnp.sqrt(jnp.sum(jnp.square(flat), axis=-1, keepdims=True) + 1e-6)
        scale = jnp.sqrt(jnp.float32(flat.shape[-1]))             # sqrt(H * D)
        return (flat / norm * scale).reshape(shape)

    def _sample_noise(self, rng, shape):
        """Sample the flow latent prior.

        sphere_latent=True  → hypersphere prior (LPS, default).
        sphere_latent=False → standard Gaussian (matches the pretrained pi05).
        Toggleable so we can A/B test whether the sphere prior is the issue.
        """
        if self.sphere_latent:
            return self._sample_sphere_noise(rng, shape)
        return jax.random.normal(rng, shape)

    # ------------------------------------------------------------------
    # Expert-layout hooks (overridden by subclasses with more experts)
    # ------------------------------------------------------------------

    def _experts_prefix(self, prefix_tokens):
        """Token list for the prefix-only KV pass. [paligemma, action]."""
        return [prefix_tokens, None]

    def _experts_action(self, action_tokens):
        """Token list for an action-suffix pass against the cached prefix."""
        return [None, action_tokens]

    def _adarms_action(self, adarms_cond):
        """adaRMS conditioning list aligned with _experts_action."""
        return [None, adarms_cond]

    def _action_out_index(self) -> int:
        """Index of the action expert in the llm output list."""
        return 1

    # ------------------------------------------------------------------
    # Shared prefix KV (computed ONCE, reused by every suffix pass)
    # ------------------------------------------------------------------

    def _embed_prefix_kv(self, observation):
        """Run the backbone over the prefix ONCE; return (kv_cache, prefix_mask).

        The prefix never attends to the suffix, so its per-layer K/V are
        independent of which suffix (action / critic, z_t / z_s) follows.
        Computing it once and reusing the KV cache avoids recomputing the
        ~768-token image+language prefix through the 2B backbone on every pass.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            self._experts_prefix(prefix_tokens), mask=attn_mask, positions=positions
        )
        return kv_cache, prefix_mask

    def _suffix_attn_and_positions(self, prefix_mask, suffix_mask, suffix_ar_mask):
        """Build the (suffix → prefix+suffix) attention mask and suffix positions."""
        b, s = suffix_mask.shape
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = jnp.broadcast_to(prefix_mask[:, None, :], (b, s, prefix_mask.shape[1]))
        full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        return full_attn, positions

    def _action_velocity(self, kv_cache, prefix_mask, observation, noisy_actions, t, r,
                         compute_dtype=None):
        """Velocity prediction for one action suffix, reusing the cached prefix KV.

        compute_dtype: if set (e.g. jnp.float32), the suffix tokens + adaRMS cond
            are cast to it so the WHOLE suffix forward runs in that precision
            (gemma upcasts weights to the activation dtype).  Used by the JVP
            branch: forward-mode AD (dudt) is far more precision-sensitive than a
            plain forward eval, and the default bf16 makes the tangent noisy.
            The cached prefix KV (bf16) is promoted at the attention concat.
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_r(
            observation, noisy_actions, t, r
        )
        if compute_dtype is not None:
            suffix_tokens = suffix_tokens.astype(compute_dtype)
            if adarms_cond is not None:
                adarms_cond = adarms_cond.astype(compute_dtype)
        full_attn, positions = self._suffix_attn_and_positions(prefix_mask, suffix_mask, suffix_ar_mask)
        outs, _ = self.PaliGemma.llm(
            self._experts_action(suffix_tokens),
            mask=full_attn,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=self._adarms_action(adarms_cond),
        )
        suffix_out = outs[self._action_out_index()]
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    # ------------------------------------------------------------------
    # Alpha schedule (reads internal step counter)
    # ------------------------------------------------------------------

    def _alpha_now(self):
        """Current alpha (JAX scalar) from the internal training-step counter.

        use_jvp=False → alpha floors at alpha_min (stays in the discrete branch).
        use_jvp=True  → alpha snaps to 0 at the end (JVP MeanFlow branch).
        """
        floor = 0.0 if self.use_jvp else self._alpha_min
        return alpha_schedule(
            self.train_step.value,
            self._warmup_end, self._transition_end,
            self._alpha_gamma, self._alpha_eta, floor,
        )

    # ------------------------------------------------------------------
    # Flow input sampling (shared by both branches)
    # ------------------------------------------------------------------

    def _sample_two_times(self, t_rng, r_rng, batch_shape):
        """Sample two timesteps in [0,1] per the configured marginal.

        time_sampler="minmax": sigmoid(normal·1 - 0.4)   (official alpha-flow).
        time_sampler="beta":   beta(1.5, 1)·0.999 + 0.001 (matches pretrained pi05).
        """
        if self.time_sampler == "beta":
            t1 = jax.random.beta(t_rng, 1.5, 1.0, batch_shape) * 0.999 + 0.001
            t2 = jax.random.beta(r_rng, 1.5, 1.0, batch_shape) * 0.999 + 0.001
        else:  # "minmax"
            t1 = jax.nn.sigmoid(jax.random.normal(t_rng, batch_shape) * 1.0 - 0.4)
            t2 = jax.nn.sigmoid(jax.random.normal(r_rng, batch_shape) * 1.0 - 0.4)
        return t1, t2

    def _sample_flow_inputs(self, rng, actions, flow_ratio: float, alpha=None):
        """Sample (noise, t, r, n_fm) for the alpha-flow objective.

        FM border samples (first n_fm): r = t  (pure TFM supervision).
        MF samples (rest):              r < t  (t = max, r = min).

        large_span_ratio>0: force the LAST n_ls MF samples to span ~[0,1]
        (t≥0.85, r≤0.15), so training sees the full-interval transport that
        1-NFE inference (t=1, r=0) actually uses (AlphaFlowTSE).

        large_span_warmup_gate (with alpha given): skip the large-span overwrite
        while alpha==1 (FM warmup) — there the target is v (r-independent), so
        large-span samples teach nothing about the interval.  They turn on once
        alpha<1 (transition + floor).
        """
        noise_rng, t_rng, r_rng, ls_rng = jax.random.split(rng, 4)
        b           = actions.shape[0]
        batch_shape = actions.shape[:-2]
        noise = self._sample_noise(noise_rng, actions.shape)

        n_fm  = int(b * flow_ratio)
        t1, t2 = self._sample_two_times(t_rng, r_rng, batch_shape)
        t_all = jnp.maximum(t1, t2)
        r_mf  = jnp.minimum(t1, t2)
        t     = t_all
        r     = jnp.concatenate([t_all[:n_fm], r_mf[n_fm:]], axis=0)

        n_ls = int((b - n_fm) * self._large_span_ratio)
        if n_ls > 0:
            ls_t_rng, ls_r_rng = jax.random.split(ls_rng)
            t_big = 0.85 + 0.15 * jax.random.uniform(ls_t_rng, batch_shape)   # [0.85, 1.0]
            r_big = 0.15 * jax.random.uniform(ls_r_rng, batch_shape)          # [0.0, 0.15]
            idx = b - n_ls                                                    # last n_ls are MF
            if self._large_span_warmup_gate and alpha is not None:
                # JIT-safe phase gate: apply large-span only past FM warmup
                # (alpha<1).  ls_pos is a static [b] mask of the last n_ls slots.
                ls_pos = jnp.arange(b) >= idx
                gate   = ls_pos & (alpha < 1.0)                               # [b]
                t = jnp.where(gate, t_big, t)
                r = jnp.where(gate, r_big, r)
            else:
                t = t.at[idx:].set(t_big[idx:])
                r = r.at[idx:].set(r_big[idx:])
        return noise, t, r, n_fm

    # ------------------------------------------------------------------
    # Discrete / JVP branches (take a pre-computed prefix KV)
    # ------------------------------------------------------------------

    def _discrete_branch(self, prefix_kv, observation, actions, noise, t, r, n_fm, alpha,
                         utgt_clamp: float = 10.0):
        """Discrete alpha-flow target (matches official loss.py).  Returns [b, ah].

        prefix_kv = (kv_cache, prefix_mask) — the shared, pre-computed prefix.
        """
        kv, pm = prefix_kv
        t_e = t[..., None, None]
        s   = alpha * r + (1.0 - alpha) * t
        s_e = s[..., None, None]
        z_t = t_e * noise + (1.0 - t_e) * actions
        v_t = noise - actions
        z_s = z_t - (t_e - s_e) * v_t

        u_pred = self._action_velocity(kv, pm, observation, z_t, t, r)
        u_next = jax.lax.stop_gradient(
            self._action_velocity(kv, pm, observation, z_s, s, r)
        )
        u_tgt = jax.lax.stop_gradient(
            jnp.clip(alpha * v_t + (1.0 - alpha) * u_next, -utgt_clamp, utgt_clamp)
        )
        err = u_pred - u_tgt
        # FM-border anchor: plain adaptive L2 (weight 1/(m+c)).
        loss_fm = self._lambda_fm * _adaptive_l2_loss(err[:n_fm], weight_scale=1.0)
        # MF-sample reweighting.
        #   "bounded" (AlphaFlowTSE Eq.18): kappa/(m+alpha*kappa+eps) — does NOT
        #       suppress small-alpha samples (the mean-velocity we want to learn).
        #   "adaptive" (orig): alpha/(m+c), or a fixed mf_loss_weight to decouple
        #       the loss weight from the schedule alpha.
        if self._mf_reweight == "bounded":
            loss_mf = _bounded_l2_loss(err[n_fm:], alpha, kappa=self._reweight_kappa)
        else:
            mf_w = alpha if self._mf_loss_weight is None else self._mf_loss_weight
            loss_mf = _adaptive_l2_loss(err[n_fm:], weight_scale=mf_w)
        loss_mf = self._lambda_mf * loss_mf
        loss = jnp.concatenate([loss_fm, loss_mf], axis=0)        # [b, ah]
        raw_l2 = jnp.mean(jnp.square(err), axis=-1)               # [b, ah]  plain MSE (FM-comparable)
        # lax.cond requires both branches to return identical dtypes — pin to fp32.
        return loss.astype(jnp.float32), raw_l2.astype(jnp.float32)

    def _jvp_branch(self, prefix_kv, observation, actions, noise, t, r, n_fm,
                    utgt_clamp: float = 10.0):
        """Exact MeanFlow target via JVP (alpha = 0 phase).  Returns [b, ah].

        The prefix KV is a closure constant (independent of z, t, r), so jvp
        differentiates only the suffix pass — cheap and exact.
        n_fm is unused (FM border falls out naturally: r=t → (t-r)=0).
        """
        del n_fm
        kv, pm = prefix_kv
        t_e = t[..., None, None]
        r_e = r[..., None, None]
        z_t = t_e * noise + (1.0 - t_e) * actions
        v_t = noise - actions

        # Run the JVP suffix forward in fp32 (jvp_fp32): forward-mode AD is much
        # more precision-sensitive than a plain forward eval, and the default
        # bf16 makes the tangent dudt noisy enough to blow up the bootstrap.
        cdt = jnp.float32 if self.jvp_fp32 else None

        def fn(z_val, t_val, r_val):
            return self._action_velocity(kv, pm, observation, z_val, t_val, r_val,
                                         compute_dtype=cdt)

        u_pred, dudt = jax.jvp(fn, (z_t, t, r), (v_t, jnp.ones_like(t), jnp.zeros_like(r)))
        # Clamp the JVP target like the discrete branch.  When the mean-velocity is
        # under-trained, dudt can be large and u_tgt = v_t - (t-r)*dudt is otherwise
        # UNbounded — the immediate blow-up the moment alpha hits 0.  The discrete
        # branch already clips its target to ±utgt_clamp; matching it here makes the
        # discrete→JVP handoff continuous in target scale.
        u_tgt = jax.lax.stop_gradient(jnp.clip(v_t - (t_e - r_e) * dudt, -utgt_clamp, utgt_clamp))
        err = u_pred - u_tgt
        loss   = _adaptive_l2_loss(err, weight_scale=1.0)         # [b, ah]
        raw_l2 = jnp.mean(jnp.square(err), axis=-1)               # [b, ah]  plain MSE
        # lax.cond requires both branches to return identical dtypes — pin to fp32.
        return loss.astype(jnp.float32), raw_l2.astype(jnp.float32)

    # ------------------------------------------------------------------
    # Alpha-flow loss with shared prefix (lax.cond picks discrete vs JVP)
    # ------------------------------------------------------------------

    def _alphaflow_loss_with_prefix(self, rng, observation, actions, prefix_kv, alpha,
                                    *, flow_ratio: float | None = None, utgt_clamp: float = 10.0):
        """Returns (adaptive loss [b, ah], raw plain-MSE [b, ah]) given a prefix KV.

        The raw MSE (mean over action_dim of (u_pred - u_tgt)²) is on the same
        scale as the pi05 FM loss, so it can be compared directly across runs.

        Runtime picks exactly ONE branch via lax.cond:
          alpha > 0  → discrete alpha-flow (no JVP cost)
          alpha == 0 → exact MeanFlow via JVP
        """
        flow_ratio = self._flow_ratio if flow_ratio is None else flow_ratio
        noise, t, r, n_fm = self._sample_flow_inputs(rng, actions, flow_ratio, alpha=alpha)
        return jax.lax.cond(
            alpha > 0.0,
            lambda: self._discrete_branch(prefix_kv, observation, actions, noise, t, r, n_fm, alpha, utgt_clamp),
            lambda: self._jvp_branch(prefix_kv, observation, actions, noise, t, r, n_fm, utgt_clamp),
        )

    # ------------------------------------------------------------------
    # compute_loss: full alpha-flow objective (train.py calls this)
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
        """Returns (per-token loss [*b, ah], aux metrics dict) for wandb logging."""
        preprocess_rng, flow_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        prefix_kv = self._embed_prefix_kv(observation)   # prefix backbone ONCE
        alpha  = self._alpha_now()
        loss, raw_l2 = self._alphaflow_loss_with_prefix(flow_rng, observation, actions, prefix_kv, alpha)
        # Advance the schedule (non-trainable counter, threaded through jit).
        self.train_step.value = self.train_step.value + 1

        # phase: 1.0=TFM warmup, 0.5=discrete transition, 0.0=JVP MeanFlow
        phase = jnp.where(alpha >= 1.0, 1.0, jnp.where(alpha <= 0.0, 0.0, 0.5))
        aux = {
            "alpha": alpha,
            "phase": phase,
            "loss/alphaflow": jnp.mean(loss),
            "loss/l2_raw":    jnp.mean(raw_l2),   # plain MSE — same scale as pi05 FM loss
        }
        return loss, aux

    # ------------------------------------------------------------------
    # Inference: 1-NFE and N-NFE sampling
    # ------------------------------------------------------------------

    def sample_actions_1nfe(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        One-step (1-NFE) action sampling.

        The model predicts the mean velocity u over the full interval [r=0, t=1]:
            z_0 = z_1 - (t - r) * u = noise - 1.0 * u_pred
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self._sample_noise(rng, (batch_size, self.action_horizon, self.action_dim))

        t = jnp.ones(batch_size)   # t = 1  (pure noise)
        r = jnp.zeros(batch_size)  # r = 0  (target: clean action)

        kv_cache, prefix_mask = self._embed_prefix_kv(observation)
        u = self._action_velocity(kv_cache, prefix_mask, observation, noise, t, r)
        # No clip: return raw z_0 so out-of-range outputs are visible (matches pi0).
        return noise - u

    def sample_actions_nfe(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        mode: str = "mean",
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        Multi-step (N-NFE) sampling.  Splits [0,1] into num_steps Euler steps.

        mode="mean" (MeanFlow): r = t_next  → predicts the mean velocity over the
            step interval [t_next, t_cur].  Correct once the MeanFlow phase has
            trained.  num_steps=1 == sample_actions_1nfe.
        mode="fm" (flow matching): r = t_cur → predicts the *instantaneous*
            velocity (same as the FM-border r=t training samples / pi05).  Use
            this to evaluate the base flow BEFORE any MeanFlow training (e.g. a
            warmup-only 10k checkpoint), via standard Euler ODE integration.

        Prefix KV is computed once and reused across steps.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        b = observation.state.shape[0]
        if noise is None:
            noise = self._sample_noise(rng, (b, self.action_horizon, self.action_dim))

        kv_cache, prefix_mask = self._embed_prefix_kv(observation)
        dt = 1.0 / num_steps

        z = noise
        for i in range(num_steps):
            t_cur  = 1.0 - i * dt
            t_next = t_cur - dt
            t_vec = jnp.full(b, t_cur)
            r_vec = t_vec if mode == "fm" else jnp.full(b, t_next)   # fm: r=t, mean: r=t_next
            u = self._action_velocity(kv_cache, prefix_mask, observation, z, t_vec, r_vec)
            z = z - dt * u
        return z   # no clip (matches pi0)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 1,
        nfe_mode: str = "mean",
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Alpha-flow sampling — honours num_steps and nfe_mode.

        num_steps=1, nfe_mode="mean" → 1-NFE MeanFlow (production).
        num_steps=N, nfe_mode="mean" → N-step mean-velocity integration.
        num_steps=N, nfe_mode="fm"   → N-step instantaneous-velocity Euler ODE
                                        (FM; for base-flow eval before MeanFlow).
        Set via serve_policy --num-steps / --nfe-mode (sample_kwargs).
        """
        return self.sample_actions_nfe(rng, observation, num_steps=num_steps, mode=nfe_mode, noise=noise)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pi0AlphaFlowConfig(pi0_config.Pi0Config):
    """
    Config for Pi0AlphaFlow.  Inherits all Pi0Config fields.  Set pi05=True.

    Defaults = the AlphaFlowTSE (2603.10701) recipe (the validated JVP-free
    design — see docs/alphaflow_troubleshooting.md O7).  Everything below is a
    knob, but the defaults are the base recipe; a plain `Pi0AlphaFlowConfig(
    pi05=True)` already gives it.  Deviations (e.g. use_jvp=True) are A/B toggles.

    Alpha schedule (fractions of num_train_steps), scaled-sigmoid 1 → alpha_min:
        [0,            warmup_ratio)      → alpha = 1.0       (FM warmup)
        [warmup_ratio, transition_ratio) → alpha 1 → end_val (transition)
        [transition_ratio, 1.0]          → alpha = end_val    (floor)
    end_val = alpha_min (use_jvp=False, discrete) or 0 (use_jvp=True, JVP).
    """

    # --- alpha schedule (fractions of num_train_steps) ---
    warmup_ratio:     float = 0.05   # FM (alpha=1) phase
    transition_ratio: float = 0.667  # end of the 1→alpha_min sigmoid (then floor)
    alpha_gamma:      float = 15.0   # sigmoid steepness (paper k=15)
    # alpha_min = the DISCRETE FLOOR (use_jvp=False): alpha never goes below this
    # (AlphaFlowTSE: 0.1, well away from the JVP limit).  NOT the snap threshold.
    alpha_min:        float = 0.1
    # alpha_eta = boundary-snap threshold (paper's eta): raw alpha above 1-eta
    # snaps to 1, and (use_jvp=True) below eta snaps to 0 → JVP.  Stays SMALL,
    # decoupled from alpha_min so raising the floor doesn't distort the warmup.
    alpha_eta:        float = 5e-3

    # --- MeanFlow estimator ---
    # use_jvp=False (default): discrete teacher-student bootstrap (stable, bf16 OK).
    # use_jvp=True:  exact MeanFlow via JVP at alpha=0 (precision-sensitive — runs
    #   in fp32 when jvp_fp32; see O6).  Set transition_ratio<1 to carve a JVP phase.
    use_jvp:          bool  = False
    jvp_fp32:         bool  = True   # run the JVP suffix forward in fp32 (O6)

    # --- loss reweighting ---
    # "bounded" (default, AlphaFlowTSE Eq.18): kappa/(m+alpha*kappa+eps) — does NOT
    #   suppress the small-alpha mean-velocity samples (the real fix).
    # "adaptive": alpha/(m+c), or a fixed mf_loss_weight to decouple from alpha.
    mf_reweight:      str   = "bounded"
    reweight_kappa:   float = 1.0
    mf_loss_weight:   float | None = None  # only used when mf_reweight="adaptive"
    lambda_fm:        float = 0.6   # FM-branch loss weight  (AlphaFlowTSE)
    lambda_mf:        float = 0.4   # MF-branch loss weight  (AlphaFlowTSE)

    # --- time-pair sampling ---
    flow_ratio:       float = 0.5   # FM-border fraction (r=t pure flow-matching)
    # Fraction of MF samples from the large-span subset (t≥0.85, r≤0.15) so training
    # sees the full-interval transport that 1-NFE inference (t=1, r=0) uses.
    large_span_ratio: float = 0.15
    # Gate large-span sampling to the post-warmup phase (alpha<1).  During FM
    # warmup (alpha=1) the target is v (r-independent), so forced large-span
    # samples teach nothing about the interval — skip them and only oversample
    # the full span once mean-velocity learning has begun (transition + floor).
    large_span_warmup_gate: bool = True
    # "minmax" = sigmoid(normal-0.4) (official); "beta" = beta(1.5,1) (pi05). A/B.
    time_sampler:     str   = "minmax"

    # --- conditioning / prior ---
    # emb(t)+emb(Δ), Δ=t-r (AlphaFlowTSE) when True; emb(t)+emb(r) when False.
    delta_conditioning: bool = True
    # Latent prior: True = hypersphere (LPS), False = standard Gaussian (pi05).
    sphere_latent:    bool  = True

    # Total training steps — used to convert schedule ratios → absolute steps.
    # The model tracks its own step counter internally so train.py needs no changes.
    num_train_steps:  int   = 30_000

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
