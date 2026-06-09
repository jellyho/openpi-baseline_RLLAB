"""
Pi0 RLT — "RL Token" bottleneck on top of a frozen Pi0.5 VLA.

Implements the representation-learning stage of *RL Token: Bootstrapping Online
RL with Vision-Language-Action Models* (Xu et al., Physical Intelligence,
arXiv:2604.23073).  We add a small **encoder–decoder bottleneck** to a frozen
pi05 policy that compresses the VLA's internal features into one compact
``RL token`` (z_rl).  That token is the state representation a lightweight
actor–critic later consumes for sample-efficient online RL.

NOTE: this is the paper's encoder–decoder *autoregressive reconstruction*
bottleneck (Fig. 2 / Eq. 1–2), NOT a variational autoencoder — there is no KL
term and no sampling.  z_rl is a deterministic readout.

We encode only the IMAGE embeddings and the proprioceptive state (the task's
language instruction is fixed, so it is dropped — paper Sec. IV-A footnote).

Architecture (everything in the VLA is FROZEN during this stage)
────────────────────────────────────────────────────────────────
    PaliGemma (VLM, FROZEN)  ──►  z_{1:M}  = final-layer IMAGE embeddings
                                  (image tokens only; language dropped)
    proprio  (obs.state)     ──►  continuous proprio token (explicitly added)

    Encoder g_φ (bidirectional transformer)
        input  : [proj(z_{1:M}), proj(proprio), e_rl<rl-token>]
        output : z_rl = readout at the <rl> position  →  bottleneck (rlt_token_dim)

    Decoder d_φ (causal transformer, teacher-forced)
        autoregressively reconstructs z̄_{1:M} from [z_rl, z̄_{1:i-1}]  (Eq. 2)
        + a small head reconstructs proprio from z_rl
        (z̄ = stop_gradient(z): the VLA is frozen w.r.t. the reconstruction loss)

Only the ``rlt_*`` parameters train (see ``get_freeze_filter``); the VLA backbone
and action expert keep their pretrained weights.  Inheriting ``Pi0`` keeps the
base policy's ``sample_actions`` intact so the frozen VLA can still propose
actions for the downstream RL stage.

Inference
─────────
    extract_rl_token(observation) -> z_rl [b, rlt_token_dim]
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


# ---------------------------------------------------------------------------
# Small standalone transformer blocks (separate from the Gemma backbone)
# ---------------------------------------------------------------------------


def _sincos_posemb(length: int, dim: int) -> jax.Array:
    """Fixed sinusoidal positional embedding, shape [length, dim] (dim even)."""
    pos = jnp.arange(length, dtype=jnp.float32)[:, None]
    i = jnp.arange(dim // 2, dtype=jnp.float32)[None, :]
    freq = jnp.exp(-jnp.log(10000.0) * (2.0 * i / dim))
    ang = pos * freq
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # [length, dim]


class _Mlp(nnx.Module):
    def __init__(self, dim: int, hidden: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))


class _Block(nnx.Module):
    """Pre-norm Transformer block (self-attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, decode=False, dropout_rate=0.0, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = _Mlp(dim, mlp_hidden, rngs=rngs)

    def __call__(self, x, mask):
        # mask: bool, broadcastable to [b, num_heads, q, kv]; True = attend.
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pi0RLTConfig(pi0_config.Pi0Config):
    """Config for Pi0RLT.  Inherits all Pi0Config fields; requires ``pi05=True``.

    The VLA backbone + action expert are frozen; only the RL-token encoder /
    decoder (and the proprio reconstruction head) train.  Initialize from a
    pi05 (base or task-finetuned) checkpoint via ``AlphaFlowWeightLoader`` so
    the new ``rlt_*`` params keep their initialised values.
    """

    # Bottleneck size of the RL token (paper value: 2048).
    rlt_token_dim: int = 2048
    # Hidden width of the encoder/decoder transformer (d_model).
    rlt_width: int = 1024
    rlt_encoder_depth: int = 4
    rlt_decoder_depth: int = 4
    rlt_num_heads: int = 8
    rlt_mlp_ratio: int = 4
    # Weight on the proprio reconstruction term (forces proprio into the bottleneck).
    proprio_loss_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0RLTConfig requires pi05=True")
        if self.rlt_width % 2 != 0:
            raise ValueError(f"rlt_width must be even (sincos posemb), got {self.rlt_width}")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RLT":
        return Pi0RLT(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze everything except the RL-token modules (``rlt_*``).

        Trainable: rlt_enc_in_proj, rlt_proprio_in_proj, rlt_token_embed,
        rlt_encoder, rlt_out_proj, rlt_dec_rl_proj, rlt_dec_tgt_proj,
        rlt_decoder, rlt_dec_out_proj, rlt_proprio_out_proj.
        Frozen: SigLIP, PaliGemma (VLM), action expert.
        """
        return nnx.Not(nnx_utils.PathRegex(".*rlt_.*"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Pi0RLT(Pi0):
    """Pi0.5 (frozen) + an encoder–decoder RL-token bottleneck."""

    def __init__(self, config: Pi0RLTConfig, rngs: nnx.Rngs):
        # Build the full frozen pi05 (PaliGemma img+llm, action expert).
        super().__init__(config, rngs)

        vlm_width = _gemma.get_config(config.paligemma_variant).width  # W (e.g. 2048)
        d = config.rlt_width
        mlp_hidden = d * config.rlt_mlp_ratio

        self._vlm_width = vlm_width
        self.rlt_width = d
        self.rlt_token_dim = config.rlt_token_dim
        self.proprio_loss_weight = config.proprio_loss_weight

        self._enc_depth = config.rlt_encoder_depth
        self._dec_depth = config.rlt_decoder_depth

        # ── Encoder ─────────────────────────────────────────────────────────
        # Blocks are kept in an nnx.Dict (string keys) rather than a Python list:
        # the weight loader flattens the param tree with flax.traverse_util
        # (sep="/"), which cannot join integer list indices.  String keys keep
        # the checkpoint merge working.
        self.rlt_enc_in_proj = nnx.Linear(vlm_width, d, rngs=rngs)            # W → d
        self.rlt_proprio_in_proj = nnx.Linear(config.action_dim, d, rngs=rngs)  # proprio → d
        self.rlt_token_embed = nnx.Param(jax.random.normal(rngs.params(), (d,)) * 0.02)
        self.rlt_encoder = nnx.Dict(
            {f"blk_{i}": _Block(d, config.rlt_num_heads, mlp_hidden, rngs=rngs) for i in range(self._enc_depth)}
        )
        self.rlt_out_proj = nnx.Linear(d, config.rlt_token_dim, rngs=rngs)   # d → bottleneck

        # ── Decoder (autoregressive reconstruction) ─────────────────────────
        self.rlt_dec_rl_proj = nnx.Linear(config.rlt_token_dim, d, rngs=rngs)  # z_rl → d (start token)
        self.rlt_dec_tgt_proj = nnx.Linear(vlm_width, d, rngs=rngs)            # z̄ → d (teacher forcing)
        self.rlt_decoder = nnx.Dict(
            {f"blk_{i}": _Block(d, config.rlt_num_heads, mlp_hidden, rngs=rngs) for i in range(self._dec_depth)}
        )
        self.rlt_dec_out_proj = nnx.Linear(d, vlm_width, rngs=rngs)           # d → W (reconstruction h_φ)
        # Proprio reconstruction head straight off the bottleneck.
        self.rlt_proprio_out_proj = nnx.Linear(config.rlt_token_dim, config.action_dim, rngs=rngs)

        self.deterministic = True

    # ------------------------------------------------------------------
    # Frozen VLA prefix → final-layer token embeddings z_{1:M}
    # ------------------------------------------------------------------

    def _image_prefix(self, observation: _model.Observation):
        """Embed ONLY the image tokens as the backbone prefix (no language).

        Mirrors the image branch of ``Pi0.embed_prefix`` but skips the language
        tokens entirely, so the frozen-backbone forward runs over a shorter
        (image-only) prefix — faster, and the fixed task instruction is unneeded
        for the RL token anyway (paper Sec. IV-A footnote).  Image tokens attend
        to each other (ar_mask = False).
        """
        tokens, input_mask, ar_mask = [], [], []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _prefix_embeddings(self, observation: _model.Observation):
        """Run the frozen backbone over the IMAGE prefix once; return (z, img_mask).

        z = stop_gradient(final-layer image hidden states), float32, [b, N, W].
        The stop-gradient keeps the VLA frozen w.r.t. the reconstruction loss
        (Eq. 2 uses z̄ = sg(z)); the backbone is also excluded from training by
        the freeze filter.  Only image tokens are fed through the backbone (no
        language) — both for the speedup and because the instruction is fixed.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self._image_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        z = jax.lax.stop_gradient(outs[0].astype(jnp.float32))  # [b, N, W]
        return z, prefix_mask

    # ------------------------------------------------------------------
    # Encoder g_φ : (z, proprio, <rl>) → z_rl bottleneck
    # ------------------------------------------------------------------

    def _encode_rl_token(self, z, prefix_mask, state):
        """Compress the VLA embeddings + proprio into one RL token [b, rlt_token_dim]."""
        b, M, _ = z.shape
        d = self.rlt_width

        zt = self.rlt_enc_in_proj(z)                                  # [b, M, d]
        pt = self.rlt_proprio_in_proj(state)[:, None, :]              # [b, 1, d]
        rl = jnp.broadcast_to(self.rlt_token_embed.value[None, None], (b, 1, d))  # [b, 1, d]
        x = jnp.concatenate([zt, pt, rl], axis=1)                     # [b, M+2, d]
        x = x + _sincos_posemb(M + 2, d)[None]                        # positional

        # Bidirectional: every (valid) query attends to every valid key.  The
        # proprio token and the <rl> token are always valid.
        valid = jnp.concatenate([prefix_mask, jnp.ones((b, 2), dtype=jnp.bool_)], axis=1)  # [b, M+2]
        mask = valid[:, None, None, :]                               # [b, 1, 1, M+2]

        for i in range(self._enc_depth):
            x = self.rlt_encoder[f"blk_{i}"](x, mask)
        return self.rlt_out_proj(x[:, -1])                           # [b, rlt_token_dim]

    # ------------------------------------------------------------------
    # Decoder d_φ : autoregressive reconstruction of z̄_{1:M} from z_rl
    # ------------------------------------------------------------------

    def _decode(self, z_rl, z_tgt, prefix_mask):
        """Teacher-forced AR reconstruction; returns recon [b, M, W].

        Input sequence (length M): [z_rl, z̄_1, ..., z̄_{M-1}] with a causal mask,
        so position j (seeing z_rl + z̄_{1:j}) predicts z̄_{j+1} (Eq. 2).
        """
        b, M, _ = z_tgt.shape
        d = self.rlt_width

        start = self.rlt_dec_rl_proj(z_rl)[:, None, :]               # [b, 1, d]
        shifted = self.rlt_dec_tgt_proj(z_tgt[:, : M - 1])          # [b, M-1, d]
        x = jnp.concatenate([start, shifted], axis=1)               # [b, M, d]
        x = x + _sincos_posemb(M, d)[None]

        causal = jnp.tril(jnp.ones((M, M), dtype=jnp.bool_))        # [M, M]
        # Key validity: position 0 (z_rl) always valid; position j (z̄_{j}) valid
        # iff the corresponding prefix token is valid.
        kv_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), prefix_mask[:, : M - 1]], axis=1)  # [b, M]
        mask = causal[None, None] & kv_valid[:, None, None, :]      # [b, 1, M, M]

        for i in range(self._dec_depth):
            x = self.rlt_decoder[f"blk_{i}"](x, mask)
        return self.rlt_dec_out_proj(x)                            # [b, M, W]

    # ------------------------------------------------------------------
    # Inference: extract the RL token
    # ------------------------------------------------------------------

    def extract_rl_token(self, observation: _model.Observation) -> at.Float[at.Array, "b t"]:
        """Frozen forward → RL token z_rl [b, rlt_token_dim] for downstream RL."""
        observation = _model.preprocess_observation(None, observation, train=False)
        z, prefix_mask = self._prefix_embeddings(observation)
        return self._encode_rl_token(z, prefix_mask, observation.state)

    # ------------------------------------------------------------------
    # Base-VLA action sampling (precompute reference action chunks)
    # ------------------------------------------------------------------

    def sample_base_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_samples: int,
        num_steps: int = 10,
    ) -> at.Float[at.Array, "b n ah ad"]:
        """Sample ``num_samples`` base-VLA action chunks per state.

        Returns [b, num_samples, H, action_dim] in normalized model space — the
        SAME distribution as ``Pi0.sample_actions`` (π_vla), just drawn many times
        per state.  These are the reference action chunks ã ~ π_vla that the
        downstream RLT actor conditions on / is regularized toward.

        Efficiency: the full VLA prefix (2B backbone over image+language+state) is
        run ONCE per state; the action-expert flow-matching denoising is vmapped
        over the ``num_samples`` noises (the action expert is small, so this is
        cheap relative to re-running the backbone per sample).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        b = observation.state.shape[0]
        dt = -1.0 / num_steps

        # Prefix KV once (this is the real π_vla prefix: images + language + state).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def denoise(noise):
            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, b)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn, positions=pos,
                    kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            def cond(carry):
                return carry[1] >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        noises = jax.random.normal(rng, (num_samples, b, self.action_horizon, self.action_dim))
        chunks = jax.vmap(denoise)(noises)              # [n, b, H, D]
        return jnp.transpose(chunks, (1, 0, 2, 3))       # [b, n, H, D]

    # ------------------------------------------------------------------
    # compute_loss: autoregressive reconstruction (Eq. 2) + proprio
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,  # noqa: ARG002  (unused: this stage trains the bottleneck only)
        *,
        train: bool = False,
    ):
        """Returns (per-sample loss [b], aux dict).

        L_ro = mean_i ‖ h_φ(d_φ([z_rl, z̄_{1:i-1}]))_i − z̄_i ‖²   (over valid prefix tokens)
             + proprio_loss_weight · ‖ proprio_head(z_rl) − state ‖²
        """
        observation = _model.preprocess_observation(rng, observation, train=train)
        state = observation.state

        z, prefix_mask = self._prefix_embeddings(observation)        # [b, M, W], [b, M]
        z_rl = self._encode_rl_token(z, prefix_mask, state)          # [b, rlt_token_dim]
        recon = self._decode(z_rl, z, prefix_mask)                  # [b, M, W]

        # Masked reconstruction MSE (mean over feature dim, masked mean over tokens).
        err = jnp.mean(jnp.square(recon - z), axis=-1)             # [b, M]
        m = prefix_mask.astype(jnp.float32)
        recon_loss = jnp.sum(err * m, axis=-1) / (jnp.sum(m, axis=-1) + 1e-6)  # [b]

        proprio_recon = self.rlt_proprio_out_proj(z_rl)            # [b, ad]
        proprio_loss = jnp.mean(jnp.square(proprio_recon - state), axis=-1)    # [b]

        total = recon_loss + self.proprio_loss_weight * proprio_loss  # [b]

        z_rl_norm = jnp.linalg.norm(z_rl, axis=-1)
        aux = {
            "loss/recon": jnp.mean(recon_loss),
            "loss/proprio": jnp.mean(proprio_loss),
            "loss/total": jnp.mean(total),
            "rlt/z_abs_mean": jnp.mean(jnp.abs(z_rl)),
            "rlt/z_norm_mean": jnp.mean(z_rl_norm),
            "rlt/z_batch_std": jnp.mean(jnp.std(z_rl, axis=0)),  # ~0 ⇒ token collapse
            "rlt/target_abs_mean": jnp.mean(jnp.abs(z)),
        }
        return total, aux


# ===========================================================================
# Pi0RLTJoint — single-forward variant (language INCLUDED in the RL token)
# ===========================================================================
#
# The base Pi0RLT runs the backbone TWICE per state during annotation/inference:
# an image-only pass for the RL token, and the full π_vla pass (image+language+
# state) for action sampling.  Pi0RLTJoint removes the extra pass by sourcing the
# RL token from the IMAGE-token hidden states of the SAME full forward used for
# sampling.  Because the full prefix is bidirectional, those image embeddings are
# now LANGUAGE-conditioned (the token is no longer language-invariant) — which is
# fine, and arguably better, for a generalist whose instruction varies.  This
# changes the token definition, so a Joint model must be trained from scratch and
# is NOT checkpoint-compatible with a vanilla Pi0RLT.


@dataclasses.dataclass(frozen=True)
class Pi0RLTJointConfig(Pi0RLTConfig):
    """Pi0RLT with a single (language-included) backbone forward for the RL token."""

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RLTJoint":
        return Pi0RLTJoint(self, rngs=nnx.Rngs(rng))


class Pi0RLTJoint(Pi0RLT):
    """Pi0RLT whose RL token comes from the full (image+language) prefix forward."""

    def _num_image_tokens(self, observation: _model.Observation, prefix_len: int) -> int:
        """Image tokens are the FIRST tokens of ``embed_prefix`` (language follows)."""
        n_lang = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
        return prefix_len - n_lang

    @override
    def _prefix_embeddings(self, observation: _model.Observation):
        """Full prefix (image+language) → backbone ONCE → image-token hidden states.

        Returns (z, img_mask) with z = sg(image hidden states) — same shape as the
        parent's image-only z, but LANGUAGE-conditioned (image tokens attended to
        the language tokens in the bidirectional prefix).  Used by ``compute_loss``
        and ``extract_rl_token``, so the token is trained and extracted on the same
        with-language embeddings.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        n_img = self._num_image_tokens(observation, prefix_tokens.shape[1])
        z = jax.lax.stop_gradient(outs[0][:, :n_img].astype(jnp.float32))  # [b, n_img, W]
        return z, prefix_mask[:, :n_img]

    def extract_token_and_base_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_samples: int,
        num_steps: int = 10,
    ) -> tuple[at.Float[at.Array, "b t"], at.Float[at.Array, "b n ah ad"]]:
        """SINGLE backbone forward → (z_rl, base-action chunks).

        The joint win: the full π_vla prefix (image+language+state) is run once and
        reused for BOTH the RL token (image hidden states) and base-action sampling
        (KV cache) — no separate language-free pass.  Use this in annotation /
        inference instead of ``extract_rl_token`` + ``sample_base_actions``.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        b = observation.state.shape[0]
        dt = -1.0 / num_steps

        # One full prefix forward → hidden states (for the token) + KV cache (for sampling).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outs, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)

        # (a) RL token from the image-token hidden states (language-conditioned).
        n_img = self._num_image_tokens(observation, prefix_tokens.shape[1])
        z = jax.lax.stop_gradient(outs[0][:, :n_img].astype(jnp.float32))
        z_rl = self._encode_rl_token(z, prefix_mask[:, :n_img], observation.state)

        # (b) base-action chunks from the SAME KV cache (action-expert denoising, vmapped).
        def denoise(noise):
            def step(carry):
                x_t, time = carry
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, b)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
                pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn, positions=pos,
                    kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            def cond(carry):
                return carry[1] >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        noises = jax.random.normal(rng, (num_samples, b, self.action_horizon, self.action_dim))
        chunks = jax.vmap(denoise)(noises)                 # [n, b, H, D]
        base = jnp.transpose(chunks, (1, 0, 2, 3))          # [b, n, H, D]
        return z_rl, base
