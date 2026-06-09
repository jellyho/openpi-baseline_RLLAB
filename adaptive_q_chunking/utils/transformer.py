"""Causal Transformer prefix-conditioned critic for Adaptive Q-Chunking (ACSAC).

This module implements the cross-horizon, prefix-conditioned Q-network described in
Section 4.1 / Appendix B of ACSAC ("Adaptive Chunk Size Actor-Critic with Causal
Transformer Q-Network").

The critic ingests the token sequence ``(s_t, a_t, a_{t+1}, ..., a_{t+H-1})`` (one state
token followed by ``H`` action tokens) and, in a *single* forward pass, returns the
prefix-conditioned Q-values

    [ Q(s_t, a_{t:t+1}), Q(s_t, a_{t:t+2}), ..., Q(s_t, a_{t:t+H}) ].

A causal attention mask guarantees that the ``h``-th output depends only on
``(s_t, a_{t:t+h})`` and not on the suffix ``a_{t+h:t+H}`` (prefix consistency,
Proposition G.7). Because all ``H`` heads share a single backbone and are trained
against per-horizon return targets, the values live on a common discounted-return scale
and are therefore comparable across prefix lengths (Theorem G.9). This is what makes the
joint arg-max over ``(candidate index n, prefix length h)`` meaningful.

Design choices (documented in implementation_plan.html):
  * Pre-LayerNorm before every attention/MLP sublayer + a final LayerNorm. The bounded NTK
    from LayerNorm (SEEM, ref. [43]) is what lets ACSAC bootstrap from the online critic
    with a stopped gradient instead of a Polyak target network.
  * One output head per prefix position (``per_position_head=True``, the default), following
    the paper's "one output head per position" (Prop. G.7): a distinct
    ``(n_embd -> num_atoms)`` projection at each of the ``H`` action-token positions. A shared
    (tied-weights) head is available via ``per_position_head=False``.
  * ``num_atoms`` selects the critic head: ``1`` -> a scalar (regression) Q-value;
    ``>1`` -> per-prefix categorical logits for the distributional (HL-Gauss) critic.
  * Learned absolute positional embeddings over ``L = H + 1`` tokens (H is small).
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import default_init, ensemblize


class CausalPrefixCritic(nn.Module):
    """A single causal Transformer critic emitting all ``H`` prefix-conditioned Q-values.

    Attributes:
        action_dim: Per-step action dimension ``d``.
        horizon: Maximum chunk size ``H`` (number of action tokens).
        num_layers: Number of Transformer blocks (``n_layer``).
        num_heads: Number of attention heads (``n_head``).
        head_dim: Per-head dimension (``d_head``); the embedding size is
            ``n_embd = num_heads * head_dim``.
        mlp_dim: Hidden width of the position-wise feed-forward sublayer.
        layer_norm: Whether to apply Pre-LayerNorm (recommended; required for the
            online-critic-as-target stability argument).
    """

    action_dim: int
    horizon: int
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 16
    mlp_dim: int = 512
    layer_norm: bool = True
    num_atoms: int = 1   # 1 -> scalar (regression); >1 -> distributional logits (HL-Gauss)
    per_position_head: bool = True   # paper: "one output head per position" (Prop G.7)
    state_encoder_dims: Sequence[int] = ()   # MLP hidden dims applied to the obs before the
                                             # state token; helps digest a high-dim latent (2048)

    @nn.compact
    def __call__(self, observations, actions):
        """Return prefix-conditioned Q-values.

        Args:
            observations: ``(..., obs_dim)`` (already encoded if an encoder is used).
            actions: ``(..., H * action_dim)`` flattened action chunk. The chunk is
                reshaped internally to ``(..., H, action_dim)``.

        Returns:
            If ``num_atoms == 1``: ``(..., H)`` scalar prefix Q-values, where the ``h``-th
            entry (0-indexed) is ``Q(s, a_{t:t+h+1})``.
            If ``num_atoms > 1``: ``(..., H, num_atoms)`` per-prefix categorical logits for
            the distributional (HL-Gauss) critic.
        """
        n_embd = self.num_heads * self.head_dim
        seq_len = self.horizon + 1  # 1 state token + H action tokens.

        # --- Tokenize ------------------------------------------------------------------
        chunk = actions.reshape(actions.shape[:-1] + (self.horizon, self.action_dim))
        # Optional MLP encoder on the observation (e.g. to digest a 2048-d VLA latent) before
        # projecting to the single state token.
        s = observations
        for hdim in self.state_encoder_dims:
            s = nn.gelu(nn.Dense(hdim, kernel_init=default_init())(s))
        state_token = nn.Dense(n_embd, kernel_init=default_init())(s)
        state_token = state_token[..., None, :]                       # (..., 1, n_embd)
        action_tokens = nn.Dense(n_embd, kernel_init=default_init())(chunk)  # (..., H, n_embd)
        x = jnp.concatenate([state_token, action_tokens], axis=-2)    # (..., L, n_embd)

        # Learned absolute positional embeddings.
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02),
                             (seq_len, n_embd))
        x = x + pos_emb

        # Causal mask: token i attends only to tokens j <= i. Shape (..., 1, L, L).
        mask = nn.make_causal_mask(jnp.ones(x.shape[:-1], dtype=bool))

        # --- Transformer blocks (Pre-LayerNorm) ----------------------------------------
        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x) if self.layer_norm else x
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=n_embd,
                kernel_init=default_init(),
            )(h, h, mask=mask)
            x = x + attn

            h = nn.LayerNorm()(x) if self.layer_norm else x
            h = nn.Dense(self.mlp_dim, kernel_init=default_init())(h)
            h = nn.gelu(h)
            h = nn.Dense(n_embd, kernel_init=default_init())(h)
            x = x + h

        if self.layer_norm:
            x = nn.LayerNorm()(x)

        # --- Read prefix Q-values from the action-token hidden states -------------------
        # Action token at position p (1..H) has attended to {s_t, a_t, ..., a_{t+p-1}},
        # so reading a head there yields Q(s_t, a_{t:t+p}).
        action_hidden = x[..., 1:, :]                                 # (..., H, n_embd)
        if self.per_position_head:
            # One output head per prefix position h (paper: "one output head per position").
            # A distinct (n_embd -> num_atoms) projection for each of the H positions, via einsum.
            head_w = self.param('head_kernel', default_init(),
                                (self.horizon, n_embd, self.num_atoms))
            head_b = self.param('head_bias', nn.initializers.zeros,
                                (self.horizon, self.num_atoms))
            q = jnp.einsum('...hd,hda->...ha', action_hidden, head_w) + head_b  # (..., H, atoms)
        else:
            # Shared head (tied weights across positions).
            q = nn.Dense(self.num_atoms, kernel_init=default_init())(action_hidden)
        if self.num_atoms == 1:
            return q.squeeze(-1)                                      # (..., H) scalar critic
        return q                                                      # (..., H, num_atoms) logits


class PrefixValue(nn.Module):
    """Ensemble wrapper around :class:`CausalPrefixCritic`.

    Mirrors ``utils.networks.Value``: an optional shared observation encoder is applied
    once, then the Transformer critic is ensemblized over ``num_ensembles`` members. The
    output gains a leading ensemble axis.

    Attributes:
        action_dim: Per-step action dimension.
        horizon: Maximum chunk size ``H``.
        num_ensembles: Number of critics ``K`` (ACSAC uses 2 and takes the min).
        num_layers, num_heads, head_dim, mlp_dim, layer_norm: Transformer config.
        encoder: Optional observation encoder (e.g. for pixel inputs).
    """

    action_dim: int
    horizon: int
    num_ensembles: int = 2
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 16
    mlp_dim: int = 512
    layer_norm: bool = True
    num_atoms: int = 1   # 1 -> scalar critic; >1 -> distributional (HL-Gauss) logits
    per_position_head: bool = True   # paper: one output head per position
    state_encoder_dims: Sequence[int] = ()   # obs-encoder MLP hidden dims
    encoder: nn.Module = None

    def setup(self):
        critic_cls = CausalPrefixCritic
        if self.num_ensembles > 1:
            critic_cls = ensemblize(critic_cls, self.num_ensembles)
        self.critic = critic_cls(
            action_dim=self.action_dim,
            horizon=self.horizon,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_dim=self.mlp_dim,
            layer_norm=self.layer_norm,
            num_atoms=self.num_atoms,
            per_position_head=self.per_position_head,
            state_encoder_dims=self.state_encoder_dims,
        )

    def __call__(self, observations, actions):
        """Return prefix-conditioned outputs for the ensemble.

        ``(K, ..., H)`` scalar Q-values if ``num_atoms == 1``; otherwise
        ``(K, ..., H, num_atoms)`` categorical logits.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        return self.critic(observations, actions)
