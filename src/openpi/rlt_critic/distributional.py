"""HL-Gauss distributional value utilities for the AQC critic.



A distributional critic predicts a categorical distribution over ``num_bins`` fixed atoms
on a value support ``[v_min, v_max]`` instead of a single scalar. A *scalar* regression
target ``y`` is turned into a soft histogram by integrating a truncated Gaussian centred at
``y`` over each bin (``transform_to_probs``), and the critic is trained with cross-entropy.
The scalar value is read back as the support-weighted mean of the predicted probabilities
(``transform_from_probs``). This "classification instead of regression" objective is more
stable and scales better than MSE for value learning.

Usage::

    to_probs, from_probs = hl_gauss_transform(v_min, v_max, num_bins, sigma)
    target_dist = to_probs(scalar_target)               # (..., num_bins)
    loss = cross_entropy(logits, target_dist)           # train
    value = from_probs(jax.nn.softmax(logits, -1))      # read scalar value
"""

from typing import Optional

import jax
import jax.numpy as jnp


def _log1mexp(x: jax.Array) -> jax.Array:
    """Numerically stable ``log(1 - exp(-|x|))``."""
    x = jnp.abs(x)
    return jnp.where(
        x < jnp.log(2),
        jnp.log(-jnp.expm1(-x)),
        jnp.log1p(-jnp.exp(-x)),
    )


def _log_sub_exp(x: jax.Array, y: jax.Array) -> jax.Array:
    """Numerically stable ``log(exp(max(x,y)) - exp(min(x,y)))``."""
    larger = jnp.maximum(x, y)
    smaller = jnp.minimum(x, y)
    return larger + _log1mexp(jnp.maximum(larger - smaller, 0))


def _normal_cdf_log_difference(x: jax.Array, y: jax.Array) -> jax.Array:
    """``log(ndtr(x) - ndtr(y))`` assuming ``x >= y`` (stable for large |x|,|y|)."""
    is_y_positive = y >= 0
    x_hat = jnp.where(is_y_positive, -y, x)
    y_hat = jnp.where(is_y_positive, -x, y)
    return _log_sub_exp(
        jax.scipy.special.log_ndtr(x_hat),
        jax.scipy.special.log_ndtr(y_hat),
    )


def hl_gauss_transform(min_value: float, max_value: float, num_bins: int,
                       sigma: Optional[float] = None):
    """Build the (scalar -> probs) and (probs -> scalar) HL-Gauss transforms.

    Args:
        min_value, max_value: value support ``[v_min, v_max]``.
        num_bins: number of atoms (bins). The support has ``num_bins + 1`` edges.
        sigma: Gaussian smoothing std. If ``None``, uses the paper default
            ``0.75 * (v_max - v_min) / num_bins``.

    Returns:
        ``(transform_to_probs, transform_from_probs)``.
    """
    if sigma is None:
        sigma = 0.75 * (max_value - min_value) / num_bins
    support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)

    def transform_to_probs(target: jax.Array) -> jax.Array:
        """Scalar target -> soft categorical over ``num_bins`` (last axis)."""
        bin_log_probs = _normal_cdf_log_difference(
            (support[1:] - target) / (jnp.sqrt(2) * sigma),
            (support[:-1] - target) / (jnp.sqrt(2) * sigma),
        )
        log_z = _normal_cdf_log_difference(
            (support[-1] - target) / (jnp.sqrt(2) * sigma),
            (support[0] - target) / (jnp.sqrt(2) * sigma),
        )
        return jnp.exp(bin_log_probs - log_z)

    def transform_from_probs(probs: jax.Array) -> jax.Array:
        """Categorical probs -> scalar expected value (support-weighted mean)."""
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers, axis=-1)

    return transform_to_probs, transform_from_probs


def compute_return_to_go(rewards, terminals, discount):
    """Discounted return-to-go for every state, respecting episode boundaries.

    For state ``t``: ``RTG_t = sum_{k>=0} discount^k r_{t+k}`` up to the episode terminal.
    Computed with a single backward pass (numpy), resetting at terminal flags.

    Args:
        rewards: ``(N,)`` per-step rewards (use the *processed* rewards, e.g. after sparsify).
        terminals: ``(N,)`` episode-end flags (1 at the last step of an episode).
        discount: scalar discount.

    Returns:
        ``(N,)`` float64 return-to-go.
    """
    import numpy as np
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    terminals = np.asarray(terminals).reshape(-1)
    rtg = np.zeros_like(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if terminals[i] > 0:
            running = 0.0                      # next state is a new episode: no future for i
        running = rewards[i] + discount * running
        rtg[i] = running
    return rtg


def estimate_value_support(rewards, terminals, discount, p_low=1.0, p_high=99.0, margin=0.05):
    """Data-centric value support (DEAS '`d`' mode): percentiles of the return-to-go.

    Mirrors DEAS (``datasets.py:stats`` -> ``v_min=p1-delta, v_max=p99+delta``): take the
    ``p_low``/``p_high`` percentiles of the empirical discounted returns and pad by
    ``margin`` of the inter-percentile range.

    Returns:
        ``(v_min, v_max)`` floats.
    """
    import numpy as np
    rtg = compute_return_to_go(rewards, terminals, discount)
    lo = float(np.percentile(rtg, p_low))
    hi = float(np.percentile(rtg, p_high))
    delta = margin * (hi - lo + 1e-8)
    return lo - delta, hi + delta


def compute_reward_scale(rewards, terminals, discount, p_low=1.0, margin=0.05):
    """Scale factor mapping the dataset's discounted return-to-go into ``[-1, 0]``.

    For negative dense-cost rewards (return-to-go in ``[G_min, ~0]``), returns a single
    scalar ``s`` such that ``s * reward`` gives returns whose span lands in ``[-1, 0]``:
    ``s = 1 / (|G_low| * (1 + margin))`` where ``G_low`` is the ``p_low`` percentile of the
    return-to-go (robust to a few outlier episodes; the fixed ``[-1, 0]`` support's edge bin
    absorbs any tail beyond ``-1``). Use with a fixed support ``v_min=-1, v_max=0``.

    Returns:
        ``(scale, g_low, g_high)`` — the scale and the (unscaled) return-to-go bounds used.
    """
    import numpy as np
    rtg = compute_return_to_go(rewards, terminals, discount)
    g_low = float(np.percentile(rtg, p_low))
    g_high = float(np.percentile(rtg, 100.0))
    span = max(abs(g_low), abs(g_high), 1e-8)
    scale = 1.0 / (span * (1.0 + margin))
    return scale, g_low, g_high


def universal_value_support(r_min, r_max, discount):
    """Universal value support (DEAS '`u`' mode), single-discount form.

    Theoretical return bound for a per-step reward in ``[r_min, r_max]``:
    ``[r_min/(1-gamma), r_max/(1-gamma)]`` (the geometric-sum extremes).
    """
    denom = max(1.0 - discount, 1e-6)
    return r_min / denom, r_max / denom


def categorical_cross_entropy(logits: jax.Array, target_probs: jax.Array) -> jax.Array:
    """Cross-entropy ``-Σ target · log_softmax(logits)`` over the last (atom) axis.

    Args:
        logits: ``(..., num_bins)`` predicted logits.
        target_probs: ``(..., num_bins)`` target distribution (e.g. from
            ``transform_to_probs``).

    Returns:
        ``(...)`` per-element cross-entropy (atom axis reduced).
    """
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits, axis=-1), axis=-1)
