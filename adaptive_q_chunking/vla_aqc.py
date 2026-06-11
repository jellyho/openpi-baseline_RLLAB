"""AQC critic learning for the frozen-VLA data (driven by VLAAQCConfig).

Implements the ACSAC per-prefix critic update on the LeRobot VLA dataset:

  * ``target_kind='td'`` (paper-faithful): per-prefix multi-step TD with the EMaQ-style
    joint-max bootstrap over the precomputed ``base_action`` candidates,
        V(s') = min_K max_{n in [N], h' in [H]} Q_phi-bar(s', base_action(s')^{(n)}_{1:h'}),
    using mc_return at the (-0.5) failure terminal instead of the candidate max.
  * ``target_kind='mc'`` (baseline): regress directly to the precomputed mc_return.

All hyperparameters come from ``vla_config.VLAAQCConfig`` (capacity, distributional support,
discount/candidates/prefix-grid, optimizer). This is a slim critic-only trainer (no flow
actor / env), matching the offline VLA setting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logsumexp

from utils.transformer import PrefixValue
from utils.distributional import hl_gauss_transform, categorical_cross_entropy
from vla_config import VLAAQCConfig


def _make_optimizer(opt_cfg):
    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, opt_cfg.lr, max(opt_cfg.warmup_steps, 1)),
         optax.constant_schedule(opt_cfg.lr)],
        [max(opt_cfg.warmup_steps, 1)],
    )
    chain = []
    if opt_cfg.max_grad_norm is not None:
        chain.append(optax.clip_by_global_norm(opt_cfg.max_grad_norm))
    chain.append(optax.adamw(sched, weight_decay=opt_cfg.weight_decay)
                 if opt_cfg.weight_decay > 0 else optax.adam(sched))
    return optax.chain(*chain)


class VLACriticTrainer:
    """Builds the PrefixValue critic from a VLAAQCConfig and runs the critic update."""

    def __init__(self, cfg: VLAAQCConfig, seed: int = 0):
        self.cfg = cfg
        a, d = cfg.arch, cfg.dist
        self.net = PrefixValue(
            action_dim=cfg.action_dim, horizon=cfg.horizon,
            num_ensembles=a.num_ensembles, num_layers=a.num_layers,
            num_heads=a.num_heads, head_dim=a.head_dim, mlp_dim=a.mlp_dim,
            layer_norm=a.layer_norm, num_atoms=d.num_atoms,
            per_position_head=a.per_position_head, state_encoder_dims=a.state_encoder_dims,
            macro_group_size=cfg.td.macro_group_size,
        )
        sigma = d.hl_gauss_sigma_frac * (d.v_max - d.v_min) / d.num_atoms
        self.to_probs, self.from_probs = hl_gauss_transform(d.v_min, d.v_max, d.num_atoms, sigma)
        key = jax.random.PRNGKey(seed)
        obs = jnp.zeros((1, cfg.latent_dim))
        act = jnp.zeros((1, cfg.horizon * cfg.action_dim))
        self.params = self.net.init(key, obs, act)
        self.opt = _make_optimizer(cfg.optim)
        self.opt_state = self.opt.init(self.params)

    # ---- value helpers -----------------------------------------------------------
    def _prefix_values(self, params, obs, act):
        logits = self.net.apply(params, obs, act)            # (K, ..., H, atoms)
        return self.from_probs(jax.nn.softmax(logits, -1))   # (K, ..., H)

    def _aggregate(self, qf):
        """Aggregate the (M, J) candidate x prefix Q's into V(s') per cfg.td.agg_mode.

        'max'       -> hard max (Best-of-N / EMaQ; optimistic).
        'softmax'   -> Boltzmann weighted mean: sum_j softmax(beta q)_j * q_j (conservative).
        'mellowmax' -> (1/beta) log( mean_j exp(beta q) ) (conservative; contraction-preserving).
        beta = cfg.td.agg_beta (inverse temperature); beta->inf recovers max, beta->0 = mean.
        """
        mode = self.cfg.td.agg_mode
        if mode == "max":
            return qf.max(axis=-1)
        beta = self.cfg.td.agg_beta
        if mode == "softmax":
            w = jax.nn.softmax(beta * qf, axis=-1)               # (M, J)
            return (w * qf).sum(axis=-1)
        if mode == "mellowmax":
            J = qf.shape[-1]
            return logsumexp(beta * qf, axis=-1) / beta - jnp.log(J) / beta
        raise ValueError(f"unknown agg_mode {mode!r}")

    def _expected_prefix_max(self, params, next_latents, candidates):
        """V(s') = agg_{n,h'} min_K Q(s', cand_n)  over precomputed candidates.

        One big forward over all M*N candidate sequences (fits on A100/L40S, especially
        with macro-action grouping where each sequence is only macro_H+1 tokens). Ensemble-min
        per (candidate, prefix), then aggregate over (candidate, prefix) via cfg.td.agg_mode
        (hard max by default; softmax/mellowmax for a conservative bootstrap).

        next_latents: (M, latent); candidates: (M, N, H*action_dim) -> (M,) values.
        """
        M, N, Hd = candidates.shape
        states = jnp.repeat(next_latents, N, axis=0)             # (M*N, latent)
        chunks = candidates.reshape(M * N, Hd).astype(jnp.float32)  # fp16 from loader -> f32 (cheap on GPU)
        qs = self._prefix_values(params, states, chunks)        # (K, M*N, macro_H)
        q = qs.min(axis=0)                                       # ensemble min -> (M*N, macro_H)
        return self._aggregate(q.reshape(M, N * q.shape[-1]))   # (M,) over (candidate, prefix)

    # ---- losses ------------------------------------------------------------------
    def critic_loss_mc(self, params, batch):
        """Regress every prefix to the precomputed mc_return (baseline; no bootstrap)."""
        logits = self.net.apply(params, batch["observations"], batch["action_chunks"])  # (K,B,H,atoms)
        tgt = self.to_probs(batch["mc_return"][:, None])         # (B,atoms)
        tgt = jnp.broadcast_to(tgt[None, :, None, :], logits.shape)
        ce = categorical_cross_entropy(logits, tgt)              # (K,B,H)
        loss = ce.mean()
        q = self.from_probs(jax.nn.softmax(logits, -1))
        return loss, {"critic_loss": loss, "q_mean": q.mean(),
                      "target_mean": batch["mc_return"].mean(),
                      "prefix_spread": (q.max(-1) - q.min(-1)).mean()}

    def critic_loss_td(self, params, batch, prefixes):
        cfg = self.cfg
        B, P = batch["cum_reward"].shape
        mg = cfg.td.macro_group_size
        logits = self.net.apply(params, batch["observations"], batch["action_chunks"])  # (K,B,macro_H,atoms)
        # prefixes are step-counts; convert to 0-indexed macro-prefix positions.
        macro_idx = prefixes // mg - 1                            # e.g. [10,20,30,40,50]//10-1 = [0,1,2,3,4]
        pred = logits[:, :, macro_idx, :]                         # (K,B,P,atoms)

        # Bootstrap: online critic with stop_gradient (paper default, no target network).
        nl = batch["next_latents"].reshape(B * P, cfg.latent_dim)
        cand = batch["next_candidates"].reshape(B * P, batch["next_candidates"].shape[2], -1)
        vmax = self._expected_prefix_max(jax.lax.stop_gradient(params), nl, cand).reshape(B, P)
        v_next = jnp.where(batch["term"] > 0, batch["next_mc_return"], vmax) \
            if cfg.td.terminal_uses_mc else vmax

        gamma_h = cfg.td.discount ** prefixes.astype(jnp.float32)  # (P,)
        td_target = batch["cum_reward"] + gamma_h[None, :] * batch["valid"] * v_next
        mc_col = batch["mc_return"][:, None]                       # (B,1) realized behavior return
        target = jnp.maximum(td_target, mc_col) if cfg.td.mc_floor else td_target
        target = jax.lax.stop_gradient(target)                   # (B,P)
        # Cal-QL floor diagnostics: how often mc_return wins the max (floor binds) and by how much.
        floor_gap = jnp.maximum(mc_col - td_target, 0.0)         # (B,P) lift applied where >0
        floor_active = (floor_gap > 0).astype(jnp.float32)

        tgt_probs = self.to_probs(target[..., None])             # (B,P,atoms)
        ce = categorical_cross_entropy(pred, tgt_probs[None])    # (K,B,P)
        valid = batch["valid"][None]
        denom = jnp.maximum(valid.sum() * cfg.arch.num_ensembles, 1.0)
        loss = (ce * valid).sum() / denom

        q = self.from_probs(jax.nn.softmax(pred, -1))            # (K,B,P)
        probs = jax.nn.softmax(pred, -1)
        edge = (probs[..., 0] + probs[..., -1])
        vsum = jnp.maximum(batch["valid"].sum(), 1.0)
        info = {
            "critic_loss": loss,
            # divide by valid*num_ensembles (same denom as the loss): q is (K,B,P), so dividing
            # by valid.sum() alone summed over the K ensembles -> reported 2x the true mean.
            "q_mean": (q * valid).sum() / denom,
            "target_mean": (target * batch["valid"]).sum() / vsum,
            "v_next_mean": v_next.mean(),
            "term_frac": batch["term"].mean(),
            "valid_frac": batch["valid"].mean(),
            # Cal-QL MC floor: fraction of valid targets where mc_return raises the TD target,
            # and the mean lift it applies (0 if mc_floor off -> just shows how often mc > td).
            "mc_floor_frac": (floor_active * batch["valid"]).sum() / vsum,
            "mc_floor_gap": (floor_gap * batch["valid"]).sum() / vsum,
            "mc_return_mean": (mc_col * batch["valid"]).sum() / vsum,
            "prefix_spread": (q.max(0).max(-1) - q.max(0).min(-1)).mean(),
            "dist_edge_mass": edge.mean(),
            "dist_target_oob": ((target < cfg.dist.v_min) | (target > cfg.dist.v_max)).mean(),
        }
        return loss, info

    # ---- train steps -------------------------------------------------------------
    def make_train_step(self):
        cfg = self.cfg
        if cfg.td.target_kind == "mc":
            @jax.jit
            def step(params, opt_state, batch, prefixes):
                (loss, info), grads = jax.value_and_grad(self.critic_loss_mc, has_aux=True)(params, batch)
                updates, opt_state = self.opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, info
            return step

        @jax.jit
        def step(params, opt_state, batch, prefixes):
            (loss, info), grads = jax.value_and_grad(self.critic_loss_td, has_aux=True)(
                params, batch, prefixes)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, info
        return step

    def num_params(self):
        return int(sum(x.size for x in jax.tree_util.tree_leaves(self.params)))


def to_jax_batch(b: dict):
    """numpy loader batch -> (jax arrays dict, int32 prefixes or None for MC)."""
    prefixes = jnp.asarray(b["prefixes"], dtype=jnp.int32) if "prefixes" in b else None
    keys = [k for k in b if k != "prefixes"]
    return {k: jnp.asarray(b[k]) for k in keys}, prefixes

