"""Adaptive Q-Chunking (AQC / ACSAC) agent.

Faithful implementation of "ACSAC: Adaptive Chunk Size Actor-Critic with Causal
Transformer Q-Network" (Chen et al., 2026), integrated into the QC codebase.

Compared with QC (`acfql.py`, `actor_type="best-of-n"`), AQC:
  * replaces the scalar MLP critic Q(s, a_{1:H}) with a causal Transformer critic that
    returns every prefix value Q(s, a_{1:h}) for h = 1..H in one pass (`utils.transformer`);
  * trains the critic with a per-horizon multi-step TD loss (Eq. 6), averaging the H
    squared errors at the gradient level, with an expected-prefix-max bootstrap target
    (Eq. 5/19) that maximises over N candidate chunks x H prefixes (min over the ensemble);
  * extracts the policy by a joint arg-max over (candidate index n, prefix length h)
    (Eq. 8), so the executed chunk size h* is state-dependent (adaptive replanning).

The flow-BC policy, the rejection-sampling idea, the dataset chunking, and all of the
training/eval scaffolding are reused unchanged from QC. See ../implementation_plan.html
and ../qc/AQC.md for the full design rationale and the list of documented design choices.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField
from utils.transformer import PrefixValue
from utils.distributional import hl_gauss_transform, categorical_cross_entropy


class AQCAgent(flax.struct.PyTreeNode):
    """Adaptive Q-Chunking agent (ACSAC).

    Supports two critic parameterizations via ``config['critic_type']``:
      * ``'regression'`` — scalar prefix Q-values trained with MSE (the original ACSAC).
      * ``'distributional'`` — HL-Gauss categorical prefix critic (num_atoms logits per
        prefix) trained with cross-entropy; the scalar value is the support-weighted mean.
    The two HL-Gauss transforms are carried as static (non-pytree) fields, built in
    :meth:`create` from ``v_min/v_max/num_atoms`` (only used when distributional).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    transform_to_probs: Any = nonpytree_field(default=None)
    transform_from_probs: Any = nonpytree_field(default=None)

    # ------------------------------------------------------------------ helpers
    def _aggregate(self, qs):
        """Aggregate over the critic ensemble axis (axis 0)."""
        if self.config['q_agg'] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def _prefix_values(self, observations, actions, module='critic', params=None):
        """Expected scalar prefix Q-values ``(K, ..., H)`` for either critic type.

        For the distributional critic the raw output is ``(K, ..., H, num_atoms)`` logits;
        we return the support-weighted mean of the softmax (the expected value). For the
        regression critic the raw output already is the scalar ``(K, ..., H)``.
        """
        out = self.network.select(module)(observations, actions, params=params)
        if self.config['critic_type'] == 'distributional':
            probs = jax.nn.softmax(out, axis=-1)
            return self.transform_from_probs(probs)
        return out

    def _scalar_to_probs(self, scalars):
        """Map a scalar target ``(...)`` to an HL-Gauss soft histogram ``(..., num_atoms)``."""
        flat = scalars.reshape(-1)
        probs = jax.vmap(self.transform_to_probs)(flat)
        return probs.reshape(scalars.shape + (self.config['num_atoms'],))

    def compute_flow_actions(self, observations, noises):
        """Generate length-H action chunks from the flow-BC policy by Euler integration.

        Args:
            observations: ``(..., obs_dim)``.
            noises: ``(..., H * action_dim)`` standard-normal noise.

        Returns:
            ``(..., H * action_dim)`` action chunks, clipped to ``[-1, 1]``.
        """
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)

    def _expected_prefix_max(self, states, rng):
        """Expected-prefix-max bootstrap value V*(s) (Eq. 19).

        For each state, sample ``N`` length-H chunks from the flow-BC policy, score all
        ``N x H`` prefixes with the (target) critic, aggregate over the ensemble, and take
        the max over (candidate, prefix length).

        Args:
            states: ``(M, obs_dim)`` bootstrap states.
            rng: PRNG key for proposal sampling.

        Returns:
            ``(M,)`` bootstrap values.
        """
        n = self.config['num_action_samples']
        h = self.config['horizon_length']
        full_action_dim = self.config['action_dim'] * h
        module = 'target_critic' if self.config['use_target_critic'] else 'critic'

        states_rep = jnp.repeat(states, n, axis=0)                 # (M*N, obs_dim)
        noises = jax.random.normal(rng, (states_rep.shape[0], full_action_dim))
        chunks = self.compute_flow_actions(states_rep, noises)     # (M*N, H*d)

        qs = self._prefix_values(states_rep, chunks, module)       # (K, M*N, H) expected vals
        q = self._aggregate(qs)                                    # (M*N, H)
        # max over the N*H prefixes belonging to each state.
        v = q.reshape(states.shape[0], n * h).max(axis=-1)         # (M,)
        return v

    # ------------------------------------------------------------------ losses
    def critic_loss(self, batch, grad_params, rng):
        """Per-horizon multi-step TD loss (Eq. 6)."""
        h = self.config['horizon_length']
        observations = batch['observations']                       # (B, obs_dim) = s_t
        batch_size = observations.shape[0]
        chunk = jnp.reshape(batch['actions'], (batch_size, -1))    # (B, H*d) = a_{t:t+H}

        # Prediction: all H prefix-conditioned outputs in one pass.
        #   regression:     (K, B, H) scalar Q-values
        #   distributional: (K, B, H, num_atoms) categorical logits
        pred = self.network.select('critic')(observations, chunk, params=grad_params)

        # Bootstrap value V*(s_{t+h}) at every next state h = 1..H (always a scalar).
        next_observations = batch['next_observations']             # (B, H, obs_dim)
        flat_next = jnp.reshape(next_observations, (batch_size * h, -1))
        v_next = self._expected_prefix_max(flat_next, rng)         # (B*H,)
        v_next = v_next.reshape(batch_size, h)                     # (B, H)

        # Per-horizon scalar target: G_h = r^h_t + gamma^h * mask_h * V*(s_{t+h}).
        # batch['rewards'][:, h-1] already holds the h-step cumulative discounted reward.
        discounts = self.config['discount'] ** jnp.arange(1, h + 1)  # gamma^1..gamma^H
        target = batch['rewards'] + discounts[None, :] * batch['masks'] * v_next
        target = jax.lax.stop_gradient(target)                     # (B, H)
        valid = batch['valid']                                     # (B, H)

        if self.config['critic_type'] == 'distributional':
            # Cross-entropy of the per-prefix logits against the HL-Gauss soft target.
            target_probs = self._scalar_to_probs(target)           # (B, H, num_atoms)
            ce = categorical_cross_entropy(pred, target_probs[None])  # (K, B, H)
            critic_loss = (ce * valid[None]).mean()                # 1/H gradient averaging
            q_value = self.transform_from_probs(jax.nn.softmax(pred, axis=-1))  # (K,B,H)
        else:
            # Mean squared per-horizon TD error.
            td_error = (pred - target[None]) ** 2                  # (K, B, H)
            critic_loss = (td_error * valid[None]).mean()
            q_value = pred

        info = {
            'critic_loss': critic_loss,
            'q_mean': q_value.mean(),
            'q_max': q_value.max(),
            'q_min': q_value.min(),
            'v_next_mean': v_next.mean(),
            'target_mean': target.mean(),
            # Cross-horizon spread of the per-state prefix values (max_h Q - min_h Q).
            # If this collapses toward 0 the prefixes look identical and the adaptive h*
            # selection degenerates to h=1 -- the canary for adaptive-chunking health.
            'prefix_spread': (q_value.max(axis=-1) - q_value.min(axis=-1)).mean(),
        }
        if self.config['critic_type'] == 'distributional':
            # --- distributional health checks (verify the categorical critic is working) ---
            probs = jax.nn.softmax(pred, axis=-1)                  # (K, B, H, num_atoms)
            edges = jnp.linspace(self.config['v_min'], self.config['v_max'],
                                 self.config['num_atoms'] + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mean = (probs * centers).sum(-1)
            var = jnp.maximum((probs * centers ** 2).sum(-1) - mean ** 2, 0.0)
            info.update({
                # Belief sharpness: entropy and value-std should DROP as the critic gets
                # confident. Stuck-high = not learning; ~0 = over-collapsed/degenerate.
                'dist_entropy': -(probs * jnp.log(probs + 1e-8)).sum(-1).mean(),
                'dist_value_std': jnp.sqrt(var).mean(),
                # Support adequacy: both should stay near 0. High edge mass / OOB targets
                # mean the [v_min, v_max] support is too tight and values are being clipped.
                'dist_edge_mass': (probs[..., 0] + probs[..., -1]).mean(),
                'dist_target_oob_frac': ((target < self.config['v_min'])
                                         | (target > self.config['v_max'])).mean(),
            })
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Flow-matching behavior-cloning loss on full-length chunks (Eq. 7)."""
        h = self.config['horizon_length']
        observations = batch['observations']
        batch_size = observations.shape[0]
        chunk = jnp.reshape(batch['actions'], (batch_size, -1))    # (B, H*d)
        full_action_dim = chunk.shape[-1]

        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        x_1 = chunk
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(observations, x_t, t, params=grad_params)

        # Only behavior-clone valid chunk steps (mask out steps past an episode boundary).
        bc_flow_loss = jnp.mean(
            jnp.reshape((pred - vel) ** 2, (batch_size, h, self.config['action_dim']))
            * batch['valid'][..., None]
        )
        return bc_flow_loss, {
            'actor_loss': bc_flow_loss,
            'bc_flow_loss': bc_flow_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (critic + flow-BC actor)."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Polyak-average the target network (only used when use_target_critic=True)."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Single gradient update."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        if agent.config['use_target_critic']:
            agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Vectorized multi-update (UTD > 1)."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ------------------------------------------------------------ policy extraction
    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Adaptive policy extraction (Eq. 8).

        Samples ``N`` candidate chunks from the flow-BC policy, scores every prefix with
        the critic, and selects the joint arg-max ``(n*, h*)``. Returns the *full* best
        chunk together with the selected prefix length ``h*`` so that the (jit-compiled)
        output shape is static; the rollout loop executes only the first ``h*`` actions
        and replans at the chunk boundary.

        Args:
            observations: ``(obs_dim,)`` single state.
            rng: PRNG key.

        Returns:
            ``(best_chunk, h_star)`` where ``best_chunk`` is ``(H * action_dim,)`` and
            ``h_star`` is an int in ``[1, H]``.
        """
        n = self.config['num_action_samples']
        h = self.config['horizon_length']
        full_action_dim = self.config['action_dim'] * h

        obs_rep = jnp.broadcast_to(observations, (n,) + observations.shape)  # (N, obs_dim)
        noises = jax.random.normal(rng, (n, full_action_dim))
        chunks = self.compute_flow_actions(obs_rep, noises)                  # (N, H*d)

        qs = self._prefix_values(obs_rep, chunks)                           # (K, N, H) exp. vals
        q = self._aggregate(qs)                                             # (N, H)

        if self.config['adaptive_chunking']:
            flat_idx = jnp.argmax(q.reshape(-1))                            # over N*H
            n_star = flat_idx // h
            h_star = (flat_idx % h) + 1                                     # prefix length 1..H
        else:
            # Fixed-H control: pick the best full-length chunk (QT-QC-style ablation).
            n_star = jnp.argmax(q[:, -1])
            h_star = h

        best_chunk = chunks[n_star]                                         # (H*d,)
        return best_chunk, h_star

    @jax.jit
    def sample_actions_with_info(self, observations, rng=None):
        """Adaptive policy extraction that also returns diagnostics for visualization.

        Same selection rule as :meth:`sample_actions`, but additionally returns the
        per-prefix value curve and the chosen indices so callers can visualize *why* a
        given chunk size was selected.

        Returns:
            ``(best_chunk, h_star, info)`` where ``info`` is a dict of arrays:
              - ``q_by_h`` ``(H,)``: best value over candidates for each prefix length
                ``h=1..H`` (i.e. ``max_n Q(s, a^(n)_{1:h})``); the curve the arg-max uses.
              - ``q_all``  ``(N, H)``: every candidate's prefix values.
              - ``h_star`` scalar: selected prefix length (1..H).
              - ``n_star`` scalar: selected candidate index.
              - ``q_best`` scalar: value of the selected prefix.
        """
        n = self.config['num_action_samples']
        h = self.config['horizon_length']
        full_action_dim = self.config['action_dim'] * h

        obs_rep = jnp.broadcast_to(observations, (n,) + observations.shape)
        noises = jax.random.normal(rng, (n, full_action_dim))
        chunks = self.compute_flow_actions(obs_rep, noises)                  # (N, H*d)

        qs = self._prefix_values(obs_rep, chunks)                           # (K, N, H) exp. vals
        q = self._aggregate(qs)                                             # (N, H)
        q_by_h = q.max(axis=0)                                              # (H,)

        if self.config['adaptive_chunking']:
            flat_idx = jnp.argmax(q.reshape(-1))
            n_star = flat_idx // h
            h_star = (flat_idx % h) + 1
        else:
            n_star = jnp.argmax(q[:, -1])
            h_star = h

        best_chunk = chunks[n_star]
        info = dict(q_by_h=q_by_h, q_all=q, h_star=h_star, n_star=n_star,
                    q_best=q_by_h[h_star - 1])
        return best_chunk, h_star, info

    # ------------------------------------------------------------------- create
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new AQC agent.

        Args:
            seed: Random seed.
            ex_observations: Example observation ``(obs_dim,)``.
            ex_actions: Example single-step action ``(action_dim,)``.
            config: Configuration dict.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        horizon = config['horizon_length']
        assert horizon is not None and horizon >= 1, 'horizon_length (max chunk size H) must be set'

        full_actions = jnp.concatenate([ex_actions] * horizon, axis=-1)     # (H*d,)
        full_action_dim = full_actions.shape[-1]
        ex_times = ex_actions[..., :1]

        # Initialize with a leading batch dim (robust for the Transformer / attention).
        ex_obs_b = ex_observations[None]
        full_actions_b = full_actions[None]
        ex_times_b = ex_times[None]

        # Encoders (optional, e.g. for pixel observations).
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Number of output atoms: 1 for scalar regression, num_atoms for distributional.
        distributional = config['critic_type'] == 'distributional'
        num_atoms = config['num_atoms'] if distributional else 1

        # Causal Transformer prefix critic.
        critic_def = PrefixValue(
            action_dim=action_dim,
            horizon=horizon,
            num_ensembles=config['num_qs'],
            num_layers=config['transformer_num_layers'],
            num_heads=config['transformer_num_heads'],
            head_dim=config['transformer_head_dim'],
            mlp_dim=config['transformer_mlp_dim'],
            layer_norm=config['layer_norm'],
            num_atoms=num_atoms,
            per_position_head=config['per_position_head'],
            encoder=encoders.get('critic'),
        )

        # Flow-BC velocity field over full-length chunks (reused from QC).
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_obs_b, full_actions_b, ex_times_b)),
            critic=(critic_def, (ex_obs_b, full_actions_b)),
        )
        if config['use_target_critic']:
            network_info['target_critic'] = (copy.deepcopy(critic_def), (ex_obs_b, full_actions_b))
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_obs_b,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['weight_decay'] > 0.0:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config['weight_decay'])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        if config['use_target_critic']:
            params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        # Build the HL-Gauss transforms (static, used only by the distributional critic).
        transform_to_probs, transform_from_probs = None, None
        if distributional:
            assert config['v_min'] is not None and config['v_max'] is not None, (
                'distributional critic requires v_min and v_max (estimate from dataset returns)')
            sigma = config['hl_gauss_sigma'] * (config['v_max'] - config['v_min']) / config['num_atoms']
            transform_to_probs, transform_from_probs = hl_gauss_transform(
                min_value=config['v_min'], max_value=config['v_max'],
                num_bins=config['num_atoms'], sigma=sigma)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config),
                   transform_to_probs=transform_to_probs,
                   transform_from_probs=transform_from_probs)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='aqc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),   # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set automatically.
            lr=3e-4,            # Learning rate.
            batch_size=256,     # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Flow-BC policy hidden dims.
            layer_norm=True,    # Pre-LayerNorm in the Transformer critic (recommended).
            actor_layer_norm=False,  # LayerNorm for the flow-BC actor.
            discount=0.99,      # Discount factor.
            tau=0.005,          # Polyak rate (only used if use_target_critic=True).
            q_agg='min',        # Ensemble aggregation: ACSAC uses min.
            num_qs=2,           # Critic ensemble size K.
            flow_steps=10,      # Euler steps F for the flow-BC policy.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            horizon_length=ml_collections.config_dict.placeholder(int),  # Max chunk size H.
            action_chunking=True,  # AQC always chunks (kept for parity with QC configs).
            # --- Causal Transformer critic ---
            transformer_num_layers=2,
            transformer_num_heads=8,
            transformer_head_dim=16,   # n_embd = num_heads * head_dim = 128.
            transformer_mlp_dim=512,
            per_position_head=True,    # paper: one output head per position (Prop G.7).
            # --- Adaptive Q-Chunking ---
            num_action_samples=4,     # Rejection-sampling size N (bootstrap & extraction).
            adaptive_chunking=True,   # Joint (n, h) arg-max; False -> fixed-H control.
            use_target_critic=False,  # Paper: online critic stop-grad. True -> Polyak target.
            # --- Distributional critic (HL-Gauss); regression keeps the scalar critic ---
            critic_type='regression',  # 'regression' (scalar MSE) | 'distributional' (HL-Gauss CE).
            num_atoms=101,            # Number of categorical atoms (distributional only).
            v_min=ml_collections.config_dict.placeholder(float),  # Value support min (distributional).
            v_max=ml_collections.config_dict.placeholder(float),  # Value support max (distributional).
            hl_gauss_sigma=0.75,      # HL-Gauss smoothing (in units of bin width).
            # --- Misc (reused from QC) ---
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.0,
        )
    )
    return config
