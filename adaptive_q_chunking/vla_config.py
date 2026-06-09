"""Config + run management for VLA AQC critic learning.

Mirrors the openpi ``src/openpi/training/config.py`` pattern: a frozen, fully-typed
``VLAAQCConfig`` (with nested groups), a named ``CONFIGS`` registry of documented presets,
and a ``checkpoint_dir = base/name/exp_name`` convention. Every hyperparameter we decided to
think carefully about lives here in one place:

  * batch size & training length          -> OptimConfig
  * distributional-RL knobs (atoms, support, HL-Gauss sigma) -> DistConfig
  * model capacity (n_embd, depth, heads, state encoder)     -> ArchConfig
  * MC-return / discount / TD bootstrap / N candidates       -> TDConfig
  * reward normalization & value-support choice              -> RewardConfig

Each run writes its full resolved config to ``<run_dir>/config.json`` and gets a descriptive
auto ``run_name`` so a run is self-documenting. Checkpoints/logs default to lustre (home is
small); code stays in the repo.

Usage::

    from vla_config import get_config
    cfg = get_config("vla_aqc_td_a51")                 # a named preset
    cfg = dataclasses.replace(cfg, seed=1)             # tweak
    print(cfg.run_name, cfg.run_dir)
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Nested config groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchConfig:
    """Causal-Transformer critic capacity (scaled up for the 2048-d VLA latent)."""
    num_ensembles: int = 2          # K (min-aggregated target)
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 48              # n_embd = num_heads * head_dim = 384
    mlp_dim: int = 1024
    per_position_head: bool = True  # paper: one output head per prefix position
    layer_norm: bool = True
    # State-token encoder: MLP hidden dims applied to rl_token before the n_embd projection.
    # () = single linear Dense (ACSAC-faithful default); (512,) = MLP to better digest the
    # 2048-d latent (opt-in; see the vla_aqc_td_a51_stateenc preset).
    state_encoder_dims: tuple[int, ...] = ()

    @property
    def n_embd(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class DistConfig:
    """Distributional (HL-Gauss) value head."""
    num_atoms: int = 51             # 51 -> bin width 0.01 over [-0.5, 0]
    hl_gauss_sigma_frac: float = 0.75   # sigma = frac * (v_max - v_min) / num_atoms
    # How the value support [v_min, v_max] is set:
    #   'fixed'       -> use v_min/v_max below (default [-0.5, 0], the mc_return range)
    #   'reward_norm' -> rewards scaled into [-1, 0]; support fixed to [-1, 0]
    #   'data'        -> p1/p99 of return-to-go + margin (DEAS data-centric)
    support_mode: Literal["fixed", "reward_norm", "data"] = "fixed"
    v_min: float = -0.5
    v_max: float = 0.0


@dataclass(frozen=True)
class TDConfig:
    """MC-return / multi-step TD bootstrap (the ACSAC expected-prefix-max)."""
    discount: float = 0.995         # verified mc_t = r_t + 0.995 * mc_{t+1}
    # Target kind:
    #   'td' -> per-prefix multi-step TD with the N-candidate joint-max bootstrap (paper)
    #   'mc' -> regress directly to precomputed mc_return (RECAP-style baseline; no bootstrap)
    target_kind: Literal["td", "mc"] = "td"
    num_candidates: int = 32        # N (== base_action's 32); the bootstrap max is over N x H
    prefixes: tuple[int, ...] = (1, 10, 25, 50)   # prefix-subsample grid (H=50 cost knob)
    use_target_critic: bool = False # paper default: online critic, stop-grad
    tau: float = 0.005              # Polyak rate (only if use_target_critic)
    terminal_uses_mc: bool = True   # at the -0.5 failure terminal, bootstrap = mc_return
    # Bootstrap memory/speed knob: candidates processed per forward in the joint-max.
    # 1 = minimal memory (slow, scan over all N); N = one big forward (fastest, most memory).
    # On 24GB use ~8; on 96GB (pro6000) set = num_candidates for max throughput.
    bootstrap_candidate_tile: int = 8


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping for the value scale (interacts with DistConfig.support_mode)."""
    reward_normalize: bool = False  # scale rewards so return-to-go lands in [-1, 0]
    support_margin: float = 0.05    # margin for 'data' support / normalization


@dataclass(frozen=True)
class OptimConfig:
    batch_size: int = 256
    lr: float = 3e-4
    num_train_steps: int = 500_000  # ceiling; use eval-based early stop (~200-400k typical)
    warmup_steps: int = 2_000
    weight_decay: float = 0.0
    max_grad_norm: Optional[float] = None   # None = no clipping (QC has none)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VLAAQCConfig:
    # --- identity ---
    name: str                       # registry key; part of the checkpoint path
    exp_name: str = ""              # set per-run (defaults to run_name if empty)
    notes: str = ""                 # free-text description of what this run is testing

    # --- problem shape (from the dataset; see vla_data) ---
    action_dim: int = 14
    horizon: int = 50               # H == base_action chunk length
    latent_dim: int = 2048

    # --- data ---
    data_root: str = ("/lustre/gwanwoo13/rss_post_training/"
                      "Challenge-phase1-dataset/insert-mouse-battery_annotated")
    commander_filter: Optional[tuple[str, ...]] = None  # e.g. ("inference",) or ("teleop",)
    shuffle_buffer_groups: int = 8
    num_workers: int = 8

    # --- grouped hyperparameters ---
    arch: ArchConfig = field(default_factory=ArchConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    td: TDConfig = field(default_factory=TDConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # --- run management ---
    seed: int = 0
    log_interval: int = 500
    eval_interval: int = 25_000
    save_interval: int = 25_000
    keep_period: Optional[int] = 100_000   # checkpoints at step % keep_period == 0 are kept
    checkpoint_base_dir: str = "/lustre/gwanwoo13/rss_post_training/phase1/critic_learning"
    wandb_enabled: bool = True
    wandb_project: str = "AQC-VLA"
    wandb_entity: str = "gwanwoo-yonsei-university"

    # ---- derived identity --------------------------------------------------
    @property
    def run_name(self) -> str:
        """Descriptive, self-documenting run name built from the load-bearing settings."""
        # Compact spec only (the family name lives in the parent dir via `name`); avoids
        # doubling the name. e.g. "a51_sup-fixed_emb384x4L_N32_P4_b256_g0.995_s0".
        a, d, t = self.arch, self.dist, self.td
        parts = [
            f"a{d.num_atoms}",                          # atoms
            f"sup-{d.support_mode}",                    # value support choice
            f"emb{a.n_embd}x{a.num_layers}L",           # capacity
            f"N{t.num_candidates}",                     # candidates
            f"P{len(t.prefixes)}",                      # prefix grid size
            f"b{self.optim.batch_size}",
            f"g{t.discount}",
        ]
        if self.reward.reward_normalize:
            parts.append("rnorm")
        if self.td.use_target_critic:
            parts.append("tgt")
        if self.commander_filter:
            parts.append("+".join(self.commander_filter))
        parts.append(f"s{self.seed}")
        return "_".join(parts)

    @property
    def exp(self) -> str:
        return self.exp_name or self.run_name

    @property
    def run_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self.run_dir / "checkpoints"

    def __post_init__(self):
        assert self.dist.v_min < self.dist.v_max, "value support must be non-empty"
        assert min(self.td.prefixes) >= 1 and max(self.td.prefixes) <= self.horizon
        assert self.td.num_candidates >= 1

    # ---- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: Optional[pathlib.Path] = None) -> pathlib.Path:
        """Write the fully-resolved config to ``<run_dir>/config.json`` (run is self-documenting)."""
        path = path or (self.run_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        d["_run_name"] = self.run_name
        path.write_text(json.dumps(d, indent=2, default=str))
        return path


# ---------------------------------------------------------------------------
# Named registry of presets (the "what is this run" catalogue)
# ---------------------------------------------------------------------------
_CONFIGS = [
    # Paper-faithful default: per-prefix TD, 32-candidate joint-max bootstrap, HL-Gauss a51
    # over the fixed [-0.5, 0] support, n_embd=384 / 4 layers (~15M params).
    VLAAQCConfig(
        name="vla_aqc_td_a51",
        notes="Default ACSAC-TD critic on mouse-battery: 32-candidate joint-max bootstrap, "
              "HL-Gauss 51 atoms over fixed [-0.5,0], n_embd=384/4L, B=256.",
    ),
    # MC baseline: regress directly to mc_return (no bootstrap). Cheap stability sanity check.
    VLAAQCConfig(
        name="vla_mc_baseline",
        notes="MC baseline: HL-Gauss regression to precomputed mc_return (no bootstrap). "
              "Stable lower-bound reference for the TD runs.",
        td=TDConfig(target_kind="mc"),
    ),
    # Capacity sweep brackets (share everything except arch).
    VLAAQCConfig(
        name="vla_aqc_td_a51_small",
        notes="Capacity bracket (small): n_embd=256 / 2 layers, linear state projection.",
        arch=ArchConfig(head_dim=32, num_layers=2, mlp_dim=512),
    ),
    VLAAQCConfig(
        name="vla_aqc_td_a51_large",
        notes="Capacity bracket (large): n_embd=512 / 6 layers, linear state projection.",
        arch=ArchConfig(head_dim=64, num_layers=6, mlp_dim=2048),
    ),
    # Ablation: same as default but with an MLP state-encoder (2048->512->n_embd) instead of
    # the linear projection -- to test whether the 2048-d latent needs nonlinear digestion.
    VLAAQCConfig(
        name="vla_aqc_td_a51_stateenc",
        notes="Default + MLP state-encoder (512,) on the 2048-d latent (vs linear default).",
        arch=ArchConfig(state_encoder_dims=(512,)),
    ),
    # Reward-normalized variant (fixed [-1,0] support via reward scaling).
    VLAAQCConfig(
        name="vla_aqc_td_a51_rnorm",
        notes="Same as default but reward-normalized to [-1,0] (support_mode=reward_norm).",
        dist=DistConfig(support_mode="reward_norm", v_min=-1.0, v_max=0.0),
        reward=RewardConfig(reward_normalize=True),
    ),
]

CONFIGS = {c.name: c for c in _CONFIGS}
assert len(CONFIGS) == len(_CONFIGS), "duplicate config name in registry"


def get_config(name: str) -> VLAAQCConfig:
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; available: {sorted(CONFIGS)}")
    return CONFIGS[name]
