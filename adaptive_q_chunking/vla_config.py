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
    num_atoms: int = 201            # 201 -> bin width 0.005 over [-1, 0] (matches RECAP)
    hl_gauss_sigma_frac: float = 0.75   # sigma = frac * (v_max - v_min) / num_atoms
    # How the value support [v_min, v_max] is set:
    #   'fixed'       -> use v_min/v_max below (default [-1, 0] for undiscounted MC)
    #   'reward_norm' -> rewards scaled into [-1, 0]; support fixed to [-1, 0]
    #   'data'        -> p1/p99 of return-to-go + margin (DEAS data-centric)
    support_mode: Literal["fixed", "reward_norm", "data"] = "fixed"
    v_min: float = -1.0            # covers undiscounted MC range; discounted [-0.5,0] fits inside
    v_max: float = 0.0


@dataclass(frozen=True)
class TDConfig:
    """MC-return / multi-step TD bootstrap (the ACSAC expected-prefix-max)."""
    discount: float = 0.999         # long-horizon task; effective horizon 1/(1-g)=1000 steps
    # Target kind:
    #   'td' -> per-prefix multi-step TD with the N-candidate joint-max bootstrap (paper)
    #   'mc' -> regress directly to precomputed mc_return (RECAP-style baseline; no bootstrap)
    target_kind: Literal["td", "mc"] = "td"
    # Discount used to compute cumulative rewards in the TD bootstrap.
    # For MC regression (target_kind='mc') we use undiscounted (gamma=1) MC return,
    # computed from raw rewards — matching RECAP's Eq.1 and independent of the precomputed
    # mc_return column (which uses gamma=0.995). Set mc_gamma=None to use the dataset column.
    # NOW None: the dataset was re-annotated in place with the correct FULL-EPISODE mc_return
    # at gamma=0.999 (data_annoation/reward_annotate.py), so read that precomputed column
    # directly. (Recomputing per-row-group truncates episodes at ~1000-row group boundaries;
    # the disk column is the correct un-truncated value.) For the TD path next_mc_return is only
    # read at terminals (= terminal reward), so this is exact; for the MC baseline it avoids
    # the truncation. Set to a float only to override with an in-loader recompute.
    mc_gamma: Optional[float] = None   # use the precomputed gamma=0.999 column on disk
    num_candidates: int = 32        # N (== base_action's 32); the bootstrap max is over N x H
    # Prefix subsample grid, stored as STEP COUNTS (not macro-prefix indices).
    # Default: all 5 macro-prefix positions when macro_group_size=10.
    # With macro_group_size=1 (standard): use e.g. (1, 10, 25, 50).
    prefixes: tuple[int, ...] = (10, 20, 30, 40, 50)
    # Group this many consecutive per-step actions into one transformer token.
    # 10 → horizon 50 becomes 5 macro-action tokens (L=6 sequence, same as OGBench).
    # 1 = standard per-step tokenisation (H=50 action tokens).
    macro_group_size: int = 10
    terminal_uses_mc: bool = True   # at the -0.5 failure terminal, bootstrap = mc_return
    # How to aggregate the N-candidate x prefix Q's into the bootstrap value V(s'):
    #   'max'      -> hard max (Best-of-N / EMaQ; optimistic, current default)
    #   'softmax'  -> Boltzmann weighted mean  sum_j softmax(beta*q)_j * q_j  (conservative)
    #   'mellowmax'-> (1/beta) log mean_j exp(beta*q)  (conservative; contraction-preserving)
    # beta is the inverse temperature: beta->inf recovers max, beta->0 gives the plain mean.
    # NOTE softmax is scale-VARIANT (only beta*(Q-gap) matters); our values live in [-1,0] with
    # small gaps, so beta must be larger than ACH's beta=1. beta=4 = mild conservatism: it keeps
    # large gaps (0.2-0.4) but blurs small ones (~0.05). Sweep beta when ablating soft vs max.
    agg_mode: Literal["max", "softmax", "mellowmax"] = "max"
    agg_beta: float = 4.0
    # Cal-QL-style MC floor on the TD target: target = max(td_target, mc_return(start state)).
    # Lower-bounds the value by the realized behavior return V^beta(s) -> boosts early/over-
    # pessimistic TD estimates up to the achievable return and injects the terminal/return signal
    # densely (counters slow propagation from sparse terminals). Valid (Q* >= V^beta), self-
    # releases once TD >= mc. Default on; set False to ablate.
    mc_floor: bool = True


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping for the value scale (interacts with DistConfig.support_mode)."""
    reward_normalize: bool = False  # scale rewards so return-to-go lands in [-1, 0]
    support_margin: float = 0.05    # margin for 'data' support / normalization
    # In-loader relabel of the raw reward column. NOW DISABLED (=None): the dataset was
    # re-annotated IN PLACE (data_annoation/reward_annotate.py) so the disk already holds
    # living=-4e-4, success=0.0, failure=-0.5, and mc_return at gamma=0.999. Applying the
    # x4 relabel again would DOUBLE it (-4e-4 -> -1.6e-3). Keep None unless pointing the
    # loader at a fresh raw dataset (then set relabel_living=-4e-4, relabel_fail=-0.5).
    relabel_living: Optional[float] = None    # disk already relabeled; None = use as-is
    relabel_fail: float = -0.5                # only used if relabel_living is set


@dataclass(frozen=True)
class OptimConfig:
    batch_size: int = 256
    lr: float = 3e-4
    num_train_steps: int = 500_000  # ceiling; use eval-based early stop (~200-400k typical)
    warmup_steps: int = 2_000
    weight_decay: float = 0.0
    max_grad_norm: Optional[float] = None   # None = no clipping (QC has none)


# ---------------------------------------------------------------------------
# Challenge tasks: annotated dataset (rl_token/base_action + relabeled reward/mc_return).
# Select with VLAAQCConfig.task or `--task <name>`; data_root is derived from this.
# ---------------------------------------------------------------------------
_DATA_BASE = "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset"
TASKS = {
    "insert-mouse-battery":  f"{_DATA_BASE}/insert-mouse-battery_annotated",   # ready (relabeled)
    "seal-water-bottle-cap": f"{_DATA_BASE}/seal-water-bottle-cap_annotated",  # ready (relabeled)
    "tower-of-hanoi-game":   f"{_DATA_BASE}/tower-of-hanoi-game_annotated",    # NOT annotated yet
}


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
    task: str = "insert-mouse-battery"   # which challenge task (see TASKS); --task to switch
    commander_filter: Optional[tuple[str, ...]] = None  # e.g. ("inference",) or ("teleop",)
    shuffle_buffer_groups: int = 8
    num_workers: int = 8            # parquet read THREADS per loader process
    prefetch_depth: int = 3         # batches prefetched (thread queue / per torch worker)
    # Loader worker PROCESSES (torch DataLoader, openpi-style; see vla_loader.py).
    # The pipeline is host-bound, so this is the throughput knob: each process streams a
    # disjoint shard of row-groups and yields full global batches => ~linear scaling until
    # CPU/lustre saturates. Total read threads = loader_processes * num_workers (size
    # SLURM --cpus-per-task accordingly). 0 = legacy in-process thread prefetch.
    loader_processes: int = 4

    # --- grouped hyperparameters ---
    arch: ArchConfig = field(default_factory=ArchConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    td: TDConfig = field(default_factory=TDConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # --- run management ---
    seed: int = 0
    log_interval: int = 1_000       # ~7.7 min @ 2.16 it/s (slow run -> less log noise)
    eval_interval: int = 10_000     # trajectory value-curve viz -> W&B eval/value_curves (~77 min)
    eval_n_success: int = 3         # fixed success episodes shown in the eval plot
    eval_n_fail: int = 3            # fixed failure episodes shown in the eval plot
    save_interval: int = 25_000
    keep_period: Optional[int] = 100_000   # checkpoints at step % keep_period == 0 are kept
    checkpoint_base_dir: str = "/scratch/gwanwoo13/rss_pft/phase1/critic_learning"
    wandb_enabled: bool = True
    wandb_project: str = "rlt_critic_learning"
    wandb_entity: str = "RSS-PFT_RLLAB"

    # ---- derived identity --------------------------------------------------
    @property
    def data_root(self) -> str:
        """Dataset path for the selected task (single source of truth: TASKS)."""
        return TASKS[self.task]

    @property
    def run_name(self) -> str:
        """Descriptive, self-documenting run name built from the load-bearing settings."""
        # Task prefix so runs on different datasets never collide (same config, diff task).
        # e.g. "seal-water-bottle-cap_a201_sup-fixed_emb384x4L_N32_P5_b256_g0.999_mcfloor_s0".
        # To resume a pre-existing run whose dir lacks the prefix, just `mv` its dir to the new
        # run_name (printed at train start) -- resume finds checkpoints by dir name.
        a, d, t = self.arch, self.dist, self.td
        parts = [
            self.task,                                  # which dataset
            f"a{d.num_atoms}",                          # atoms
            f"sup-{d.support_mode}",                    # value support choice
            f"emb{a.n_embd}x{a.num_layers}L",           # capacity
            f"N{t.num_candidates}",                     # candidates
            f"P{len(t.prefixes)}",                      # prefix grid size
            f"b{self.optim.batch_size}",
            f"g{t.discount}",
        ]
        if t.agg_mode != "max":                      # tag soft aggregation (avoid collision)
            parts.append(f"{t.agg_mode}{t.agg_beta:g}")
        if t.mc_floor:                               # tag Cal-QL MC floor (default on)
            parts.append("mcfloor")
        if self.reward.reward_normalize:
            parts.append("rnorm")
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
        assert self.task in TASKS, f"unknown task {self.task!r}; available: {sorted(TASKS)}"
        assert self.dist.v_min < self.dist.v_max, "value support must be non-empty"
        assert self.horizon % self.td.macro_group_size == 0, \
            f"horizon {self.horizon} must be divisible by macro_group_size {self.td.macro_group_size}"
        macro_H = self.horizon // self.td.macro_group_size
        assert all(p % self.td.macro_group_size == 0 for p in self.td.prefixes), \
            "all prefixes (step counts) must be divisible by macro_group_size"
        assert min(self.td.prefixes) >= self.td.macro_group_size
        assert max(self.td.prefixes) <= self.horizon
        assert len(self.td.prefixes) <= macro_H, "more prefixes than macro-tokens"
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
    # MC baseline with macro-action grouping (recommended first run):
    # macro_group_size=10 -> 5 tokens, L=6 sequence, 5 prefix candidates {10,20,30,40,50}.
    VLAAQCConfig(
        name="vla_mc_macro",
        notes="MC regression baseline with macro-action grouping (group_size=10). "
              "5 macro-tokens, L=6 sequence (same as OGBench critic). "
              "Start here: stable MC target, cheap, meaningful replan granularity.",
        td=TDConfig(target_kind="mc", macro_group_size=10,
                    prefixes=(10, 20, 30, 40, 50)),
    ),
    # TD variant with macro-action grouping (after MC baseline works).
    VLAAQCConfig(
        name="vla_aqc_td_macro",
        notes="AQC-TD with macro-action grouping. 32 candidates x 5 prefixes = 160 evals/step "
              "(vs 1600 without grouping). Switch to this after vla_mc_macro validates.",
        td=TDConfig(target_kind="td", macro_group_size=10,
                    prefixes=(10, 20, 30, 40, 50)),
    ),
    # Soft-aggregation variant: conservative bootstrap via Boltzmann softmax over the
    # N x prefix Q's instead of hard max (Best-of-N). beta=4 = mild conservatism at our
    # [-1,0] scale. Ablation against vla_aqc_td_macro (max) for offline overestimation.
    VLAAQCConfig(
        name="vla_aqc_td_macro_softmax",
        notes="AQC-TD macro with softmax(beta=4) bootstrap aggregation (conservative vs max). "
              "Compare against vla_aqc_td_macro to test offline overestimation.",
        td=TDConfig(target_kind="td", macro_group_size=10,
                    prefixes=(10, 20, 30, 40, 50),
                    agg_mode="softmax", agg_beta=4.0),
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
