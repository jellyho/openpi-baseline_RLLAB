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
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Nested config groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchConfig:
    """Causal-Transformer critic capacity (scaled up for the 2048-d VLA latent).

    Default sizing targets the requested ~10M-parameter critic: n_embd=384, 3 layers,
    mlp=1024, K=2 ensembles, per-position head -> ~10.7M params (the 2048->384 state
    projection and the macro_H x n_embd x num_atoms heads dominate the fixed cost).
    """
    num_ensembles: int = 2          # K (min-aggregated target)
    num_layers: int = 3             # 3L @ n_embd=384, K=2 -> ~10.7M (was 4L ~13.5M)
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
    num_atoms: int = 201            # 201 over [-1, 0] -> bin width 0.005
    hl_gauss_sigma_frac: float = 0.75   # sigma = frac * (v_max - v_min) / num_atoms
    # How the value support [v_min, v_max] is set:
    #   'fixed'       -> use v_min/v_max below
    #   'reward_norm' -> rewards scaled into [-1, 0]; support fixed to [-1, 0]
    #   'data'        -> p1/p99 of return-to-go + margin (DEAS data-centric)
    support_mode: Literal["fixed", "reward_norm", "data"] = "fixed"
    # v3 annotation (reward_annotate.py): raw living=-1/step, failure=-0.4*T_max, gamma=0.9999,
    # globally normalized so the deepest return-to-go is exactly -1 -> mc_return in [-1, 0].
    # This makes steps-to-go (hence prefix length) informative, which the [-0.5,0] v1/v2 scheme
    # did not (its tiny -1e-4 living cost gave a near-flat value -> prefix_spread collapsed).
    v_min: float = -1.0
    v_max: float = 0.0


@dataclass(frozen=True)
class TDConfig:
    """MC-return / multi-step TD bootstrap (the ACSAC expected-prefix-max)."""
    discount: float = 0.9999        # MUST match the precomputed mc_return column. v3 annotation
                                    # uses gamma=0.9999 (near-undiscounted; failure penalty stays
                                    # visible from episode start), mc_return in [-1,0]. cum_reward +
                                    # gamma^h*v_next and the MC floor all live on this scale.
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
    # Prefix subsample grid, stored as STEP COUNTS (not macro-prefix indices). Must each be a
    # multiple of macro_group_size and number at most macro_H = horizon // macro_group_size.
    # macro_group_size=25 -> macro_H=2 -> replan at 25 / 50 steps. (For =10 use (10,20,30,40,50);
    # for the standard =1 use e.g. (1,10,25,50).)
    prefixes: tuple[int, ...] = (25, 50)
    # Group this many consecutive per-step actions into one transformer token.
    # 25 → horizon 50 becomes 2 macro-action tokens (replan granularity 25/50 steps).
    # 10 → 5 macro-tokens; 1 = standard per-step tokenisation (H=50 action tokens).
    macro_group_size: int = 25
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
    # ReLU-blend MC-warmup target (the agreed design; see agent.critic_loss_td):
    #     target = G_MC + beta * ReLU(TD - G_MC),   beta ramped 0 -> 1.
    # mc_floor=True turns this on (default). It SUBSUMES the old behaviours:
    #   * beta=0 (warmup)          -> pure MC regression (ground the success manifold first);
    #   * beta=1 (after ramp)      -> max(G_MC, TD)  == the Cal-QL hard floor / max(MC,Q);
    #   * mc_warmup=mc_ramp=0      -> beta=1 from step 0 (the old hard-max floor, no warmup);
    #   * mc_floor=False           -> pure TD, no floor and no warmup.
    # The earlier OGBench worry that a hard floor inflates value early is exactly what the
    # warmup (beta=0 then a slow ramp) prevents: no bootstrap is trusted until MC has grounded
    # the values, so the NxH max can't amplify a randomly-high early Q.
    mc_floor: bool = True
    # beta(step) schedule (used only when mc_floor=True): beta=0 for the first mc_warmup_steps
    # (pure MC warmup), then a cosine ramp to 1 over mc_ramp_steps, then stays 1. Default ramp=0 =
    # a STEP function: beta jumps 0->1 at mc_warmup_steps. Since the target is the MC-floored
    # max(MC, TD), the jump can only raise the target above the MC floor (never below), so once MC
    # has grounded the values over the warmup a hard switch is safe; set mc_ramp_steps>0 to soften.
    mc_warmup_steps: int = 20_000
    mc_ramp_steps: int = 0
    # Speed optimization (no effect on the math beyond the MC-vs-beta=0 masking nuance): during the
    # beta=0 warmup the bootstrap term is multiplied by 0, so base_action is read + the NxH prefix-max
    # is computed only to be discarded. With warmup_skip the warmup runs as a pure-MC step that never
    # touches base_action (~44 it/s vs ~2-3), then switches to the TD step at mc_warmup_steps.
    warmup_skip: bool = True
    # Target network: bootstrap from an EMA copy of the online params, target <- (1-tau)*target +
    # tau*online each step. 0 = no target net (online critic). 0.005 (DEFAULT) stabilises the TD
    # bootstrap (the moving-target oscillation). At the warmup->TD switch the target is resynced to
    # the warmed-up online params (train loop) so the first bootstrap isn't from stale init weights.
    target_tau: float = 0.005
    # Randomly subsample this many of the N=num_candidates base actions for the bootstrap max each
    # batch (REDQ-style): <=0 or >=N uses all N. Smaller (8/16) = more conservative max (less
    # overestimation) AND less next_candidates data to move (faster). Applied in the loader.
    bootstrap_subset: int = 0
    # N-step return: accumulate N EXTRA real-reward steps BEYOND each commit prefix h before
    # bootstrapping -> target = (h+N)-step realized return + gamma^(h+N) * V(s_{t+h+N}). 0 = the
    # standard h-step backup. Larger N pushes the (unstable) Best-of-N bootstrap further out
    # (gamma^(h+N) smaller) => more real signal, less bootstrap reliance ("horizon reduction").
    # Implemented in the loader (shifts the next-state to t+h+N and the cum_reward to h+N steps).
    n_step: int = 0


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping for the value scale (interacts with DistConfig.support_mode)."""
    reward_normalize: bool = False  # scale rewards so return-to-go lands in [-1, 0]
    support_margin: float = 0.05    # margin for 'data' support / normalization
    # In-loader relabel of the raw reward column. DISABLED (=None): the v3 datasets already
    # hold the final normalized values on disk (reward = unnormalized_reward / Z, mc_return
    # at gamma=0.9999; see data_annoation/reward_annotate.py). Never re-relabel v3 data.
    relabel_living: Optional[float] = None    # disk already final; None = use as-is
    relabel_fail: float = -0.5                # only used if relabel_living is set (legacy)


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
# Local workspace on the B200 box (see setup_env.sh). Override with --data_root for other
# machines/clusters. mouse-battery is present (184GB, rl_token/base_action/reward/mc_return);
# the others are placeholders with the same naming so --task can switch once they land.
# Base dir holding the per-task <task>_annotated datasets. Set RLT_DATA_BASE in setup_env.sh
# (default = the /data5 box) so this isn't hand-edited per machine.
_DATA_BASE = os.environ.get("RLT_DATA_BASE", "/data5/jellyho/PFR_RSS/dataset/phase1_annotated")
TASKS = {
    # mouse-battery is v3-annotated on disk here (reward_annotate.py: living=-1/step,
    # fail=-0.4*T_max, gamma=0.9999, globally normalized so mc_return in [-1,0]). Matches the
    # config defaults: support [-1,0] + td.discount=0.9999.
    "insert-mouse-battery":  f"{_DATA_BASE}/insert-mouse-battery_v3_annotated",
    "seal-water-bottle-cap": f"{_DATA_BASE}/seal-water-bottle-cap_v3_annotated",
    "tower-of-hanoi-game":   f"{_DATA_BASE}/tower-of-hanoi-game_v3_annotated",
    "generalist":   f"{_DATA_BASE}/generalist_v3_annotated",
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
    data_root_override: Optional[str] = None  # absolute dataset path; overrides TASKS[task]
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
    # Preload the WHOLE dataset into RAM (decoded numpy, base_action fp16) at startup, so training
    # reads incur zero disk I/O and zero parquet re-decode -- only the per-batch host->device copy
    # remains. Needs RAM >= dataset size (~160GB for the full base_action set on a 377GB node) and
    # is meant for the single-process loader_processes=0 path (sharded multiprocess would duplicate
    # the cache per worker -> auto-ignored there). One-time decode cost at startup (a few minutes).
    # DEFAULT off: the bulk TD phase is GPU-bound (the loader already outruns the ~2-3 it/s step),
    # so preload barely moves end-to-end it/s. Opt in with --preload only if the full-set lustre I/O
    # (not the page cache) is shown to be the wall; size SLURM --mem >= dataset size accordingly.
    preload: bool = False
    # Fast index-based loader over a frame-indexed memmap (built once by scripts/preprocess_memmap.py).
    # When set, the loader gathers batches by random index from the memmap (OS page cache holds it
    # ONCE in RAM, shared read-only across all DDP workers -> no per-worker duplication, true global
    # shuffle, and next_candidates gathered lazily per-batch -- no parquet read, no giant concat/
    # permute). This is the recommended DDP fast path. "" -> the parquet streaming loader.
    memmap_dir: Optional[str] = None

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
    eval_n_intervention: int = 3    # fixed intervention episodes (inference+teleop; teleop span shaded)
    eval_n_fail: int = 3            # fixed failure episodes shown in the eval plot
    save_interval: int = 25_000
    keep_period: Optional[int] = 100_000   # checkpoints at step % keep_period == 0 are kept
    checkpoint_base_dir: str = field(default_factory=lambda: os.environ.get(
        "RLT_CRITIC_CKPT_DIR", "/data5/jellyho/PFR_RSS/checkpoints/rlt_critic_runs"))
    wandb_enabled: bool = True
    wandb_project: str = "rlt_critic_learning"
    wandb_entity: str = "RSS-PFT_RLLAB"

    # ---- derived identity --------------------------------------------------
    @property
    def data_root(self) -> str:
        """Dataset path: the explicit override if set, else TASKS[task]."""
        return self.data_root_override or TASKS[self.task]

    def mc_blend_beta(self, step: int) -> float:
        """ReLU-blend warmup coefficient beta(step) in [0, 1] (see agent.critic_loss_td).

        beta = 0 for the first ``td.mc_warmup_steps`` (pure MC warmup), then a cosine ramp
        to 1 over ``td.mc_ramp_steps``, then stays 1 (target = max(MC, TD)). When mc_floor
        is off there is no blend (returns 1.0; the loss then uses the pure-TD branch).
        """
        if not self.td.mc_floor:
            return 1.0
        w, r = self.td.mc_warmup_steps, self.td.mc_ramp_steps
        if step < w:
            return 0.0
        if r <= 0 or step >= w + r:
            return 1.0
        t = (step - w) / r
        return 0.5 * (1.0 - math.cos(math.pi * t))

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
        if t.mc_floor:                               # tag ReLU-blend floor + warmup schedule
            if t.mc_warmup_steps == 0 and t.mc_ramp_steps == 0:
                parts.append("mcfloor")              # beta=1 from step 0 (hard max, no warmup)
            else:
                parts.append(f"warm{t.mc_warmup_steps // 1000}k+{t.mc_ramp_steps // 1000}k")
        if self.reward.reward_normalize:
            parts.append("rnorm")
        if self.commander_filter:
            parts.append("+".join(self.commander_filter))
        if t.target_tau > 0:                         # EMA target network -> distinct run_dir per tau
            parts.append(f"tnet{t.target_tau:g}")
        if t.bootstrap_subset > 0:                   # random candidate subset -> distinct run_dir per M
            parts.append(f"sub{t.bootstrap_subset}")
        if t.n_step > 0:                             # N-step return -> distinct run_dir per N
            parts.append(f"n{t.n_step}")
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
    # ---- PRIMARY: the agreed design ----------------------------------------------------
    # Per-prefix multi-step TD with the 32-candidate joint-max bootstrap, ReLU-blend MC
    # warmup (beta 0 for 20k steps, then a step to 1), HL-Gauss 201 atoms over [-0.5,0], ~10M critic
    # (n_embd=384/3L/K2), macro grouping (5 macro-tokens -> replan at 10/20/30/40/50 steps).
    VLAAQCConfig(
        name="vla_aqc_warmup",
        notes="PRIMARY. AQC-TD on mouse-battery: pure-MC warmup for 20k steps (no base_action) "
              "then a HARD switch to TD bootstrap (ramp=0) with the MC floor + EMA target net. "
              "~10M critic, [-0.5,0] support, gamma=0.995, B=256.",
    ),
    # ---- Warmup stage 1 in isolation: pure MC regression (no bootstrap) -----------------
    # Cheap, very stable; good first sanity run and a lower-bound reference. Equivalent to
    # the PRIMARY run's first phase (beta=0) run forever.
    VLAAQCConfig(
        name="vla_mc",
        notes="Pure MC regression to the precomputed mc_return (no bootstrap). Stable "
              "lower-bound reference; the beta=0 phase of vla_aqc_warmup in isolation.",
        td=TDConfig(target_kind="mc"),
    ),
    # ---- Ablations on the transition --------------------------------------------------
    # Hard max(MC,TD) from step 0 (no warmup): the old Cal-QL floor. Tests whether the
    # warmup ramp actually matters vs trusting the bootstrap immediately.
    VLAAQCConfig(
        name="vla_aqc_hardmax",
        notes="Ablation: hard max(MC,TD) floor from step 0 (mc_warmup=mc_ramp=0, beta=1). "
              "No warmup -- the pre-warmup baseline to measure the ramp's benefit.",
        td=TDConfig(mc_warmup_steps=0, mc_ramp_steps=0),
    ),
    # Pure TD, no MC floor at all (mc_floor=False). The other extreme: tests how slow raw
    # bootstrapping is on this long-horizon offline data (the problem the warmup addresses).
    VLAAQCConfig(
        name="vla_aqc_no_floor",
        notes="Ablation: pure multi-step TD, no MC floor and no warmup (mc_floor=False). "
              "Baseline for how slow unaided long-horizon propagation is.",
        td=TDConfig(mc_floor=False),
    ),
    # Conservative bootstrap: softmax(beta=20) aggregation over the NxH candidate-prefix Q's
    # instead of the hard Best-of-N max, on top of the warmup. Guards offline overestimation.
    VLAAQCConfig(
        name="vla_aqc_warmup_softmax",
        notes="PRIMARY + conservative softmax(beta=20) bootstrap aggregation (soft Best-of-N) "
              "over the candidate x prefix Q's. Compare vs vla_aqc_warmup (hard max).",
        td=TDConfig(agg_mode="softmax", agg_beta=20.0),
    ),
    # ---- Capacity brackets (share everything except arch) -------------------------------
    VLAAQCConfig(
        name="vla_aqc_warmup_small",
        notes="Capacity bracket (small): n_embd=256 / 2 layers, mlp=512 (~4M).",
        arch=ArchConfig(head_dim=32, num_layers=2, mlp_dim=512),
    ),
    VLAAQCConfig(
        name="vla_aqc_warmup_large",
        notes="Capacity bracket (large): n_embd=512 / 6 layers, mlp=2048 (~35M).",
        arch=ArchConfig(head_dim=64, num_layers=6, mlp_dim=2048),
    ),
    # Ablation: MLP state-encoder (2048->512->n_embd) instead of the single linear projection
    # -- tests whether the 2048-d VLA latent needs nonlinear digestion before the critic.
    VLAAQCConfig(
        name="vla_aqc_warmup_stateenc",
        notes="PRIMARY + MLP state-encoder (512,) on the 2048-d latent (vs linear default).",
        arch=ArchConfig(state_encoder_dims=(512,)),
    ),
    # ---- Fast-iteration DEBUG config on the 30-episode seal_mini subset ------------------
    # 10 success + 10 failure + 10 intervention (built by .diag/build_mini.py). Short beta
    # schedule + frequent eval so the value-curve behaviour is visible in minutes. seal_mini is
    # written ONE FULL EPISODE PER PARQUET ROW-GROUP, so an in-loader mc_gamma override (td.mc_gamma)
    # recomputes FULL-EPISODE mc_return correctly -- handy for sweeping the discount on this set.
    VLAAQCConfig(
        name="vla_aqc_insert-mouse-battery",
        task="insert-mouse-battery",
        notes="MC warmup (0k) -> hard TD switch (ramp=0), "
              "EMA target net (tau=0.005) for stability.",
        td=TDConfig(mc_warmup_steps=0, mc_ramp_steps=0),
        optim=OptimConfig(num_train_steps=1_000_000, learning_rate=3e-4),  # faster convergence on the small set
        log_interval=100, eval_interval=10_000, save_interval=50_000, keep_period=250_000,
    ),
    VLAAQCConfig(
        name="vla_aqc_seal-water-bottle-cap",
        task="seal-water-bottle-cap",
        notes="MC warmup (0k) -> hard TD switch (ramp=0), "
              "EMA target net (tau=0.005) for stability.",
        td=TDConfig(mc_warmup_steps=0, mc_ramp_steps=0),
        optim=OptimConfig(num_train_steps=1_000_000, learning_rate=3e-4),  # faster convergence on the small set
        log_interval=100, eval_interval=10_000, save_interval=50_000, keep_period=250_000,
    ),
    VLAAQCConfig(
        name="vla_aqc_tower-of-hanoi-game",
        task="tower-of-hanoi-game",
        notes="MC warmup (0k) -> hard TD switch (ramp=0), "
              "EMA target net (tau=0.005) for stability.",
        td=TDConfig(mc_warmup_steps=0, mc_ramp_steps=0),
        optim=OptimConfig(num_train_steps=1_000_000, learning_rate=3e-4),
        log_interval=100, eval_interval=10_000, save_interval=50_000, keep_period=250_000,
    ),
    VLAAQCConfig(
        name="vla_aqc_generalist",
        task="generalist",
        notes="MC warmup (0k) -> hard TD switch (ramp=0), "
              "EMA target net (tau=0.005) for stability.",
        td=TDConfig(mc_warmup_steps=0, mc_ramp_steps=0),
        optim=OptimConfig(num_train_steps=1_000_000, learning_rate=3e-4),
        log_interval=100, eval_interval=10_000, save_interval=50_000, keep_period=250_000,
    ),
    # ---- DEBUG exp2: paper-style PROGRESS reward (Sec 3.1, Eq.1) -------------------------
    # Target mc_return = gamma^(T-t) * I(success): positive "task progress" rising to 1 at a
    # successful terminal, 0 for failures (no penalty). Support [0,1]. Schedule = the universal
    # one: pure-MC warmup (5k steps, no base_action) then a HARD switch to TD bootstrap (ramp=0).
    # Data = seal_mini_progress (.diag/build_mini_progress.py); has an explicit `done` column for
    # the bootstrap terminal mask. NOTE: progress signal WITHOUT the paper's hindsight-failure
    # augmentation (truncate-at-retry); natural failures here are flat-0. Add hindsight next.
    VLAAQCConfig(
        name="vla_aqc_mini_progress",
        task="seal-water-bottle-cap",
        data_root_override="/lustre/jellyho/seal_mini_progress",
        notes="DEBUG exp2: progress reward gamma^(T-t)*I, support [0,1]. MC warmup (5k) -> hard TD "
              "switch (ramp=0) + EMA target net. Sparse outcome reward (+1/-0.5 terminal) + done column.",
        dist=DistConfig(v_min=0.0, v_max=1.0),
        td=TDConfig(target_kind="td", mc_warmup_steps=5_000, mc_ramp_steps=0),
        optim=OptimConfig(num_train_steps=50_000),
        log_interval=100, eval_interval=1_000, save_interval=5_000, keep_period=25_000,
    ),
]

CONFIGS = {c.name: c for c in _CONFIGS}
assert len(CONFIGS) == len(_CONFIGS), "duplicate config name in registry"


def get_config(name: str) -> VLAAQCConfig:
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; available: {sorted(CONFIGS)}")
    return CONFIGS[name]
