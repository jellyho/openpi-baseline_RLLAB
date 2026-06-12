# `openpi.rlt_critic` — RLT / Adaptive-Q-Chunking critic

A lightweight **ACSAC prefix-critic** trained on the *precomputed* RLT latents of a frozen
VLA, absorbed into openpi (no dependency on the standalone `adaptive_q_chunking/` tree). It
learns `Q(z_rl, a_{t:t+h})` for **every prefix length** `h` from the annotated LeRobot columns
`rl_token` (2048-d frozen-VLA latent) and `base_action` (N candidate action chunks), so training
runs **without any VLA forward pass** — fast and cheap. At deployment the critic scores the
actor's N samples and selects both the chunk and how many steps to commit (adaptive Q-chunking).

This is the AQC line (a small transformer on latents), *not*
`pi0_alphaflow_critic.Pi0WithCritic` (a 3rd Gemma expert needing the full VLA forward).

> **Non-breaking by construction.** Everything here is new files under `src/openpi/rlt_critic/`
> + `scripts/train_rlt_critic*` + `scripts/smoke_aqc_full.py`. No existing openpi model,
> `Policy`, training config, or checkpoint is modified. Stages 2–3 *compose* `Pi0RLT`/`Policy`.

---

## End-to-end pipeline

```
(1) RLT annotate     scripts/compute_rl_tokens.py        -> rl_token + base_action columns
(2) reward v3        adaptive_q_chunking/data_annoation/reward_annotate.py  -> reward + mc_return
(3) train critic     scripts/train_rlt_critic.py (.sh / _supervised.sh)     -> critic checkpoints
(4) merge            python -m openpi.rlt_critic.merge   -> one deployable bundle dir
(5) deploy           inference.create_aqc_policy(bundle) -> AQCPolicy.infer(obs) (adaptive)
```

Steps (1)–(2) produce the dataset the critic reads; (3) is this package's core; (4)–(5) are
`merge.py` / `inference.py`.

---

## Data & reward (v3)

The critic reads an annotated LeRobot v3.0 dataset (one task = one dataset). Columns:

| column | dtype | shape | meaning |
|---|---|---|---|
| `rl_token` | f32 | `[2048]` | frozen-VLA bottleneck latent = critic state token |
| `base_action` | f16 | `[32,50,14]` | N=32 base-policy candidate chunks (raw action space) |
| `action` | f32 | `[14]` | executed behavior action (raw) |
| `reward` | f32 | `[1]` | **v3-normalized** reward |
| `mc_return` | f32 | `[1]` | **v3-normalized** return-to-go (γ=0.9999) in `[-1, 0]` |
| `unnormalized_reward` | f32 | `[1]` | raw reward before /Z (re-normalization later) |

**v3 reward scheme** (`reward_annotate.py`): raw living = `-1`/step, success terminal = `0`,
failure terminal = `-0.4·T_max`; γ=0.9999 return-to-go; globally normalized by `Z = |min return|`
so `mc_return ∈ [-1, 0]`. This makes *steps-to-go* (hence prefix length) informative — the earlier
v1/v2 scheme (living `-1e-4`) gave a near-flat value, collapsing `prefix_spread` to ~1e-4 (no
adaptive-chunking signal). Local mouse-battery numbers: `T_max=10000`, `C_fail=4000`, `Z=7548`,
1119 episodes (124 failures), success-start median `-0.30`, failure-start `-0.67`.

Re-annotate a v1/v2 dataset in place (idempotent; skips if already v3):

```bash
.venv/bin/python adaptive_q_chunking/data_annoation/reward_annotate.py \
  --input <dataset_root> --inplace --workers 4          # add --dry_run for the design summary
```

---

## Design — MC-warmup → `max(MC, Q)` via a ReLU-blend

Offline RL on long-horizon data; the target is `max(MC return, Q-backup)`. Pure bootstrapping
propagates slowly, so the critic **warms up on the Monte-Carlo return**, then *gradually* trusts
the bootstrap, via one interpolated target ([agent.py](agent.py) `critic_loss_td`):

```
y = G_MC + β · ReLU( r + γ·Q̄(s', a') − G_MC )
```

- `β = 0` → pure MC warmup (regress every prefix to the realized return-to-go; grounds the whole
  success manifold with no bootstrap, and suppresses early Q-backup overestimation).
- `β = 1` → `max(G_MC, Q-backup)` — the Cal-QL-floored target the algorithm ultimately wants.
- MC stays a **hard lower-bound floor**; `β` scales only the *improvement* term above it. `β` ramps
  `0 → 1` (cosine) via `config.mc_blend_beta(step)`: `mc_warmup_steps` pure MC, then `mc_ramp_steps`
  ramp, then `β = 1` forever.

`mc_floor=False` → pure TD (no floor); `mc_warmup=mc_ramp=0` → `β=1` from step 0 (hard-max floor).

---

## Train (Stage 1)

```bash
# detached single run (survives the shell)
scripts/train_rlt_critic.sh vla_aqc_warmup 64 3                 # CONFIG BATCH GPU

# overnight / unattended: auto-resume on crash from the last 25k checkpoint
scripts/train_rlt_critic_supervised.sh vla_aqc_warmup 64 3

# direct
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.038 \
  .venv/bin/python scripts/train_rlt_critic.py --config vla_aqc_warmup --batch_size 64 --loader_processes 4
```

- **Critic** (`vla_aqc_warmup`, default): `n_embd=384 / 3 layers / K=2 ensembles`, HL-Gauss 201
  atoms over `[-1, 0]`, macro-grouping 10 → replan granularity at **10/20/30/40/50** steps →
  **~10.75M params**. Bootstrap over `N=32 × 5` prefixes, `γ=0.9999`, warmup 20k + ramp 30k.
- **Data**: `config.TASKS[task]` (default local mouse-battery); override with `--data_root <path>`.
- **Output**: msgpack `params`+`opt_state` checkpoints (every 25k) + `metrics.csv` + offline W&B
  under `config.checkpoint_base_dir/<name>/<run_name>/`. Resume with `--resume`.
- **Presets** (`config.py`): `vla_aqc_warmup` (primary) · `vla_mc` (pure-MC baseline) ·
  `vla_aqc_hardmax` / `vla_aqc_no_floor` / `vla_aqc_warmup_softmax` (transition ablations) ·
  `vla_aqc_warmup_{small,large,stateenc}` (capacity). CLI overrides: `--batch_size --lr --mc_floor
  --mc_warmup_steps --mc_ramp_steps --agg_beta --task --data_root --seed --loader_processes`.

**Multi-GPU**: data-parallel (DDP-equivalent) is built in — params replicated, batch sharded over
the device mesh, grad all-reduce inserted by XLA. It activates automatically when several GPUs are
visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`, `batch_size % n_gpu == 0`). **FSDP is intentionally absent**:
a 10M critic fits on one GPU, so sharding params would only add comm overhead.

**Memory**: the bootstrap forward (over `B·P·N` candidate sequences) is the driver. On a busy GPU
with ~7GB free, `B=64` fits (~6.9GB); `B≥128` needs a freer GPU. `PREALLOCATE=false` keeps any
cap-OOM contained to this job (never evicts co-tenants).

---

## Stage 2 — merge (`merge.py`)

```bash
python -m openpi.rlt_critic.merge \
  --rlt-config <rlt_config_name> --rlt-checkpoint <rlt_step_dir> \
  --critic-run-dir <critic_run_dir> --critic-step latest --out <bundle>
```

Builds a deployable **bundle** directory: `params/` (RLT orbax params, symlink or `--copy-rlt`) +
`critic/{params.msgpack,net.json}` + `aqc_manifest.json`. A bundle (not a single spliced pytree)
keeps the nnx VLA and the linen critic robustly co-located and framework-agnostic.

**Two deployable flavors — same merge, same critic, just the `--rlt-config`.** merge is
RLT-agnostic (it only records `rlt_config_name`); the inference wrapper then auto-detects the model
class and picks the matching forward path, so the *same trained critic* deploys against either RLT:

| `--rlt-config` | model | backbone forwards / step | token |
|---|---|---|---|
| `pi05_<task>_rlt` | `Pi0RLT` | **2** (image-only token + full sampling) | language-free |
| `pi05_<task>_rlt_joint` | `Pi0RLTJoint` | **1** (token from the sampling forward) | language-conditioned |

The critic action space is identical for both (raw joint), so a critic trained on one task's
annotations serves either RLT flavor of that task.

## Stage 3 — adaptive inference (`inference.py`)

`AQCAdaptive.load(bundle).sample_actions(rng, obs, exec_mode=...)`: RLT samples N base chunks +
`z_rl` (Joint: one backbone forward; vanilla: token + sampling) → decode to raw action space →
prefix critic scores `Q(z_rl, a_{1:h})` (ensemble-min) → joint arg-max `(n*, h*)` → return the
chosen chunk in **raw action space** (absolute joint targets) under:

- **`truncate`** — `chunk[:h*]` (`[h*, 14]`): execute `h*` steps, then replan.
- **`absolute_hold`** — full `[H, 14]` with the tail held at the `h*`-th action: execute `h*`
  effective steps when outputting absolute joint targets.

**Deploy** as an openpi policy:

```python
from openpi.rlt_critic.inference import create_aqc_policy
policy = create_aqc_policy("<bundle>", exec_mode="truncate")   # AQCPolicy(BasePolicy)
out = policy.infer(obs_dict)   # -> {actions, h_star, n_star, q_by_h, policy_timing}
```

`infer` applies the RLT config's input transforms (repack / normalize / tokenize / resize), runs
the adaptive selection, and returns raw actions. norm_stats load from the RLT checkpoint that
produced the bundle (same stats as training/annotation). Drop-in for the websocket policy server.

> **Raw-space critic query.** The critic trains on the raw joint `action` / `base_action` columns
> (no normalization). `sample_actions` therefore **decodes** each sampled chunk
> (Unnormalize → AbsoluteActions → YamOutputs) back to that raw joint space *before* scoring — the
> decoder is identical to the one `compute_rl_tokens.py` used to write `base_action`, so the
> train/infer action spaces match exactly. (Requires `norm_stats`; without it the score is a
> shape-only passthrough and is loudly warned — smoke tests only.)

**Verify either flavor end-to-end without a trained critic** (`scripts/smoke_aqc_full.py
--mock-critic` fabricates a random-init critic; values are meaningless, it proves the path runs):

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_aqc_full.py \
  --rlt-config pi05_seal-water-bottle-cap_rlt \
  --rlt-checkpoint checkpoints/pi05_seal-water-bottle-cap_rlt/.../20000 \
  --mock-critic --num-samples 2 --num-flow-steps 2
# vanilla (joint=False, decode=real): real 2B forward -> critic prefix-Q -> (n*,h*) -> both exec modes  ✓
```

---

## Files

| file | role |
|---|---|
| `config.py` | `VLAAQCConfig` (frozen) + preset registry + `mc_blend_beta` schedule |
| `agent.py` | `VLACriticTrainer`: MC / ReLU-blend-TD losses, expected-prefix-max bootstrap |
| `transformer.py` | `PrefixValue` causal critic (macro-grouped, per-position heads, ensemble) |
| `data.py` | `VLALeRobotDataset`: row-group-streaming loader over the annotated parquet |
| `loader.py` | torch multiprocess batch loader (openpi pattern; JAX-free in workers) |
| `networks.py` / `distributional.py` | `default_init`/`ensemblize`; HL-Gauss transforms |
| `merge.py` | Stage 2 — build the deployable bundle |
| `inference.py` | Stage 3 — `AQCAdaptive`, `AQCPolicy`, `create_aqc_policy` |
| `scripts/train_rlt_critic.py` `.sh` `_supervised.sh` | training entry / launcher / auto-resume |
| `scripts/smoke_aqc_full.py` | full e2e smoke (real RLT backbone) |

## Status

- Stages 1–3 implemented; **full end-to-end verified** with a real RLT checkpoint
  (`scripts/smoke_aqc_full.py` on hanoi `rlt_joint/99999`: load 2B backbone → real RLT forward →
  decode → critic prefix-Q → `(n*,h*)` → `truncate`→`[20,14]`, `absolute_hold`→`[50,14]`).
- For a *task-correct* deployment the merge needs the **matching RLT checkpoint** (e.g.
  mouse-battery `_rlt`) — the local one is tower-of-hanoi — plus the finished v3 critic. Then
  `merge` once + `create_aqc_policy` / `smoke_aqc_full.py` once.
