# Adaptive Q-Chunking (AQC / ACSAC)

This directory adds **Adaptive Q-Chunking** — a faithful implementation of
**ACSAC: Adaptive Chunk Size Actor-Critic with Causal Transformer Q-Network**
(`../acsac.pdf`) — on top of the existing **QC / Q-Chunking** codebase.

The full design rationale, paper summary, and every documented design decision live in
[`../implementation_plan.html`](../implementation_plan.html). This file is the practical
quick-reference: what changed, which files, and how to train / evaluate.

---

## TL;DR — what AQC does differently

QC trains a **fixed-size** chunked critic `Q(s, a_{1:H})` and always executes `H` actions
before replanning. AQC instead learns a critic that scores **every prefix**
`Q(s, a_{1:h})` for `h = 1…H` at once, and lets the **chunk size become state-dependent**:
at each replanning state it picks the prefix that maximises the critic and executes only
that many actions.

| | QC (`acfql`, best-of-n) | **AQC (`aqc`)** |
|---|---|---|
| Critic | MLP, scalar `Q(s, a_{1:H})` | **Causal Transformer**, vector `[Q(s, a_{1:h})]_{h=1..H}` |
| TD target | one `H`-step target | **`H` per-horizon targets**, gradient-averaged (Eq. 6) |
| Bootstrap states | only `s_{t+H}` | **all `s_{t+1..t+H}`** |
| Action selection | best-of-`N` chunks | **best-of-`N×H`** (candidate × prefix), joint arg-max (Eq. 8) |
| Bootstrap value | `max_n` | **`max_{n,h}`** expected-prefix-max (Eq. 19) |
| Ensemble agg. | mean | **min** (K=2) |
| Output head | single scalar head | **one head per prefix position** (Prop. G.7); shared head optional |
| Critic | scalar regression | **regression or distributional** (HL-Gauss); selectable |
| Target network | Polyak (`τ`) | **online critic, stop-grad** (Pre-LN stability); Polyak optional |
| Deployment | fixed `H` actions / replan | **adaptive `h*` actions / replan** |
| Policy | flow-BC (+ optional distill) | flow-BC + rejection sampling (reused from QC) |

The flow-BC policy, dataset chunking, replay buffer, training loop, evaluation harness,
and config system are **reused unchanged** from QC.

---

## Files

### Added
| File | Purpose |
|---|---|
| `utils/transformer.py` | `CausalPrefixCritic` (causal Transformer emitting all `H` prefix Q-values; per-position output heads, scalar or `num_atoms`-way categorical) + `PrefixValue` (encoder + K-ensemble wrapper). |
| `utils/distributional.py` | HL-Gauss transform + categorical cross-entropy and the value-support estimators (`data` / `universal`) for the distributional critic. |
| `agents/aqc.py` | `AQCAgent`: per-horizon multi-step TD loss (regression MSE or distributional CE), expected-prefix-max target, flow-BC actor loss, adaptive `(n*, h*)` extraction, `get_config()`. |
| `scripts/aqc/*.sh`, `scripts/aqc/sweep/` | Example launch commands + the `cfg.json -> manifest -> array` sweep launcher. |
| `AQC.md` | This document. |

### Modified (backward-compatible — QC behaviour is unchanged)
| File | Change |
|---|---|
| `agents/__init__.py` | Register `aqc=AQCAgent`. |
| `evaluation.py` | Rollout handles an adaptive `(chunk, h*)` return and logs the executed chunk-size distribution (`chunk_len_mean/min/max`). Falls back to the old behaviour for fixed-chunk agents. |
| `main.py` | Online rollout handles the adaptive `(chunk, h*)` return. |

No dataset change was needed: `Dataset.sample_sequence` already returns per-step
cumulative discounted rewards, all intermediate next-observations, masks, and a per-step
`valid` flag — exactly the quantities the per-horizon targets require.

---

## How to train

Run everything from this `qc/` directory. `--horizon_length` is the **maximum** chunk
size `H`; `--agent.num_action_samples` is the rejection-sampling size `N`.

### Offline + offline-to-online (paper protocol: 1M offline + 1M online)
```bash
MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --run_group=aqc --env_name=cube-double-play-singletask-task1-v0 \
  --horizon_length=5 --agent.num_action_samples=4 \
  --offline_steps=1000000 --online_steps=1000000 \
  --eval_interval=100000 --eval_episodes=50
```
or simply: `bash scripts/aqc/offline_to_online.sh cube-double 0`

### Sparse domains (scene / puzzle) — add `--sparse=True`
```bash
MUJOCO_GL=egl python main.py --agent agents/aqc.py \
  --env_name=scene-play-singletask-task1-v0 --sparse=True \
  --horizon_length=5 --agent.num_action_samples=4
```

### Offline only
```bash
bash scripts/aqc/offline.sh cube-double-play-singletask-task1-v0 0 False
```

### Distributional critic (HL-Gauss)
Swap the scalar regression critic for a categorical (HL-Gauss) one with `--agent.critic_type`.
The value support (bin range) is set by `--support_type`: `data` (p1/p99 of dataset
return-to-go + margin), `universal` (`r/(1-γ)` bounds), or `custom` (`--agent.v_min/v_max`).
```bash
MUJOCO_GL=egl python main.py --agent agents/aqc.py \
  --env_name=cube-double-play-singletask-task1-v0 \
  --horizon_length=5 --agent.num_action_samples=4 \
  --agent.critic_type=distributional --agent.num_atoms=101 --support_type=data
```

### Quick smoke test (real env, a few thousand steps)
```bash
bash scripts/aqc/smoke.sh cube-double-play-singletask-task1-v0
```

### cube-quadruple (streamed 100M dataset, `N=8`)
```bash
bash scripts/aqc/cube_quadruple_100m.sh /path/to/quadruple_npz_dir
```

### Per-task hyperparameters (paper Table 3)
| Domain | `--horizon_length` (H) | `--agent.num_action_samples` (N) | `--sparse` |
|---|---|---|---|
| scene-sparse | 5 | 4 | True |
| puzzle-3x3-sparse | 5 | 4 | True |
| cube-double | 5 | 4 | False |
| cube-triple | 5 | 4 | False |
| cube-quadruple-100M | 5 | 8 | False |

---

## How to evaluate

Evaluation runs **automatically** inside `main.py` every `--eval_interval` steps and at the
end of each phase (success rate over `--eval_episodes` episodes, reported to W&B + CSV).
At deployment the policy is greedy-adaptive: it samples `N` chunks, scores all prefixes,
executes the arg-max prefix `a^{(n*)}_{1:h*}`, then replans. The evaluator additionally logs
`eval/chunk_len_mean` (and min/max), i.e. the **executed chunk-size distribution** — the
analysis in Figure 3 of the paper falls out for free.

To reproduce the paper's exact numbers:
* **Offline protocol** — average success rate over the last three eval epochs (800K/900K/1M).
* **Offline-to-online protocol** — success rate at 1M (end of offline) and 2M (end of online).

---

## Key configuration flags (`agents/aqc.py:get_config()`)

| Flag | Default | Meaning |
|---|---|---|
| `--horizon_length` (CLI) | 5 | Maximum chunk size `H`. |
| `--agent.num_action_samples` | 4 | Rejection-sampling size `N`. |
| `--agent.adaptive_chunking` | `True` | Joint `(n, h)` arg-max. Set `False` for a fixed-`H` Transformer-critic control. |
| `--agent.use_target_critic` | `False` | Paper default = online critic with stopped gradient. Set `True` for a Polyak target (`τ=5e-3`) if you observe value divergence. |
| `--agent.q_agg` | `min` | Ensemble aggregation (ACSAC uses `min`). |
| `--agent.num_qs` | 2 | Critic ensemble size `K`. |
| `--agent.per_position_head` | `True` | One output head per prefix position (paper, Prop. G.7). `False` = a single tied head shared across positions. |
| `--agent.transformer_num_layers / _num_heads / _head_dim` | 2 / 8 / 16 | Causal Transformer (`n_embd = heads × head_dim = 128`). |
| `--agent.flow_steps` | 10 | Euler steps `F` for the flow-BC policy. |
| `--agent.critic_type` | `regression` | `regression` (scalar MSE) or `distributional` (HL-Gauss categorical CE). |
| `--agent.num_atoms` | 101 | Categorical atoms (distributional only). |
| `--agent.hl_gauss_sigma` | 0.75 | HL-Gauss label smoothing, in bin-width units (distributional only). |
| `--support_type` (CLI) | `data` | Value-support range for the distributional critic: `data` / `universal` / `custom` (the last uses `--agent.v_min/v_max`). |

### Switching between QC and AQC
* **QC** (fixed chunk, rejection sampling): `python main.py --agent agents/acfql.py --agent.actor_type=best-of-n --agent.actor_num_samples=32 --horizon_length=5 …`
* **AQC** (adaptive chunk): `python main.py --agent agents/aqc.py --horizon_length=5 --agent.num_action_samples=4 …`

Both share the same env flags, dataset handling, and logging, so swapping is a one-line change.

---

## Environment & rendering
* Reference conda env with the full stack (jax, flax, distrax, ogbench, mujoco, wandb):
  `deas_real`. Datasets auto-download on first use (specify the task name; no manual setup).
  The provided scripts activate it automatically (override with `CONDA_ENV=...`).
* **GPU fix (`unset LD_LIBRARY_PATH`):** `deas_real`'s JAX ships its own CUDA wheels, but the
  cluster's default-loaded `cuda/12.8` module puts a conflicting CUDA on `LD_LIBRARY_PATH`,
  which makes the JAX plugin fail (`cuSPARSE library was not found`) and **silently fall back
  to CPU**. All scripts `unset LD_LIBRARY_PATH` after activating the env so JAX uses the
  bundled wheels and sees the GPU. Verify with `python -c "import jax; print(jax.devices())"`
  → should print `[CudaDevice(0)]`.
* On a node **with a GPU/display**, use `MUJOCO_GL=egl` (the scripts' default). On a
  **headless / CPU-only** node use `MUJOCO_GL=osmesa` (software rendering) — EGL needs a
  display and will raise `Cannot initialize a headless EGL display` otherwise.

## SLURM
`scripts/aqc/sbatch_aqc.sh` is an array-job launcher mirroring the project's working
`sbatch_acfql.sh` (partitions `big_suma_rtx3090,suma_rtx4090,base_suma_rtx3090,suma_a6000`,
`--qos=big_qos`, the broken-node exclude list, `deas_real`, `unset LD_LIBRARY_PATH`,
`MUJOCO_GL=egl`). It maps each array task to one `(env, seed)` and auto-streams the large
cube-triple / cube-quadruple datasets (and bumps `N` to 8 for cube-quadruple, per Table 3):
```bash
# 5 cube-double tasks x 1 seed:
sbatch --array=0-4 scripts/aqc/sbatch_aqc.sh aqc aqc_cube_double \
  "cube-double-play-singletask-task1-v0 cube-double-play-singletask-task2-v0 \
   cube-double-play-singletask-task3-v0 cube-double-play-singletask-task4-v0 \
   cube-double-play-singletask-task5-v0" "0" 4 False
```
Verified on an RTX 3090 (`srun … unset LD_LIBRARY_PATH …`): env + real-dataset training +
adaptive eval all run; after the one-time JIT compile, updates are ~15 ms each.

## Notes & caveats
* End-to-end validated on **real OGBench** (cube-quadruple shard): env creation, real
  chunked-batch training, and a real adaptive-rollout eval episode (executed chunk sizes
  1–5) all run. Validated on **state-based** OGBench manipulation. Pixel observations are
  wired (the encoder output becomes the state token) but untested — treat as future work.
* The bootstrap evaluates `N` proposals at each of the `H` next-states (faithful to Eq. 6);
  this is the dominant per-step cost and is on par with QC's `N=32` sampling.
* Reference dependency versions are pinned in `requirements.txt` (jax 0.6.0, flax 0.10.5).
  The Transformer uses only stable Flax APIs (`MultiHeadDotProductAttention`,
  `make_causal_mask`) and has been verified on flax 0.10.x–0.12.x.
