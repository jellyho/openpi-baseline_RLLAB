# Adaptive Q-Chunking (ACSAC)

A faithful implementation of **ACSAC — Adaptive Chunk Size Actor-Critic with Causal
Transformer Q-Network** ([`acsac.pdf`](acsac.pdf)), built on top of the **QC / Q-Chunking**
codebase ([`qc/`](qc/)). It learns a causal-Transformer critic that scores **every action
prefix** `Q(s, a_{1:h})` for `h = 1…H` in one pass, then lets the executed chunk size
become **state-dependent** (pick the prefix that maximizes the critic, execute it, replan).

An optional **distributional critic** (HL-Gauss categorical value) is included alongside the
default scalar-regression critic.

## Contents

| Path | What it is |
|---|---|
| [`qc/`](qc/) | The full codebase (agents, critic, training loop, eval, sweep infra). |
| [`qc/AQC.md`](qc/AQC.md) | **Practical quick-reference**: what AQC changes, every file, all flags, per-task hyperparameters. Start here. |
| [`implementation_plan.html`](implementation_plan.html) | Design rationale, paper summary, every documented design decision (DD-*). |
| [`aqc_reading_guide.html`](aqc_reading_guide.html) | Walkthrough of the method + the Transformer critic, with LaTeX equations. |
| [`aqc_distributional_guide.html`](aqc_distributional_guide.html) | The distributional (HL-Gauss) critic and value-support estimation, with equations. |
| [`acsac.pdf`](acsac.pdf) | The paper. |

The new code lives in `qc/utils/transformer.py` (`CausalPrefixCritic` + `PrefixValue`),
`qc/agents/aqc.py` (`AQCAgent`), `qc/utils/distributional.py` (HL-Gauss), and the
`qc/scripts/aqc/` launch + sweep scripts. QC's flow-BC policy, dataset chunking, replay
buffer, training loop, and evaluation harness are reused unchanged.

---

## Setup

Python **3.11**, JAX **0.6.0** / Flax **0.10.5**. Pinned deps are in
[`qc/requirements.txt`](qc/requirements.txt) (includes the CUDA-12 JAX plugin wheels, so it
is GPU-ready out of the box). OGBench datasets **auto-download** on first use — just name the
task, no manual download step.

### Option A — conda (reference env)

The reference env used for all runs is `deas_real` (already present on the cluster). To
recreate it elsewhere:

```bash
conda create -n deas_real python=3.11 -y
conda activate deas_real
pip install -r qc/requirements.txt
```

### Option B — uv

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r qc/requirements.txt
```

### Verify the install (GPU should be visible)

```bash
cd qc
unset LD_LIBRARY_PATH          # see "GPU caveat" below
python -c "import jax; print(jax.devices())"     # -> [CudaDevice(0)] on a GPU node
```

### Two environment gotchas (important)

- **GPU caveat — `unset LD_LIBRARY_PATH`.** The pinned JAX ships its own CUDA wheels. If a
  system `cuda/12.x` module is loaded, it puts a conflicting CUDA on `LD_LIBRARY_PATH`, the
  JAX plugin fails (`cuSPARSE library was not found`) and **silently falls back to CPU**.
  Always `unset LD_LIBRARY_PATH` after activating the env. (All provided scripts do this.)
- **Rendering — `MUJOCO_GL`.** On a node with a GPU/display use `MUJOCO_GL=egl` (scripts'
  default). On a **headless / CPU-only** node use `MUJOCO_GL=osmesa` (software rendering);
  EGL otherwise raises `Cannot initialize a headless EGL display`.

Optional: `export WANDB_ENTITY=... WANDB_PROJECT=...` to control logging (scripts default to
the project's W&B). Use `--offline` W&B by exporting `WANDB_MODE=offline`.

---

## Quick start

All commands run from the [`qc/`](qc/) directory.

### Smoke test (a few thousand steps, real env, ~minutes)

```bash
cd qc
bash scripts/aqc/smoke.sh cube-double-play-singletask-task1-v0
```

### Offline training (paper offline protocol: 1M steps)

```bash
cd qc
bash scripts/aqc/offline.sh cube-double-play-singletask-task1-v0 0 False
#                            <env>                                <seed> <sparse>
```

Or the raw command (regression critic, dense reward):

```bash
cd qc
unset LD_LIBRARY_PATH
MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py \
  --run_group=aqc --env_name=cube-double-play-singletask-task1-v0 \
  --horizon_length=5 --agent.num_action_samples=4 \
  --offline_steps=1000000 --online_steps=0 \
  --eval_interval=100000 --eval_episodes=50
```

`--horizon_length` is the **maximum** chunk size `H`; `--agent.num_action_samples` is the
rejection-sampling size `N`. Sparse domains (scene / puzzle) need `--sparse=True`.

### Distributional critic (HL-Gauss)

```bash
cd qc
unset LD_LIBRARY_PATH
MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py --env_name=cube-double-play-singletask-task1-v0 \
  --horizon_length=5 --agent.num_action_samples=4 \
  --agent.critic_type=distributional --agent.num_atoms=101 \
  --support_type=data \
  --offline_steps=1000000 --eval_interval=100000 --eval_episodes=50
```

`--support_type` sets the value-bin range: `data` (p1/p99 of dataset return-to-go + margin),
`universal` (`r/(1-γ)` bounds), or `custom` (`--agent.v_min/--agent.v_max`).

### Per-task hyperparameters (paper Table 3)

| Domain | `--horizon_length` (H) | `--agent.num_action_samples` (N) | `--sparse` |
|---|---|---|---|
| scene-sparse | 5 | 4 | True |
| puzzle-3x3-sparse | 5 | 4 | True |
| cube-double | 5 | 4 | False |
| cube-triple | 5 | 4 | False |
| cube-quadruple-100M | 5 | 8 | False |

Evaluation runs **automatically** inside `main.py` every `--eval_interval` steps (success
rate over `--eval_episodes`), and additionally logs the executed chunk-size distribution
(`eval/chunk_len_mean/min/max`).

See [`qc/AQC.md`](qc/AQC.md) for the full flag reference and the QC↔AQC swap.

---

## Running on SLURM

**Single launcher** (one array task per `(env, seed)`):

```bash
cd qc
sbatch --array=0-4 scripts/aqc/sbatch_aqc.sh aqc aqc_cube_double \
  "cube-double-play-singletask-task1-v0 cube-double-play-singletask-task2-v0 \
   cube-double-play-singletask-task3-v0 cube-double-play-singletask-task4-v0 \
   cube-double-play-singletask-task5-v0" "0" 4 False
```

**Grid sweep** (`cfg.json` → flat job manifest → one array job). Each config becomes a W&B
group with only the seed varying:

```bash
cd qc/scripts/aqc/sweep
python submit_aqc_sweep.py --cfg cfg_aqc_sweep.json --max-gpus 16
python submit_aqc_sweep.py --cfg cfg_aqc_sweep.json --dry-run   # preview the manifest first
```

Edit the `cfg_*.json` to change the grid. Each file has `fixed` (held constant), `sweep`
(grid dims: `critic_type`, `num_atoms`, `horizon_length`, `seed`, …), and per-`envs`
settings (`sparse`, `v_min`, `v_max`). The provided configs cover sparse
(`cfg_aqc_sweep.json`) and dense (`cfg_aqc_sweep_dense.json`) variants.

Both launchers activate `deas_real`, `unset LD_LIBRARY_PATH`, set `MUJOCO_GL=egl`, target the
project's RTX/A6000 partitions, and exclude the known-bad nodes. Override the env with
`CONDA_ENV=...`.
