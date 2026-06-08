# Adaptive Q-Chunking (ACSAC)

Implementation of ACSAC ([`acsac.pdf`](acsac.pdf)) on top of the QC codebase ([`qc/`](qc/)).
A causal-Transformer critic scores every action prefix `Q(s, a_{1:h})` for `h = 1…H`, and
the executed chunk size is chosen per state. A distributional (HL-Gauss) critic is available
via `--agent.critic_type=distributional`.

Detailed flag reference and per-task hyperparameters: [`qc/AQC.md`](qc/AQC.md).

## Setup

Python 3.11, deps pinned in [`qc/requirements.txt`](qc/requirements.txt) (includes the
CUDA-12 JAX wheels). OGBench datasets auto-download on first use.

conda:
```bash
conda create -n deas_real python=3.11 -y && conda activate deas_real
pip install -r qc/requirements.txt
```

uv:
```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r qc/requirements.txt
```

Two required runtime settings:
- `unset LD_LIBRARY_PATH` — a loaded system CUDA module shadows JAX's bundled wheels and
  forces a silent CPU fallback (`cuSPARSE library was not found`).
- `MUJOCO_GL=egl` on a GPU/display node, `MUJOCO_GL=osmesa` on a headless/CPU node.

Verify: `cd qc && unset LD_LIBRARY_PATH && python -c "import jax; print(jax.devices())"`
should print `[CudaDevice(0)]`.

## Run

From `qc/`:

```bash
# smoke test (a few thousand steps)
bash scripts/aqc/smoke.sh cube-double-play-singletask-task1-v0

# offline training (1M steps)
bash scripts/aqc/offline.sh cube-double-play-singletask-task1-v0 0 False   # <env> <seed> <sparse>
```

Raw command:
```bash
unset LD_LIBRARY_PATH
MUJOCO_GL=egl python main.py \
  --agent agents/aqc.py --env_name=cube-double-play-singletask-task1-v0 \
  --horizon_length=5 --agent.num_action_samples=4 \
  --offline_steps=1000000 --online_steps=0 \
  --eval_interval=100000 --eval_episodes=50
```

`--horizon_length` is the max chunk size `H`; `--agent.num_action_samples` is the
rejection-sampling size `N`. Add `--sparse=True` for scene/puzzle. Evaluation (success rate
+ executed chunk-size distribution) runs automatically every `--eval_interval` steps.

## SLURM

```bash
# one array task per (env, seed)
sbatch --array=0-4 scripts/aqc/sbatch_aqc.sh aqc aqc_cube_double \
  "cube-double-play-singletask-task1-v0 ... task5-v0" "0" 4 False

# grid sweep: cfg.json -> manifest -> array job (each task = its own W&B group)
cd scripts/aqc/sweep
python submit_aqc_sweep.py --cfg cfg_aqc_sweep.json --dry-run   # preview
python submit_aqc_sweep.py --cfg cfg_aqc_sweep.json --max-gpus 16
```

The sweep `cfg_*.json` files hold `fixed` / `sweep` (grid) / per-`envs` settings. All
launchers activate `deas_real`, `unset LD_LIBRARY_PATH`, and set `MUJOCO_GL=egl`.
