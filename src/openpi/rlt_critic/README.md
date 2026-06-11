# `openpi.rlt_critic` — RLT / Adaptive-Q-Chunking critic

A lightweight **ACSAC prefix-critic** trained on the *precomputed* RLT latents of a frozen
VLA, absorbed into openpi (no dependency on the standalone `adaptive_q_chunking/` tree).

It learns `Q(z_rl, a_{t:t+h})` for every prefix length `h` from the annotated LeRobot
dataset columns — `rl_token` (2048-d frozen-VLA latent) and `base_action` (N=32 candidate
action chunks) — so training runs **without any VLA forward pass** and is fast/cheap. At
deployment the critic scores the actor's N samples and selects the chunk + horizon to
execute (adaptive Q-chunking).

This is the AQC line (small transformer on latents), *not* `pi0_alphaflow_critic.Pi0WithCritic`
(a 3rd Gemma expert that needs the full VLA forward).

## Design — MC-warmup → `max(MC, Q)` via a ReLU-blend

Offline RL on long-horizon robot data; the target is `max(MC return, Q-backup)`. Pure
bootstrapping propagates slowly, so the critic **warms up on the Monte-Carlo return** and
then *gradually* trusts the bootstrap, via a single interpolated target:

```
y = G_MC + β · ReLU(  r + γ·Q̄(s', a') − G_MC )      # agent.critic_loss_td
```

- `β = 0` → pure MC warmup (regress every prefix to the realized return-to-go; grounds the
  whole success manifold with no bootstrap, and suppresses early Q-backup overestimation).
- `β = 1` → `max(G_MC, Q-backup)` — the Cal-QL-floored target the algorithm ultimately wants.
- MC stays a **hard lower-bound floor**; `β` only scales the *improvement* term above it
  (the slow/risky signal). `β` is ramped `0 → 1` (cosine) by `config.mc_blend_beta(step)`:
  `mc_warmup_steps` of pure MC, then `mc_ramp_steps` of ramp, then `β = 1` forever.

`mc_floor=False` disables the floor (pure TD); `mc_warmup=mc_ramp=0` gives `β=1` from step 0
(the old hard-max floor). See ablation presets in `config.py`.

## Train (Stage 1 — implemented)

```bash
# detached launch (survives the shell); see the script header for knobs
scripts/train_rlt_critic.sh vla_aqc_warmup 64 3        # CONFIG BATCH GPU

# or directly
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.038 \
  .venv/bin/python scripts/train_rlt_critic.py --config vla_aqc_warmup --batch_size 64 --loader_processes 4
```

- Default critic: `n_embd=384 / 3 layers / K=2`, HL-Gauss 201 atoms over `[-0.5, 0]`,
  macro-grouping 10 (replan at 10/20/30/40/50 steps) → **~10.75M params**.
- Data: `config.TASKS[task]` (default the local mouse-battery dataset, override `--data_root`).
  mouse-battery is the v1/v2 annotation: `reward` living=-1e-4 / fail-terminal=-0.5,
  precomputed `mc_return` at γ=0.995 in `[-0.5,0]` → support `[-0.5,0]`, `td.discount=0.995`.
- Checkpoints (msgpack params+opt_state) + `metrics.csv` under `config.checkpoint_base_dir`.
  Resume: add `--resume`.
- Presets: `vla_aqc_warmup` (primary), `vla_mc` (pure-MC warmup baseline),
  `vla_aqc_hardmax` / `vla_aqc_no_floor` / `vla_aqc_warmup_softmax` (transition ablations),
  `*_small` / `*_large` / `*_stateenc` (capacity).

### Memory note
The bootstrap forward (over `B·P·N` candidate sequences) is the memory driver. On a busy GPU
with ~7GB free, `B=64` fits (~6.9GB); `B≥128` needs a freer GPU. `PREALLOCATE=false` keeps a
cap-OOM contained to this job.

## Files
- `config.py`   — `VLAAQCConfig` (frozen) + preset registry + `mc_blend_beta` schedule.
- `agent.py`    — `VLACriticTrainer`: MC / ReLU-blend-TD losses, expected-prefix-max bootstrap.
- `transformer.py` — `PrefixValue` causal critic (macro-grouped, per-position heads, ensemble).
- `data.py`     — `VLALeRobotDataset`: row-group-streaming loader over the annotated parquet.
- `loader.py`   — torch multiprocess batch loader (openpi pattern; stays JAX-free in workers).
- `networks.py` / `distributional.py` — `default_init`/`ensemblize`, HL-Gauss transforms.
- `scripts/train_rlt_critic.py` (+ `.sh`) — training entry / detached launcher.

## Roadmap (not yet implemented)
- **Stage 2 — merge.** Fold the trained critic into the `Pi0RLT` / `Pi0RLTJoint` checkpoint
  as one deployable model (wrap `PrefixValue` as an `nnx` submodule via `nnx_bridge.ToNNX`,
  co-locate params in one orbax `params/` tree; `weight_loaders.AlphaFlowWeightLoader` pattern).
- **Stage 3 — adaptive inference.** In the openpi `Policy`: actor samples N chunks →
  critic scores all prefixes → joint arg-max `(n*, h*)` → execute `h*`. Two execution modes:
  (a) truncate the chunk to `h*`; (b) absolute-joint: run to `h*` then hold the last action.
