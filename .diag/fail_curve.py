"""Load the b256 critic checkpoint and, on a FAILURE trajectory, compute three curves:
  V_demo  = critic full-prefix value of the demo chunk (ensemble-mean) -- what eval plots
  mc      = mc_return (the Cal-QL floor)
  target  = max(mc, TD_full)  with TD_full = cum_reward + gamma^H * v_next  (beta=1)
            v_next = _expected_prefix_max(next state) (or next_mc at terminal)
Tells us: is the TARGET flat (bootstrap problem) or sloped-but-V-underfits (optimization)?
"""
import sys, pathlib, glob
sys.path.insert(0, "scripts")
import numpy as np, jax, jax.numpy as jnp
import pyarrow.parquet as pq, pyarrow as pa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from openpi.rlt_critic.config import get_config
from openpi.rlt_critic.agent import VLACriticTrainer
from openpi.rlt_critic.data import _list_col_to_numpy, LATENT_DIM, ACTION_DIM, BASE_ACTION_SHAPE
from train_rlt_critic import load_checkpoint, list_checkpoints

RUN = pathlib.Path("/lustre/jellyho/PFR_RSS/checkpoints/rlt_critic_runs/vla_aqc_warmup/"
                   "seal-water-bottle-cap_a201_sup-fixed_emb384x3L_N32_P5_b256_g0.9999_warm20k+30k_s0")
EP = 491
cfg = get_config("vla_aqc_warmup")
H, Dr, gamma = cfg.horizon, cfg.action_dim, cfg.td.discount
trainer = VLACriticTrainer(cfg, seed=0)
step = list_checkpoints(RUN / "checkpoints")[-1]
params, _ = load_checkpoint(RUN / "checkpoints", step, trainer.params, trainer.opt_state)
print(f"loaded critic step {step}")

# --- load the failure episode (rl_token/action/base_action/mc/reward) ---
files = sorted(glob.glob("/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/"
                         "seal-water-bottle-cap_annotated_v3/data/chunk-*/file-*.parquet"))
def leaf(pf, n):
    rg = pf.metadata.row_group(0)
    return [i for i in range(rg.num_columns) if rg.column(i).path_in_schema == n][0]
cols = ["episode_index", "frame_index", "reward", "mc_return", "action", "rl_token", "base_action"]
tbl = None
for f in files:
    pf = pq.ParquetFile(f); ei = leaf(pf, "episode_index")
    for g in range(pf.metadata.num_row_groups):
        s = pf.metadata.row_group(g).column(ei).statistics
        if s and s.has_min_max and s.min <= EP <= s.max:
            lo, hi = max(0, g - 16), min(pf.metadata.num_row_groups, g + 16)
            tbl = pa.concat_tables([pf.read_row_group(gg, columns=cols) for gg in range(lo, hi)]); break
    if tbl is not None: break
ep = np.asarray(tbl["episode_index"].to_pylist()); fi = np.asarray(tbl["frame_index"].to_pylist())
m = ep == EP; o = np.argsort(fi[m])
rew = np.asarray(tbl["reward"].to_pylist(), np.float32)[m][o]
mc  = np.asarray(tbl["mc_return"].to_pylist(), np.float32)[m][o]
act = _list_col_to_numpy(tbl["action"], (ACTION_DIM,))[m][o]                 # (n,14)
z   = _list_col_to_numpy(tbl["rl_token"], (LATENT_DIM,)).astype(np.float32)[m][o]   # (n,2048)
ba  = _list_col_to_numpy(tbl["base_action"], BASE_ACTION_SHAPE, dtype=np.float16)[m][o]  # (n,32,50,14)
n = len(mc); S = n - H
print(f"episode {EP}: {n} frames, S={S} starts, reward[end]={rew[-1]:.3f}, mc[start..end]={mc[0]:.3f}..{mc[-1]:.3f}")

starts = np.arange(S)
# V_demo: critic full-prefix value (ensemble-mean) of demo chunk
def critic_q(zz, chunks):  # zz (M,L), chunks (M,H*Dr) -> full-prefix q, mean & min over K
    logits = trainer.net.apply(params, jnp.asarray(zz), jnp.asarray(chunks))   # (K,M,macroH,atoms)
    q = trainer.from_probs(jax.nn.softmax(logits, -1))[..., -1]                 # (K,M) full prefix
    return np.asarray(q.mean(0)), np.asarray(q.min(0))
Vmean = np.empty(S, np.float32); Vmin = np.empty(S, np.float32)
for s0 in range(0, S, 256):
    s1 = min(s0 + 256, S); idx = starts[s0:s1]
    ch = np.stack([act[i:i + H] for i in idx]).reshape(len(idx), H * Dr)
    Vmean[s0:s1], Vmin[s0:s1] = critic_q(z[idx], ch)

# v_next = _expected_prefix_max(next state s_{i+H}); cum_reward; TD_full; target
nxt = starts + H                                                               # H..n-1
vnext = np.empty(S, np.float32)
for s0 in range(0, S, 64):
    s1 = min(s0 + 64, S); j = nxt[s0:s1]
    cand = jnp.asarray(ba[j].reshape(len(j), ba.shape[1], -1))                 # (m,32,700)
    vnext[s0:s1] = np.asarray(trainer._expected_prefix_max(params, jnp.asarray(z[j]), cand))
is_term = (rew >= -1e-6) | (rew <= -0.05)
term = is_term[nxt]
vnext_f = np.where(term, mc[nxt], vnext)                                       # terminal uses mc
cum = np.zeros(S, np.float32)
for jstep in range(H):
    cum += (gamma ** jstep) * rew[starts + jstep]
td_full = cum + (gamma ** H) * vnext_f
target = np.maximum(mc[starts], td_full)                                       # beta=1

print(f"V_demo(mean): [{Vmean.min():.3f},{Vmean.max():.3f}]  mc:[{mc[:S].min():.3f},{mc[:S].max():.3f}]  "
      f"target:[{target.min():.3f},{target.max():.3f}]  td_full:[{td_full.min():.3f},{td_full.max():.3f}]")
print(f"mean |V_demo - mc| = {np.mean(np.abs(Vmean - mc[:S])):.3f}   mean(V_demo - mc) = {np.mean(Vmean - mc[:S]):.3f}")
print(f"term_frac in this episode's windows = {term.mean():.4f}")
ix = np.linspace(0, S - 1, 8).astype(int)
print("  t      mc     V_demo  target  td_full  vnext")
for i in ix:
    print(f"  {i:5d}  {mc[i]:6.3f}  {Vmean[i]:6.3f}  {target[i]:6.3f}  {td_full[i]:6.3f}  {vnext[i]:6.3f}")

x = starts
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, mc[:S], color="#1c2330", lw=2.2, label="mc_return (floor)")
ax.plot(x, target, color="#e67e22", lw=1.6, ls="--", label="TD target = max(mc, TD_full)")
ax.plot(x, Vmean, color="#b42318", lw=1.8, label="V_demo (critic, ensemble-mean)")
ax.plot(x, Vmin, color="#b42318", lw=1.0, alpha=0.5, label="V_demo (ensemble-min)")
ax.set_title(f"FAILURE ep {EP} | critic step {step} | b256")
ax.set_xlabel("frame t"); ax.set_ylabel("value"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
fig.tight_layout(); out = ".diag/fail_curve.png"; fig.savefig(out, dpi=120)
print("saved", out)
