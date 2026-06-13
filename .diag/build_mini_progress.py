"""Exp2 data (PAPER-style progress reward, TD-ready): seal_mini ->seal_mini_progress.

Per trajectory (one episode per row-group):
  mc_return = GAMMA^(T-t) * I(success)         # progress target in [0,1]
  reward    = 0 everywhere except the TERMINAL: +1 (success) / -0.5 (failure)   # outcome marker:
              keeps the reward-based eval categorisation working AND, with the mc_floor, the TD
              target stays consistent (failure target floored to 0 = progress).
  done      = explicit terminal signal (1 at the last frame) -> data.py reads THIS for is_term,
              so living=0 doesn't trip the reward heuristic.

Hindsight (Sec 3.1): for every RECOVERED intervention (policy failed -> teleop -> success) ALSO
emit a NEW truncated FAILURE = frames before the first teleop, progress 0, terminal reward -0.5,
done at the cut, new episode_index. (The original success episode is kept too -> paired data.)
Env: GAMMA (0.999). Use with config target_kind='td', dist support [0,1].
"""
import os, numpy as np, pyarrow as pa, pyarrow.parquet as pq

GAMMA = float(os.environ.get("GAMMA", "0.999")); MIN_TRUNC = 60
SRC = "/lustre/jellyho/seal_mini/data/chunk-000/file-000.parquet"
OUT = "/lustre/jellyho/seal_mini_progress"; os.makedirs(OUT + "/data/chunk-000", exist_ok=True)
pf = pq.ParquetFile(SRC); sch = pf.schema_arrow
MC, RW, EP = (sch.get_field_index(c) for c in ("mc_return", "reward", "episode_index"))
MCt, RWt, EPt = (sch.field(c).type for c in ("mc_return", "reward", "episode_index"))

def build(raw, prog, rew, done, ep=None):
    t = raw.set_column(MC, "mc_return", pa.array(prog, MCt)).set_column(RW, "reward", pa.array(rew, RWt))
    if ep is not None:
        t = t.set_column(EP, "episode_index", pa.array(np.full(len(prog), ep), EPt))
    return t.append_column("done", pa.array(done, pa.int8()))

writer = None; new_ep = pf.metadata.num_row_groups; n_trunc = 0
for g in range(pf.metadata.num_row_groups):
    raw = pf.read_row_group(g)
    fi = np.asarray(raw["frame_index"].to_pylist()); rew0 = np.asarray(raw["reward"].to_pylist(), float)
    cs = np.asarray([str(c).lower() for c in raw["observation.commander_state"].to_pylist()])
    is_fail = bool((rew0 <= -0.05).any()); tel = np.where(cs == "teleop")[0]; T = fi.max(); n = len(fi)
    I = 0.0 if is_fail else 1.0
    prog = (GAMMA ** (T - fi)).astype(np.float32) * np.float32(I)
    rew = np.zeros(n, np.float32); rew[-1] = np.float32(I)   # terminal reward 1(success)/0(failure) -> clean [0,1]
    done = np.zeros(n, np.int8); done[-1] = 1
    t = build(raw, prog, rew, done)
    if writer is None: writer = pq.ParquetWriter(OUT + "/data/chunk-000/file-000.parquet", t.schema)
    writer.write_table(t, row_group_size=10 ** 7)
    ep0 = int(np.asarray(raw["episode_index"].to_pylist())[0])
    print(f"  ep {ep0:2d} [{'fail' if is_fail else 'succ'}] T={T:5d} prog[{prog.min():.2f}..{prog.max():.2f}] tel={len(tel)}", end="")
    if (not is_fail) and len(tel) > 0 and int(tel.min()) > MIN_TRUNC:
        m = int(tel.min()); sub = raw.slice(0, m)
        sub = build(sub, np.zeros(m, np.float32), np.zeros(m, np.float32),
                    np.r_[np.zeros(m - 1, np.int8), np.int8(1)], ep=new_ep)   # failure: reward 0, done at cut
        writer.write_table(sub, row_group_size=10 ** 7)
        print(f"  -> +trunc-fail ep{new_ep} ({m} frames)", end=""); new_ep += 1; n_trunc += 1
    print()
writer.close()
print(f"wrote {OUT}  (30 originals + {n_trunc} truncated hindsight failures; with done column)")
