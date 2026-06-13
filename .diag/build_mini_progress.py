"""Exp2 data: seal_mini with mc_return REPLACED by the paper-style PROGRESS target.

    progress(t) = GAMMA^(T-t)   on the SUCCESSFUL portion, else 0

Per category (T = episode last frame):
  success (autonomous)     -> GAMMA^(T-t) rising to 1 over the whole trajectory.
  failure (autonomous)     -> 0 everywhere (policy failed; no positive signal).
  intervention, recovered  -> HINDSIGHT: 0 for the PRE-intervention frames (the policy was
                              failing, t < first teleop), then GAMMA^(T-t) rising once the human
                              takes over and drives to success. => value 0 in the policy-failing
                              region, rising after the intervention point.
  intervention, still failed -> 0 everywhere.
All other columns copied unchanged. One episode per row-group. Env: GAMMA (0.999), HINDSIGHT (1).
"""
import os, numpy as np, pyarrow as pa, pyarrow.parquet as pq

GAMMA = float(os.environ.get("GAMMA", "0.999"))
HINDSIGHT = os.environ.get("HINDSIGHT", "1") == "1"
SRC = "/lustre/jellyho/seal_mini/data/chunk-000/file-000.parquet"
OUT = "/lustre/jellyho/seal_mini_progress"
os.makedirs(OUT + "/data/chunk-000", exist_ok=True)
pf = pq.ParquetFile(SRC)
mc_idx = pf.schema_arrow.get_field_index("mc_return")
mc_t = pf.schema_arrow.field("mc_return").type
writer = None
print(f"GAMMA={GAMMA}  HINDSIGHT={HINDSIGHT}  row_groups={pf.metadata.num_row_groups}")
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g)                              # one full episode (sorted by frame_index)
    fi = np.asarray(t["frame_index"].to_pylist())
    rew = np.asarray(t["reward"].to_pylist(), float)
    cs = np.asarray([str(c).lower() for c in t["observation.commander_state"].to_pylist()])
    is_fail = bool((rew <= -0.05).any())
    tel = np.where(cs == "teleop")[0]
    T = fi.max()
    prog = (GAMMA ** (T - fi)).astype(np.float32)          # success rise (default)
    kind = "succ"
    if is_fail:
        prog[:] = 0.0; kind = "fail"                       # autonomous/post-intervention failure
    elif HINDSIGHT and len(tel) > 0:
        m = int(tel.min()); prog[:m] = 0.0; kind = f"interv(cut@{m})"   # 0 before human takeover
    ep = int(np.asarray(t["episode_index"].to_pylist())[0])
    t = t.set_column(mc_idx, "mc_return", pa.array(prog, type=mc_t))
    if writer is None:
        writer = pq.ParquetWriter(OUT + "/data/chunk-000/file-000.parquet", t.schema)
    writer.write_table(t, row_group_size=10 ** 7)
    print(f"  ep {ep:2d} [{kind:14s}] T={T:5d}  progress[{prog.min():.2f}..{prog.max():.2f}]  teleop={len(tel)}")
writer.close()
print(f"wrote {OUT}/data/chunk-000/file-000.parquet")
