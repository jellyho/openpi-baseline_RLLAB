"""Build a tiny seal_v3 subset for fast critic iteration: 10 success + 10 failure +
10 intervention episodes, then push to HF.
  success      = autonomous (inference-only) episode ending in success (reward 0.0)
  failure      = autonomous (inference-only) episode ending in failure (reward<=-0.05)
  intervention = episode with BOTH inference & teleop frames (human took over); 5 succ + 5 fail
Episodes are re-indexed 0..29 (category order) and written one-episode-per-row-group.
Mirrors the source layout (data/chunk-000/file-000.parquet; source has no meta/).
"""
import glob, os, json, collections
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

SRC = "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/seal-water-bottle-cap_annotated_v3"
OUT = "/lustre/jellyho/seal_mini"
REPO = "jellyho/seal-water-bottle-cap_mini30"
files = sorted(glob.glob(SRC + "/data/chunk-*/file-*.parquet"))
def leaf(pf, n):
    rg = pf.metadata.row_group(0)
    return [i for i in range(rg.num_columns) if rg.column(i).path_in_schema == n][0]

# ---- 1. categorize every episode (scalars only; fast) ----
rmin = collections.defaultdict(lambda: 1e9); has_inf = collections.defaultdict(bool); has_tel = collections.defaultdict(bool)
for f in files:
    t = pq.read_table(f, columns=["episode_index", "reward", "observation.commander_state"])
    ep = np.asarray(t["episode_index"].to_pylist()); rw = np.asarray(t["reward"].to_pylist(), float)
    cs = t["observation.commander_state"].to_pylist()
    for e, r, c in zip(ep, rw, cs):
        e = int(e); rmin[e] = min(rmin[e], r)
        c = str(c).lower()
        if c == "inference": has_inf[e] = True
        if c == "teleop":    has_tel[e] = True
eps = sorted(rmin)
succ_inf = [e for e in eps if rmin[e] > -0.05 and has_inf[e] and not has_tel[e]]
fail_inf = [e for e in eps if rmin[e] <= -0.05 and has_inf[e] and not has_tel[e]]
intv_s   = [e for e in eps if has_inf[e] and has_tel[e] and rmin[e] > -0.05]
intv_f   = [e for e in eps if has_inf[e] and has_tel[e] and rmin[e] <= -0.05]
rng = np.random.default_rng(0)
def pick(pool, k): return sorted(int(x) for x in rng.choice(pool, k, replace=False))
sel_succ, sel_fail = pick(succ_inf, 10), pick(fail_inf, 10)
sel_is, sel_if = pick(intv_s, 5), pick(intv_f, 5)
selected = sel_succ + sel_fail + sel_is + sel_if
cats     = ["success"] * 10 + ["failure"] * 10 + ["intervention"] * 5 + ["intervention"] * 5
print("selected episodes:", selected)
print("pools:", dict(succ_inf=len(succ_inf), fail_inf=len(fail_inf), intv_s=len(intv_s), intv_f=len(intv_f)))

# ---- 2. map each selected episode -> its (file, row-groups) via stats ----
ep2rg = collections.defaultdict(list)
for f in files:
    pf = pq.ParquetFile(f); ei = leaf(pf, "episode_index")
    for g in range(pf.metadata.num_row_groups):
        s = pf.metadata.row_group(g).column(ei).statistics
        if s and s.has_min_max:
            for e in selected:
                if s.min <= e <= s.max: ep2rg[e].append((f, g))

# ---- 3. extract, re-index, write one-episode-per-row-group ----
os.makedirs(OUT + "/data/chunk-000", exist_ok=True)
out_path = OUT + "/data/chunk-000/file-000.parquet"
writer = None; gidx = 0; mapping = []
ep_t = None; ix_t = None
for new_ep, (e, cat) in enumerate(zip(selected, cats)):
    byf = collections.defaultdict(list)
    for (f, g) in ep2rg[e]: byf[f].append(g)
    parts = [pq.ParquetFile(f).read_row_groups(sorted(set(gs))) for f, gs in byf.items()]
    t = pa.concat_tables(parts)
    epcol = np.asarray(t["episode_index"].to_pylist())
    t = t.filter(pa.array(epcol == e))
    fi = np.asarray(t["frame_index"].to_pylist())
    t = t.take(pa.array(np.argsort(fi, kind="stable")))
    n = t.num_rows
    if ep_t is None:
        ep_t = t.schema.field("episode_index").type; ix_t = t.schema.field("index").type
    t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                     pa.array(np.full(n, new_ep), type=ep_t))
    t = t.set_column(t.schema.get_field_index("index"), "index",
                     pa.array(np.arange(gidx, gidx + n), type=ix_t)); gidx += n
    if writer is None:
        writer = pq.ParquetWriter(out_path, t.schema)
    writer.write_table(t, row_group_size=10 ** 7)   # one row-group per episode
    mapping.append({"new_ep": new_ep, "src_ep": int(e), "category": cat, "frames": int(n)})
    print(f"  ep {new_ep:2d} <- src {e:4d} [{cat:12s}] {n:5d} frames")
writer.close()
json.dump({"task": "seal-water-bottle-cap", "source": SRC, "n_frames": gidx, "episodes": mapping},
          open(OUT + "/categories.json", "w"), indent=2)
sz = os.path.getsize(out_path) / 1e9
print(f"wrote {out_path}  ({sz:.2f} GB, {gidx} frames, 30 episodes)")

# ---- 4. push to HF (private) -- disabled: run .diag/push_mini.py separately ----
print(f"LOCAL BUILD DONE. To push: REPO={REPO}  (see .diag/push_mini.py)")
