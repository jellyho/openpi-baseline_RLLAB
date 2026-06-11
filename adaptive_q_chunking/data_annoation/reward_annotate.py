"""Reward re-annotation v3: raw rewards first, then GLOBAL normalization into (-1, 0].

Scheme (RECAP-style "set natural units, then normalize"):
  1. RAW rewards:   living step = -1,  success terminal = 0,
                    failure terminal = -C_fail with C_fail = C_FAIL_FRAC * T_max
                    (T_max = longest episode in THIS dataset; 0.4 * T_max by default).
  2. RAW mc_return: discounted return-to-go with gamma = 0.9999 (near-undiscounted, so the
                    failure penalty stays visible from the episode start: gamma^T >= ~0.2
                    for all episode lengths here, unlike gamma=0.999 where it vanished).
  3. NORMALIZE:     Z = |most negative raw return| over the WHOLE dataset (global, not
                    per-file).  reward /= Z and mc_return /= Z  ->  values in [-1, 0]
                    (exactly one state sits at -1: the start of the deepest-return episode).

Columns written:
  * unnormalized_reward  (NEW)  raw reward (-1 / 0 / -C_fail), kept for re-normalization later
  * reward                      normalized (= unnormalized_reward / Z)
  * mc_return                   normalized return-to-go (gamma=0.9999, / Z)

Because Z is global, the script runs in two phases:
  Phase A (single process): scan all files' scalars, classify frames (living/success/failure
           from the CURRENT disk values), find T_max -> C_fail, build raw rewards, compute
           per-file mc (episodes never span files: meta/episodes has one data/file_index per
           episode), find global Z, divide.
  Phase B (parallel, one worker per file): rewrite only the reward/mc_return columns,
           atomic temp+rename, original column compression preserved.

Idempotency: the new scheme's signature is min(mc_return) ~ -1.0 on disk (old schemes
bottom out at -0.5/-0.6).  If the dataset already looks normalized to -1, the script skips.

Usage:
    python reward_annotate.py --input <dataset_root> --inplace [--dry_run] [--workers 4]
    # run once per dataset (insert-mouse-battery_annotated, seal-water-bottle-cap_annotated)
"""

import argparse
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_ROOT  = "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/insert-mouse-battery_annotated"
OUTPUT_ROOT = INPUT_ROOT + "_v3"

DISCOUNT    = 0.9999   # near-undiscounted: value ~ normalized steps-to-go (+ failure penalty)
LIVING_RAW  = -1.0     # raw living cost per step
C_FAIL_FRAC = 0.4      # C_fail = this fraction of the LONGEST episode's step count

# Frame classification thresholds (work on every past encoding: raw -1e-4/-0.5,
# v1 relabel -4e-4/-0.5 -- failure is the only "large" negative, success is exactly 0).
_FAIL_THRESH = -0.05   # reward <= this  -> failure terminal
_SUCC_THRESH = -1e-6   # reward >= this  -> success terminal; in between -> living


# ---------------------------------------------------------------------------
# Phase A helpers
# ---------------------------------------------------------------------------
def scan_scalars(pf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All row-groups of one file -> (episode_index, reward, mc_return) flat arrays."""
    eps, rews, mcs = [], [], []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["episode_index", "reward", "mc_return"])
        eps.append(np.asarray(t["episode_index"].to_pylist(), dtype=np.int64))
        rews.append(np.asarray(t["reward"].to_pylist(), dtype=np.float64))
        mcs.append(np.asarray(t["mc_return"].to_pylist(), dtype=np.float64))
    return np.concatenate(eps), np.concatenate(rews), np.concatenate(mcs)


def raw_rewards(rew_disk: np.ndarray, c_fail: float) -> np.ndarray:
    """Classify each frame from its current disk value and assign the new RAW reward."""
    out = np.full(rew_disk.shape, LIVING_RAW, dtype=np.float64)        # living = -1
    out[rew_disk <= _FAIL_THRESH] = -c_fail                            # failure terminal
    out[rew_disk >= _SUCC_THRESH] = 0.0                                # success terminal
    return out


def compute_mc(rew: np.ndarray, ep: np.ndarray, gamma: float) -> np.ndarray:
    """Backward discounted return-to-go, resetting at episode boundaries."""
    n = len(rew)
    mc = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = rew[i] + gamma * running
        mc[i] = running
        if i == 0 or ep[i - 1] != ep[i]:
            running = 0.0
    return mc


# ---------------------------------------------------------------------------
# Phase B: per-file column rewrite (parallel worker)
# ---------------------------------------------------------------------------
def _column_compression(pf) -> dict:
    """Mirror the source file's per-column codec (big array cols are UNCOMPRESSED)."""
    rg = pf.metadata.row_group(0)
    comp = {}
    for i in range(rg.num_columns):
        c = rg.column(i)
        codec = c.compression.lower()
        comp[c.path_in_schema] = "none" if codec == "uncompressed" else codec
    return comp


def write_file(args_tuple):
    """Rewrite reward/mc_return (+ add unnormalized_reward) of one parquet (atomic)."""
    src, dst, raw_new, rew_new, mc_new, dry_run = args_tuple
    src, dst = Path(src), Path(dst)
    tag = src.name
    pf = pq.ParquetFile(src)
    n_groups, total_rows = pf.metadata.num_row_groups, pf.metadata.num_rows
    assert len(rew_new) == total_rows, f"{tag}: column length mismatch"
    mode = "in-place" if dst.resolve() == src.resolve() else "copy"
    print(f"[{tag}] {n_groups} row-groups, {total_rows:,} rows ({mode})", flush=True)
    if dry_run:
        return {}

    t0 = time.time()
    comp = _column_compression(pf)
    comp["unnormalized_reward"] = comp.get("reward", "snappy")   # scalar col, same codec
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer, off, last_print = None, 0, time.time()
    try:
        for g in range(n_groups):
            table = pf.read_row_group(g)
            n_g = table.num_rows
            ri = table.schema.get_field_index("reward")
            table = table.set_column(ri, "reward",
                                     pa.array(rew_new[off:off + n_g], type=pa.float32()))
            table = table.set_column(table.schema.get_field_index("mc_return"), "mc_return",
                                     pa.array(mc_new[off:off + n_g], type=pa.float32()))
            raw_col = pa.array(raw_new[off:off + n_g], type=pa.float32())
            if "unnormalized_reward" in table.schema.names:      # idempotent re-run
                table = table.set_column(table.schema.get_field_index("unnormalized_reward"),
                                         "unnormalized_reward", raw_col)
            else:                                                # insert right after reward
                table = table.add_column(ri + 1, pa.field("unnormalized_reward", pa.float32()),
                                         raw_col)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=comp)
            writer.write_table(table)
            off += n_g
            now = time.time()
            if now - last_print >= 15.0 or g + 1 == n_groups:
                pct = (g + 1) * 100 // n_groups
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                eta = (now - t0) * (n_groups - g - 1) / (g + 1)
                print(f"[{tag}] [{bar}] {pct:3d}%  ({off:,}/{total_rows:,} rows)  "
                      f"elapsed={now-t0:.0f}s eta={eta:.0f}s", flush=True)
                last_print = now
        if writer:
            writer.close(); writer = None
    except BaseException:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise
    os.replace(tmp, dst)
    print(f"[{tag}] done  {time.time()-t0:.0f}s -> {dst.name} ({mode})", flush=True)
    return {"rows": total_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Re-annotate reward/mc_return (raw -> global normalize)")
    ap.add_argument("--input",   default=INPUT_ROOT)
    ap.add_argument("--output",  default="", help="output root (default <input>_v2; ignored with --inplace)")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input files (atomic)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="phase A + design summary only, no writes")
    args = ap.parse_args()

    src_root = Path(args.input)
    dst_root = src_root if args.inplace else Path(args.output or (str(src_root) + "_v3"))
    files = sorted((src_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {src_root}/data")

    print(f"=== reward_annotate v3 (raw -> global normalize) ===")
    print(f"input    : {src_root}")
    print(f"output   : {dst_root}{'  (IN-PLACE)' if args.inplace else ''}")
    print(f"scheme   : living={LIVING_RAW}, C_fail={C_FAIL_FRAC}*T_max, gamma={DISCOUNT}, "
          f"then /Z (global most-negative return)")

    # ---------------- Phase A: global scan + compute ----------------
    t0 = time.time()
    scal = {}                                    # file -> (ep, rew_disk, mc_disk)
    for f in files:
        scal[f] = scan_scalars(pq.ParquetFile(f))
    print(f"phase A  : scalars loaded ({time.time()-t0:.0f}s)")

    # idempotency: new scheme normalizes the deepest return to exactly -1 on disk.
    disk_mc_min = min(float(m.min()) for (_, _, m) in scal.values())
    if disk_mc_min < -0.9:
        print(f"SKIP     : disk mc_return min = {disk_mc_min:.4f} -> already v3-normalized.")
        return

    # T_max over the whole dataset -> C_fail.
    ep_lens = {}
    for (ep, _, _) in scal.values():
        e, c = np.unique(ep, return_counts=True)
        for k, v in zip(e, c):
            ep_lens[int(k)] = ep_lens.get(int(k), 0) + int(v)
    lens = np.array(list(ep_lens.values()))
    t_max = int(lens.max())
    c_fail = C_FAIL_FRAC * t_max
    n_fail_eps = sum(int((r <= _FAIL_THRESH).sum()) for (_, r, _) in scal.values())

    # raw rewards + per-file raw mc; global Z.
    raw = {}                                     # file -> (rew_raw, mc_raw)
    z = 0.0
    for f, (ep, rew_disk, _) in scal.items():
        rr = raw_rewards(rew_disk, c_fail)
        mm = compute_mc(rr, ep, DISCOUNT)
        raw[f] = (rr, mm)
        z = max(z, float(-mm.min()))
    print(f"phase A  : episodes={len(ep_lens)} (fail={n_fail_eps})  "
          f"T_max={t_max} -> C_fail={c_fail:.0f}  Z={z:.1f}  ({time.time()-t0:.0f}s)")

    # normalize -> final float32 columns (raw kept as unnormalized_reward); design summary.
    final = {f: (rr.astype(np.float32),                 # unnormalized_reward (raw)
                 (rr / z).astype(np.float32),           # reward (normalized)
                 (mm / z).astype(np.float32))           # mc_return (normalized)
             for f, (rr, mm) in raw.items()}
    all_mc, starts_s, starts_f = [], [], []
    for f, (ep, rew_disk, _) in scal.items():
        mc = final[f][2]                          # (raw, reward, mc_return)
        all_mc.append(mc)
        bounds = np.r_[0, np.where(np.diff(ep) != 0)[0] + 1]
        epf = np.zeros(len(ep), bool); epf[rew_disk <= _FAIL_THRESH] = True
        for b, e_id in zip(bounds, ep[bounds]):
            (starts_f if epf[ep == e_id].any() else starts_s).append(float(mc[b]))
    all_mc = np.concatenate(all_mc); ss, sf = np.array(starts_s), np.array(starts_f)
    print(f"design   : living={LIVING_RAW/z:.2e}  fail_terminal={-c_fail/z:.3f}  "
          f"mc range=[{all_mc.min():.4f}, {all_mc.max():.4f}]")
    print(f"           success start: median={np.median(ss):.3f}  range=[{ss.min():.3f},{ss.max():.3f}]")
    if len(sf):
        print(f"           failure start: median={np.median(sf):.3f}  range=[{sf.min():.3f},{sf.max():.3f}]")
    oob = int(((all_mc < -1.0 - 1e-5) | (all_mc > 1e-5)).sum())
    print(f"           oob (outside [-1,0]): {oob}  (must be 0)")
    if args.dry_run:
        print("DRY RUN  : no files written.")
        return

    # ---------------- Phase B: parallel column rewrite ----------------
    jobs = [(str(f), str(f if args.inplace else dst_root / f.relative_to(src_root)),
             final[f][0], final[f][1], final[f][2], False) for f in files]
    print(f"\nphase B  : rewriting {len(jobs)} file(s) with {args.workers} worker(s)...")
    if args.workers <= 1:
        results = [write_file(j) for j in jobs]
    else:
        with multiprocessing.Pool(processes=min(args.workers, len(jobs))) as pool:
            results = pool.map(write_file, jobs)
    print(f"\n=== done: {sum(r.get('rows', 0) for r in results):,} rows; "
          f"Z={z:.1f}, C_fail={c_fail:.0f}, gamma={DISCOUNT} ===")


if __name__ == "__main__":
    main()
