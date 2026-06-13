"""Reward re-annotation v3: raw rewards first, then FIXED-constant normalization into [-1, 0].

Scheme ("set natural units, then normalize by a fixed constant"):
  1. RAW rewards:   living step = -1,  success terminal = 0,
                    failure terminal = -C_FAIL  (C_FAIL = 10000, a fixed constant -- NOT
                    data-derived).
  2. RAW mc_return: discounted return-to-go with gamma = 0.9999 (near-undiscounted, so the
                    failure penalty stays visible from the episode start).
  3. NORMALIZE:     divide by the FIXED constant Z_DENOM = 10000 (NOT the per-dataset most
                    negative return).  reward /= Z_DENOM and mc_return /= Z_DENOM.

Why these exact numbers: C_FAIL = Z_DENOM = 1/(1-gamma) = 10000.  With that choice the
living cost per step exactly cancels the discount erosion of the failure terminal, so the
return-to-go of EVERY frame in a failure episode is exactly -10000 -> normalized -1.0
(independent of where in the episode / how long it is), and success episodes land in
(-1, 0].  Everything stays in [-1, 0] with no per-dataset scan needed.

Columns written:
  * unnormalized_reward  (NEW)  raw reward (-1 / 0 / -C_FAIL), kept for re-normalization later
  * reward                      normalized (= unnormalized_reward / Z_DENOM)
  * mc_return                   normalized return-to-go (gamma=0.9999, / Z_DENOM)

Because the constants are fixed, no global scan is needed, but the script still runs in
two phases for the per-file column build:
  Phase A (single process): scan all files' scalars, classify frames (living/success/failure
           from the CURRENT disk values), build raw rewards, compute per-file mc (episodes
           never span files), divide by the fixed Z_DENOM.
  Phase B (parallel, one worker per file): rewrite only the reward/mc_return columns,
           atomic temp+rename, original column compression preserved.

Idempotency: the new scheme's signature is min(mc_return) ~ -1.0 on disk (old schemes
bottom out at -0.5/-0.6).  If the dataset already looks normalized to -1, the script skips
(pass --force to re-annotate anyway, e.g. a merged orig+augmented set).

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

DISCOUNT    = 0.9999   # near-undiscounted; note 1/(1-DISCOUNT) = 10000 = C_FAIL = Z_DENOM
LIVING_RAW  = -1.0     # raw living cost per step
C_FAIL      = 10000.0  # raw failure-terminal cost (fixed constant, NOT data-derived)
Z_DENOM     = 10000.0  # fixed normalization denominator (reward/mc_return /= this)

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
    ap.add_argument("--force", action="store_true",
                    help="bypass the 'already v3-normalized' skip (needed when merging an "
                         "already-v3 augmented set with raw originals: the augmented part can "
                         "make disk mc_min hit -1 even though a fresh JOINT normalize is wanted)")
    args = ap.parse_args()

    src_root = Path(args.input)
    dst_root = src_root if args.inplace else Path(args.output or (str(src_root) + "_v3"))
    files = sorted((src_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {src_root}/data")

    print(f"=== reward_annotate v3 (raw -> fixed-constant normalize) ===")
    print(f"input    : {src_root}")
    print(f"output   : {dst_root}{'  (IN-PLACE)' if args.inplace else ''}")
    print(f"scheme   : living={LIVING_RAW}, C_fail={C_FAIL:.0f}, gamma={DISCOUNT}, "
          f"then /Z_DENOM={Z_DENOM:.0f} (both fixed constants)")

    # ---------------- Phase A: global scan + compute ----------------
    t0 = time.time()
    scal = {}                                    # file -> (ep, rew_disk, mc_disk)
    for f in files:
        scal[f] = scan_scalars(pq.ParquetFile(f))
    print(f"phase A  : scalars loaded ({time.time()-t0:.0f}s)")

    # idempotency: new scheme normalizes the deepest return to exactly -1 on disk.
    disk_mc_min = min(float(m.min()) for (_, _, m) in scal.values())
    if disk_mc_min < -0.9 and not args.force:
        print(f"SKIP     : disk mc_return min = {disk_mc_min:.4f} -> already v3-normalized. "
              f"(pass --force to re-annotate, e.g. a merged orig+augmented set)")
        return
    if args.force:
        print(f"force    : skip-guard bypassed (disk mc_return min = {disk_mc_min:.4f})")

    # Fixed constants (NOT data-derived): C_fail and the normalization denominator Z.
    c_fail = C_FAIL
    z = Z_DENOM
    ep_ids = set()
    for (ep, _, _) in scal.values():
        ep_ids.update(np.unique(ep).tolist())
    n_fail_eps = sum(int((r <= _FAIL_THRESH).sum()) for (_, r, _) in scal.values())

    # raw rewards + per-file raw mc (fixed C_fail; divided later by fixed Z).
    raw = {}                                     # file -> (rew_raw, mc_raw)
    for f, (ep, rew_disk, _) in scal.items():
        rr = raw_rewards(rew_disk, c_fail)
        mm = compute_mc(rr, ep, DISCOUNT)
        raw[f] = (rr, mm)
    print(f"phase A  : episodes={len(ep_ids)} (fail={n_fail_eps})  "
          f"C_fail={c_fail:.0f}  Z={z:.0f} (both fixed)  ({time.time()-t0:.0f}s)")

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
