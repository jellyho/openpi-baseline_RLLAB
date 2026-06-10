"""Reward relabeling + MC-return recomputation for the annotated VLA dataset.

Updates the existing insert-mouse-battery_annotated parquets:
  - reward:     living step -0.0001 -> RELABEL_LIVING (-0.0004)
                success terminal 0.0 -> 0.0  (unchanged)
                failure terminal -0.5 -> -0.5 (unchanged; C_fail stays same)
  - mc_return:  recompute from the updated reward with MC_GAMMA (0.999)
                (original was precomputed at gamma=0.995; long-horizon needs 0.999)

Writes updated files to OUTPUT_ROOT (same directory layout).
Files that already exist at the output path are skipped (resumable).

Usage:
    python reward_annotate.py [--dry_run] [--input INPUT] [--output OUTPUT] [--workers N]

    # 4 files in parallel (default; each file is independent — ~4x speedup on lustre):
    python reward_annotate.py --workers 4

Two-pass strategy (required because episodes span multiple ~1000-row row-groups):
  Pass 1: read episode_index + reward from ALL row-groups of the file (cheap scalars only)
          -> relabel reward -> compute full-episode mc_return backward pass (~0.5s/file)
  Pass 2: read full row-groups one at a time, swap in the precomputed reward + mc_return,
          write to output parquet row-group by row-group (I/O bottleneck: ~20 min with 4 workers)

Timing (measured on lustre, file-003 280k rows):
  pass1 scalars: 0.3s,  backward mc: 0.2s,  pass2 read+write: ~360s
  -> bottleneck is lustre write; CPU irrelevant. Speedup via parallel files.
  -> 4 files sequential: ~66 min. 4 workers parallel: ~20 min (largest file bottleneck).
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
OUTPUT_ROOT = "/lustre/gwanwoo13/rss_post_training/Challenge-phase1-dataset/insert-mouse-battery_annotated_v2"

RELABEL_LIVING = -4e-4   # was -1e-4
RELABEL_FAIL   = -0.5    # keep; raw -0.5 -> -0.5  (no change to C_fail)
MC_GAMMA       = 0.999   # was 0.995

# terminal detection: reward in {0.0, -0.5}; living is -1e-4 / -4e-4
_TERMINAL_THRESH = -0.05  # reward <= this -> failure terminal; >= -1e-6 -> success terminal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def relabel_reward(rew: np.ndarray) -> np.ndarray:
    """Map raw reward -> new scale. In-place safe; returns float32 array."""
    out = rew.copy().astype(np.float32)
    living_mask  = (rew > _TERMINAL_THRESH) & (rew < -1e-6)   # -1e-4 living cost
    failure_mask = rew <= _TERMINAL_THRESH                      # -0.5 failure terminal
    success_mask = rew >= -1e-6                                 # 0.0  success terminal
    # scale living cost; terminals stay as configured
    scale = RELABEL_LIVING / -1e-4                              # 4.0 for -4e-4
    out[living_mask]  = (rew[living_mask] * scale).astype(np.float32)
    out[failure_mask] = np.float32(RELABEL_FAIL)
    out[success_mask] = np.float32(0.0)
    return out


def compute_mc_return(rew_relabeled: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Backward pass to compute discounted MC return per frame.

    Resets accumulation at episode boundaries. Works on a FULL FILE's worth of
    frames (not per-row-group), so episodes that span multiple row-groups are
    computed correctly.
    """
    n = len(rew_relabeled)
    mc = np.empty(n, dtype=np.float32)
    g = float(MC_GAMMA)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = float(rew_relabeled[i]) + g * running
        mc[i] = running
        if i == 0 or ep[i - 1] != ep[i]:
            running = 0.0
    return mc


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def _column_compression(pf) -> dict:
    """Per-column compression of the source file (preserve original encoding).

    The big array columns (rl_token, base_action, ...) are stored UNCOMPRESSED;
    forcing snappy on them would waste CPU (float data barely compresses) and
    change the file. Mirror the source codec per column.
    """
    rg = pf.metadata.row_group(0)
    comp = {}
    for i in range(rg.num_columns):
        c = rg.column(i)
        codec = c.compression.lower()                       # 'snappy' / 'uncompressed'
        comp[c.path_in_schema] = "none" if codec == "uncompressed" else codec
    return comp


def _already_relabeled(rew_raw: np.ndarray) -> bool:
    """True if this file's living cost is already the relabeled value (~-4e-4).

    The relabel is NOT idempotent (-1e-4 -> -4e-4 -> -1.6e-3 if run twice), so an
    in-place re-run must skip files already processed. Raw living is -1e-4; after
    relabel it is RELABEL_LIVING. Use the median living magnitude as the signal.
    """
    living = rew_raw[(rew_raw > _TERMINAL_THRESH) & (rew_raw < -1e-6)]
    if len(living) == 0:
        return False
    med = float(np.median(np.abs(living)))
    # halfway between raw 1e-4 and relabeled 4e-4 is 2.5e-4
    return med > 2.5e-4


def process_file(src_path: Path, dst_path: Path, dry_run: bool = False) -> dict:
    """Relabel reward + recompute mc_return for one parquet file.

    Writes atomically: build ``<dst>.tmp`` in the same directory, then os.replace()
    over ``dst``. When dst == src this is a safe in-place update — the original is
    never left half-written even if the job dies (rl_token/base_action are expensive
    to regenerate, so the original must be protected).
    """
    inplace = (dst_path.resolve() == src_path.resolve())
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src_path)
    n_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    mode = "in-place" if inplace else "copy"

    # Filename tag prefixes every line. Workers run concurrently and share one
    # stdout, so each progress message must be a full, self-identifying line —
    # otherwise the 4 streams interleave into garbage (the old `100/923 100/281`).
    tag = src_path.name

    print(f"[{tag}] src   : {n_groups} row-groups, {total_rows:,} rows ({mode})", flush=True)
    if dry_run:
        print(f"[{tag}] dst   : {dst_path}  [DRY RUN — skipping write]", flush=True)
        return {}

    t0 = time.time()

    # ---- Pass 1: scalars only — relabel reward, compute full mc_return ----
    ep_chunks, rew_chunks = [], []
    for g in range(n_groups):
        t = pf.read_row_group(g, columns=["episode_index", "reward"])
        ep_chunks.append(np.asarray(t["episode_index"].to_pylist(), dtype=np.int64))
        rew_chunks.append(np.asarray(t["reward"].to_pylist(), dtype=np.float32))

    ep_full  = np.concatenate(ep_chunks)
    rew_raw  = np.concatenate(rew_chunks)

    # idempotency guard: skip files already relabeled (protects in-place re-runs)
    if _already_relabeled(rew_raw):
        print(f"[{tag}] SKIP  : already relabeled (living ~{RELABEL_LIVING})", flush=True)
        return {"skipped": True}

    rew_new  = relabel_reward(rew_raw)
    mc_new   = compute_mc_return(rew_new, ep_full)

    n_living  = int(((rew_raw > _TERMINAL_THRESH) & (rew_raw < -1e-6)).sum())
    n_failure = int((rew_raw <= _TERMINAL_THRESH).sum())
    n_success = int((rew_raw >= -1e-6).sum())
    print(f"[{tag}] pass1 : living={n_living:,}  failure={n_failure}  success={n_success}  "
          f"mc_return=[{mc_new.min():.4f}, {mc_new.max():.4f}]", flush=True)

    # sanity: no values outside [-1, 0]
    oob = int(((mc_new < -1.0 - 1e-5) | (mc_new > 1e-5)).sum())
    if oob:
        print(f"[{tag}] WARNING: {oob} mc_return values outside [-1, 0]! Check gamma/reward.", flush=True)

    # ---- Pass 2: read full row-groups, swap columns, write to a temp file ----
    comp = _column_compression(pf)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()   # clean a stale temp from a previous failed run
    print(f"[{tag}] pass2 : writing {tmp_path.name}  (0%)", flush=True)
    row_offset = 0
    writer = None
    next_pct = 10                       # emit a full progress line every +10%
    try:
        for g in range(n_groups):
            table = pf.read_row_group(g)    # full row-group (all columns)
            n_g = table.num_rows
            rew_col = pa.array(rew_new[row_offset: row_offset + n_g], type=pa.float32())
            mc_col  = pa.array(mc_new[row_offset: row_offset + n_g],  type=pa.float32())
            table = table.set_column(table.schema.get_field_index("reward"),    "reward",    rew_col)
            table = table.set_column(table.schema.get_field_index("mc_return"), "mc_return", mc_col)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=comp)
            writer.write_table(table)
            row_offset += n_g
            # Full self-identifying line so the 4 concurrent workers stay readable.
            pct = (g + 1) * 100 // n_groups
            if pct >= next_pct or g + 1 == n_groups:
                elapsed = time.time() - t0
                eta = elapsed * (n_groups - (g + 1)) / (g + 1)
                print(f"[{tag}] pass2 : {pct:3d}%  ({g+1:>4}/{n_groups} rg, "
                      f"{row_offset:,}/{total_rows:,} rows)  elapsed={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)
                next_pct = pct - (pct % 10) + 10
        if writer:
            writer.close()
            writer = None
    except BaseException:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()   # never leave a partial temp behind
        raise

    # ---- atomic swap: replace dst with the completed temp ----
    os.replace(tmp_path, dst_path)   # atomic on same filesystem; original safe until here
    elapsed = time.time() - t0
    print(f"[{tag}] done  : {elapsed:.0f}s  -> {dst_path.name}  ({mode})", flush=True)

    return {
        "rows": total_rows,
        "n_living": n_living,
        "n_failure": n_failure,
        "mc_min": float(mc_new.min()),
        "mc_max": float(mc_new.max()),
        "oob": oob,
    }


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------
def _worker(args_tuple):
    src_path, dst_path, dry_run = args_tuple
    return process_file(src_path, dst_path, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Relabel reward + recompute mc_return")
    parser.add_argument("--input",   default=INPUT_ROOT,  help="annotated dataset root")
    parser.add_argument("--output",  default=OUTPUT_ROOT, help="output dataset root (ignored if --inplace)")
    parser.add_argument("--inplace", action="store_true",
                        help="overwrite reward/mc_return in the input files directly "
                             "(atomic temp+rename; no 197GB copy). Idempotency-guarded.")
    parser.add_argument("--workers", type=int, default=4, help="parallel file workers (default=4)")
    parser.add_argument("--dry_run", action="store_true", help="scan only, no writes")
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = src_root if args.inplace else Path(args.output)

    src_files = sorted((src_root / "data").rglob("*.parquet"))
    if not src_files:
        raise FileNotFoundError(f"No parquet files under {src_root}/data")

    print(f"=== reward_annotate ===")
    print(f"input   : {src_root}")
    print(f"output  : {dst_root}" + ("  (IN-PLACE — overwrites originals)" if args.inplace else ""))
    print(f"relabel : living {-1e-4} -> {RELABEL_LIVING},  fail {-0.5} -> {RELABEL_FAIL}")
    print(f"gamma   : {MC_GAMMA}  (was 0.995)")
    print(f"files   : {len(src_files)}")
    print(f"workers : {args.workers}  (each file is independent; bottleneck = lustre write)")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    jobs = []
    for src_path in src_files:
        rel = src_path.relative_to(src_root)
        dst_path = src_path if args.inplace else (dst_root / rel)
        # in-place: idempotency guard inside process_file decides skip (re-run safe).
        # copy mode: skip if the output already exists.
        if not args.inplace and dst_path.exists():
            print(f"  SKIP (exists): {dst_path.name}")
            continue
        jobs.append((src_path, dst_path, args.dry_run))

    if not jobs:
        print("Nothing to do.")
        return

    print(f"\n{len(jobs)} file(s) to process with {args.workers} worker(s)...\n")

    if args.workers == 1 or args.dry_run:
        stats = [_worker(j) for j in jobs]
    else:
        with multiprocessing.Pool(processes=min(args.workers, len(jobs))) as pool:
            stats = pool.map(_worker, jobs)

    n_skipped = sum(1 for s in stats if s and s.get("skipped"))
    stats = [s for s in stats if s and not s.get("skipped")]
    print(f"\n=== summary ===")
    if n_skipped:
        print(f"skipped       : {n_skipped} file(s) (already relabeled)")
    if stats:
        print(f"processed     : {len(stats)} file(s)")
        print(f"total rows    : {sum(s['rows'] for s in stats):,}")
        print(f"failure terms : {sum(s['n_failure'] for s in stats)}")
        mc_min = min(s['mc_min'] for s in stats)
        mc_max = max(s['mc_max'] for s in stats)
        print(f"mc_return     : [{mc_min:.4f}, {mc_max:.4f}]  (must be within [-1, 0])")
        print(f"oob values    : {sum(s['oob'] for s in stats)}  (must be 0)")


if __name__ == "__main__":
    main()
