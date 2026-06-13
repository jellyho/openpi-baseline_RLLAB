#!/usr/bin/env python3
"""Create synthetic failure prefixes from annotated HIL intervention parquet data.

Each synthetic episode ends at the last autonomous ``inference`` frame before a
human takeover begins (``inference`` -> ``teleop``). The script preserves all
semantic trajectory columns, remaps dataset indexing columns, and recomputes
reward targets for the new failure episodes.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environment
    raise SystemExit(
        "Missing dependency: pyarrow. Run this with the local uv environment, e.g.\n"
        "  uv pip install pyarrow\n"
        "  uv run python create_synthetic_intervention_failures.py --help"
    ) from exc


REQUIRED_COLUMNS = {
    "observation.commander_state",
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "reward",
    "mc_return",
}

DEFAULT_GAMMA = 0.9999
DEFAULT_C_FAIL_FRAC = 0.4
LIVING_RAW = -1.0
SCRIPT_VERSION = "2026-06-14-preserve-source-mc-return"


@dataclass
class InterventionCut:
    source_episode_index: int
    source_file: str
    source_file_start: int
    prefix_len: int
    source_intervention_index: int
    source_cut_frame_index: int
    source_cut_global_index: int


@dataclass
class EpisodeScan:
    episode_index: int
    source_file: str
    file_start: int
    length: int = 0
    modes: set[str] = field(default_factory=set)
    transitions: list[InterventionCut] = field(default_factory=list)
    prev_cmd: str | None = None
    prev_frame_index: int | None = None
    prev_global_index: int | None = None
    last_inference_pos: int | None = None
    last_inference_frame_index: int | None = None
    last_inference_global_index: int | None = None
    closed: bool = False


@dataclass
class FileScanResult:
    path: str
    valid: bool
    error: str | None = None
    schema: pa.Schema | None = None
    rows: int = 0
    row_group_sizes: list[int] = field(default_factory=list)
    episodes: dict[int, EpisodeScan] = field(default_factory=dict)
    max_episode_index: int = -1
    max_global_index: int = -1
    min_reward: float | None = None
    min_raw_reward: float | None = None
    living_z_values: list[float] = field(default_factory=list)
    small_reward_z_values: list[float] = field(default_factory=list)


@dataclass
class SyntheticPlan:
    new_episode_index: int
    output_index_start: int
    cut: InterventionCut


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _as_int(value: Any) -> int:
    return int(_unwrap_scalar(value))


def _as_float(value: Any) -> float:
    return float(_unwrap_scalar(value))


def _discover_parquet_files(input_root: Path) -> list[Path]:
    data_root = input_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"input data directory does not exist: {data_root}")
    return sorted(data_root.rglob("*.parquet"))


def _validate_required_columns(path: Path, schema_names: set[str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - schema_names)
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def _scan_file(path_text: str) -> FileScanResult:
    path = Path(path_text)
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        schema_names = set(schema.names)
        _validate_required_columns(path, schema_names)

        has_raw = "unnormalized_reward" in schema_names
        scalar_cols = [
            "episode_index",
            "frame_index",
            "index",
            "observation.commander_state",
            "reward",
        ]
        if has_raw:
            scalar_cols.append("unnormalized_reward")

        result = FileScanResult(
            path=str(path),
            valid=True,
            schema=schema,
            rows=pf.metadata.num_rows,
            row_group_sizes=[
                pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)
            ],
        )

        episodes: dict[int, EpisodeScan] = {}
        closed_episodes: set[int] = set()
        current_episode: EpisodeScan | None = None
        file_offset = 0

        for row_group_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(row_group_idx, columns=scalar_cols)
            eps = table["episode_index"].to_pylist()
            frames = table["frame_index"].to_pylist()
            indices = table["index"].to_pylist()
            cmds = table["observation.commander_state"].to_pylist()
            rewards = table["reward"].to_pylist()
            raw_rewards = table["unnormalized_reward"].to_pylist() if has_raw else None

            for i in range(table.num_rows):
                ep = _as_int(eps[i])
                frame = _as_int(frames[i])
                global_index = _as_int(indices[i])
                cmd = str(_unwrap_scalar(cmds[i]))
                reward = _as_float(rewards[i])

                result.max_episode_index = max(result.max_episode_index, ep)
                result.max_global_index = max(result.max_global_index, global_index)
                result.min_reward = reward if result.min_reward is None else min(result.min_reward, reward)

                if has_raw and raw_rewards is not None:
                    raw_reward = _as_float(raw_rewards[i])
                    result.min_raw_reward = (
                        raw_reward
                        if result.min_raw_reward is None
                        else min(result.min_raw_reward, raw_reward)
                    )
                    if math.isclose(raw_reward, LIVING_RAW, rel_tol=0.0, abs_tol=1e-5) and reward < 0:
                        result.living_z_values.append(abs(LIVING_RAW / reward))
                elif reward < 0 and abs(reward) < 0.01:
                    result.small_reward_z_values.append(abs(LIVING_RAW / reward))

                if current_episode is None or current_episode.episode_index != ep:
                    if current_episode is not None:
                        current_episode.closed = True
                        closed_episodes.add(current_episode.episode_index)
                    if ep in closed_episodes:
                        raise ValueError(
                            f"{path}: episode_index {ep} is non-contiguous within the file"
                        )
                    current_episode = episodes.get(ep)
                    if current_episode is None:
                        current_episode = EpisodeScan(
                            episode_index=ep,
                            source_file=str(path),
                            file_start=file_offset + i,
                        )
                        episodes[ep] = current_episode

                pos_in_episode = current_episode.length
                if cmd == "teleop" and current_episode.prev_cmd != "teleop":
                    if (
                        current_episode.last_inference_pos is not None
                        and current_episode.last_inference_frame_index is not None
                        and current_episode.last_inference_global_index is not None
                    ):
                        prefix_len = current_episode.last_inference_pos + 1
                        current_episode.transitions.append(
                            InterventionCut(
                                source_episode_index=ep,
                                source_file=str(path),
                                source_file_start=current_episode.file_start,
                                prefix_len=prefix_len,
                                source_intervention_index=len(current_episode.transitions) + 1,
                                source_cut_frame_index=current_episode.last_inference_frame_index,
                                source_cut_global_index=current_episode.last_inference_global_index,
                            )
                        )

                if cmd == "inference":
                    current_episode.last_inference_pos = pos_in_episode
                    current_episode.last_inference_frame_index = frame
                    current_episode.last_inference_global_index = global_index

                current_episode.modes.add(cmd)
                current_episode.length += 1
                current_episode.prev_cmd = cmd
                current_episode.prev_frame_index = frame
                current_episode.prev_global_index = global_index

            file_offset += table.num_rows

        result.episodes = episodes
        return result
    except BaseException as exc:
        return FileScanResult(path=str(path), valid=False, error=f"{type(exc).__name__}: {exc}")


def _scan_files(paths: list[Path], workers: int) -> list[FileScanResult]:
    if workers <= 1 or len(paths) <= 1:
        return [_scan_file(str(path)) for path in paths]

    import multiprocessing

    with multiprocessing.Pool(processes=min(workers, len(paths))) as pool:
        return pool.map(_scan_file, [str(path) for path in paths])


def _schemas_match(reference: pa.Schema, candidate: pa.Schema) -> bool:
    if reference.names != candidate.names:
        return False
    for name in reference.names:
        if reference.field(name).type != candidate.field(name).type:
            return False
    return True


def _infer_c_fail(results: list[FileScanResult], t_max: int, c_fail_frac: float) -> tuple[float, str]:
    raw_mins = [r.min_raw_reward for r in results if r.min_raw_reward is not None]
    if raw_mins:
        min_raw = min(raw_mins)
        if min_raw < LIVING_RAW - 0.5:
            return abs(float(min_raw)), "source unnormalized_reward failure terminal"
    return c_fail_frac * float(t_max), f"{c_fail_frac} * source T_max"


def _infer_normalization_z(
    results: list[FileScanResult],
    c_fail: float,
    synthetic_raw_min: float,
) -> tuple[float, str]:
    living_z_values: list[float] = []
    small_reward_z_values: list[float] = []
    reward_mins: list[float] = []

    for result in results:
        living_z_values.extend(v for v in result.living_z_values if math.isfinite(v) and v > 0)
        small_reward_z_values.extend(
            v for v in result.small_reward_z_values if math.isfinite(v) and v > 0
        )
        if result.min_reward is not None:
            reward_mins.append(result.min_reward)

    if living_z_values:
        return float(statistics.median(living_z_values)), "source living reward scale"

    if reward_mins:
        min_reward = min(reward_mins)
        if min_reward < -0.05:
            return abs(c_fail / min_reward), "source failure reward scale"

    if small_reward_z_values:
        return float(statistics.median(small_reward_z_values)), "source small living reward scale"

    return abs(synthetic_raw_min), "synthetic raw MC fallback"


def _episode_raw_mc_min(length: int, c_fail: float, gamma: float) -> float:
    running = 0.0
    min_mc = 0.0
    for i in range(length - 1, -1, -1):
        raw = -c_fail if i == length - 1 else LIVING_RAW
        running = raw + gamma * running
        min_mc = min(min_mc, running)
    return min_mc


def _build_synthetic_plans(
    results: list[FileScanResult],
    start_episode_index: int,
    start_global_index: int,
    source_episode_index: int | None = None,
    max_synthetic_episodes: int | None = None,
) -> list[SyntheticPlan]:
    plans: list[SyntheticPlan] = []
    next_episode = start_episode_index
    next_index = start_global_index

    for result in sorted(results, key=lambda r: r.path):
        for ep in sorted(result.episodes):
            if source_episode_index is not None and ep != source_episode_index:
                continue
            episode = result.episodes[ep]
            for cut in episode.transitions:
                if cut.prefix_len <= 0:
                    continue
                plans.append(
                    SyntheticPlan(
                        new_episode_index=next_episode,
                        output_index_start=next_index,
                        cut=cut,
                    )
                )
                next_episode += 1
                next_index += cut.prefix_len
                if max_synthetic_episodes is not None and len(plans) >= max_synthetic_episodes:
                    return plans
    return plans


def _read_file_slice(
    path: Path,
    start: int,
    length: int,
    columns: list[str] | None = None,
) -> pa.Table:
    pf = pq.ParquetFile(path)
    end = start + length
    row_group_starts: list[int] = []
    offset = 0
    groups: list[int] = []

    for group_idx in range(pf.metadata.num_row_groups):
        group_len = pf.metadata.row_group(group_idx).num_rows
        group_start = offset
        group_end = offset + group_len
        if group_start < end and group_end > start:
            groups.append(group_idx)
            row_group_starts.append(group_start)
        offset = group_end

    if not groups:
        raise ValueError(f"{path}: no row groups overlap slice start={start} length={length}")

    table = pf.read_row_groups(groups, columns=columns)
    first_group_start = row_group_starts[0]
    local_start = start - first_group_start
    return table.slice(local_start, length)


def _float32_array(values: list[float]) -> pa.Array:
    return pa.array(values, type=pa.float32())


def _int64_array(values: list[int]) -> pa.Array:
    return pa.array(values, type=pa.int64())


def _replace_or_append(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.schema.names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def _compute_rewards(length: int, c_fail: float, gamma: float, z: float) -> tuple[list[float], list[float], list[float]]:
    raw = [LIVING_RAW] * length
    raw[-1] = -c_fail

    raw_mc = [0.0] * length
    running = 0.0
    for i in range(length - 1, -1, -1):
        running = raw[i] + gamma * running
        raw_mc[i] = running

    reward = [x / z for x in raw]
    mc_return = [x / z for x in raw_mc]
    return raw, reward, mc_return


def _transform_slice(
    table: pa.Table,
    plan: SyntheticPlan,
    c_fail: float,
    gamma: float,
    z: float,
    recompute_mc_return: bool,
) -> pa.Table:
    n = table.num_rows
    if n != plan.cut.prefix_len:
        raise ValueError(f"slice length mismatch: expected {plan.cut.prefix_len}, got {n}")

    source_episode = table["episode_index"].to_pylist()
    source_frame = table["frame_index"].to_pylist()
    source_index = table["index"].to_pylist()

    raw, reward, mc_return = _compute_rewards(n, c_fail, gamma, z)

    table = _replace_or_append(table, "episode_index", _int64_array([plan.new_episode_index] * n))
    table = _replace_or_append(table, "frame_index", _int64_array(list(range(n))))
    table = _replace_or_append(
        table, "index", _int64_array(list(range(plan.output_index_start, plan.output_index_start + n)))
    )
    table = _replace_or_append(table, "unnormalized_reward", _float32_array(raw))
    table = _replace_or_append(table, "reward", _float32_array(reward))
    if recompute_mc_return:
        table = _replace_or_append(table, "mc_return", _float32_array(mc_return))
    table = _replace_or_append(table, "rl_mask", _float32_array([1.0] * n))

    table = _replace_or_append(table, "source_episode_index", _int64_array([_as_int(x) for x in source_episode]))
    table = _replace_or_append(table, "source_frame_index", _int64_array([_as_int(x) for x in source_frame]))
    table = _replace_or_append(table, "source_index", _int64_array([_as_int(x) for x in source_index]))
    table = _replace_or_append(
        table, "source_intervention_index", _int64_array([plan.cut.source_intervention_index] * n)
    )
    table = _replace_or_append(
        table, "source_cut_frame_index", _int64_array([plan.cut.source_cut_frame_index] * n)
    )
    table = _replace_or_append(
        table, "source_cut_global_index", _int64_array([plan.cut.source_cut_global_index] * n)
    )
    table = _replace_or_append(
        table,
        "synthetic_label",
        pa.array(["failure_before_intervention"] * n, type=pa.string()),
    )
    return table


def _compression_from_source(path: Path) -> dict[str, str]:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_row_groups == 0:
        return {}
    row_group = pf.metadata.row_group(0)
    compression: dict[str, str] = {}
    for i in range(row_group.num_columns):
        col = row_group.column(i)
        codec = col.compression.lower()
        compression[col.path_in_schema] = "none" if codec == "uncompressed" else codec
    return compression


def _write_output(
    output_root: Path,
    plans: list[SyntheticPlan],
    c_fail: float,
    gamma: float,
    z: float,
    overwrite: bool,
    read_columns: list[str] | None,
    recompute_mc_return: bool,
) -> Path:
    output_file = output_root / "data" / "chunk-000" / "file-000.parquet"
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    written_rows = 0
    t0 = time.time()

    try:
        for i, plan in enumerate(plans, start=1):
            source_path = Path(plan.cut.source_file)
            source_table = _read_file_slice(
                source_path, plan.cut.source_file_start, plan.cut.prefix_len, read_columns
            )
            table = _transform_slice(source_table, plan, c_fail, gamma, z, recompute_mc_return)

            if writer is None:
                compression = _compression_from_source(source_path)
                compression = {k: v for k, v in compression.items() if k in table.schema.names}
                for name in table.schema.names:
                    compression.setdefault(name, "snappy")
                writer = pq.ParquetWriter(tmp_file, table.schema, compression=compression)

            writer.write_table(table)
            written_rows += table.num_rows

            if i == len(plans) or i % 25 == 0:
                elapsed = time.time() - t0
                print(
                    f"write    : {i}/{len(plans)} synthetic episodes, "
                    f"{written_rows:,} rows, elapsed={elapsed:.0f}s",
                    flush=True,
                )

        if writer is None:
            raise ValueError("no synthetic plans to write")
        writer.close()
        writer = None
        os.replace(tmp_file, output_file)
        return output_file
    except BaseException:
        if writer is not None:
            writer.close()
        if tmp_file.exists():
            tmp_file.unlink()
        raise


def _validate_output(
    path: Path,
    start_episode_index: int,
    start_global_index: int,
    recompute_mc_return: bool,
) -> None:
    pf = pq.ParquetFile(path)
    columns = [
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "reward",
        "unnormalized_reward",
        "mc_return",
        "observation.commander_state",
        "source_cut_frame_index",
        "source_frame_index",
    ]
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise ValueError(f"output missing expected columns: {missing}")

    table = pf.read(columns=columns)
    eps = [_as_int(x) for x in table["episode_index"].to_pylist()]
    frames = [_as_int(x) for x in table["frame_index"].to_pylist()]
    indices = [_as_int(x) for x in table["index"].to_pylist()]
    raw = [_as_float(x) for x in table["unnormalized_reward"].to_pylist()]
    reward = [_as_float(x) for x in table["reward"].to_pylist()]
    mc = [_as_float(x) for x in table["mc_return"].to_pylist()]
    cmds = [str(_unwrap_scalar(x)) for x in table["observation.commander_state"].to_pylist()]
    source_cut_frames = [_as_int(x) for x in table["source_cut_frame_index"].to_pylist()]
    source_frames = [_as_int(x) for x in table["source_frame_index"].to_pylist()]

    if not eps:
        raise ValueError("output has no rows")
    if min(eps) != start_episode_index:
        raise ValueError(f"first output episode_index is {min(eps)}, expected {start_episode_index}")
    if min(indices) != start_global_index:
        raise ValueError(f"first output index is {min(indices)}, expected {start_global_index}")
    if indices != list(range(start_global_index, start_global_index + len(indices))):
        raise ValueError("output index is not continuous")

    by_episode: dict[int, list[int]] = defaultdict(list)
    for row_idx, ep in enumerate(eps):
        by_episode[ep].append(row_idx)

    for ep, row_ids in by_episode.items():
        expected_frames = list(range(len(row_ids)))
        actual_frames = [frames[i] for i in row_ids]
        if actual_frames != expected_frames:
            raise ValueError(f"episode {ep}: frame_index does not reset/increment from 0")
        terminal_rows = [i for i in row_ids if raw[i] < LIVING_RAW - 0.5]
        if len(terminal_rows) != 1 or terminal_rows[0] != row_ids[-1]:
            raise ValueError(f"episode {ep}: expected exactly one terminal failure on final row")
        last = row_ids[-1]
        if cmds[last] != "inference":
            raise ValueError(f"episode {ep}: terminal commander_state is {cmds[last]!r}, expected inference")
        if source_frames[last] != source_cut_frames[last]:
            raise ValueError(f"episode {ep}: terminal source frame does not match cut frame")

    if min(reward) < -1.0001 or max(reward) > 0.0001:
        raise ValueError(f"reward outside [-1, 0]: range [{min(reward)}, {max(reward)}]")
    if recompute_mc_return and (min(mc) < -1.0001 or max(mc) > 0.0001):
        raise ValueError(f"mc_return outside [-1, 0]: range [{min(mc)}, {max(mc)}]")


def _print_summary(
    results: list[FileScanResult],
    invalid: list[FileScanResult],
    plans: list[SyntheticPlan],
    t_max: int,
    c_fail: float,
    c_fail_source: str,
    z: float,
    z_source: str,
    synthetic_raw_min: float,
    start_episode_index: int,
    start_global_index: int,
    recompute_mc_return: bool,
) -> None:
    total_source_rows = sum(r.rows for r in results)
    all_episodes = [episode for r in results for episode in r.episodes.values()]
    hil_episodes = [
        ep for ep in all_episodes if "inference" in ep.modes and "teleop" in ep.modes
    ]
    transition_hist = Counter(len(ep.transitions) for ep in hil_episodes)
    prefix_rows = sum(plan.cut.prefix_len for plan in plans)

    print("summary  :")
    print(f"  valid parquet files     : {len(results)}")
    print(f"  invalid parquet files   : {len(invalid)}")
    for item in invalid:
        print(f"    invalid: {item.path}: {item.error}")
    print(f"  source episodes         : {len(all_episodes)}")
    print(f"  source rows             : {total_source_rows:,}")
    print(f"  HIL episodes            : {len(hil_episodes)}")
    print(f"  synthetic episodes      : {len(plans)}")
    print(f"  synthetic rows          : {prefix_rows:,}")
    print(f"  interventions/HIL hist  : {dict(sorted(transition_hist.items()))}")
    print(f"  first episode_index     : {start_episode_index}")
    print(f"  first index             : {start_global_index}")
    print(f"  T_max(source task)      : {t_max}")
    print(f"  C_fail                  : {c_fail:.6g} ({c_fail_source})")
    print(
        "  mc_return handling      : "
        + ("recomputed from synthetic rewards" if recompute_mc_return else "preserved from source rows")
    )
    print(f"  raw synthetic mc min    : {synthetic_raw_min:.6g}")
    print(f"  normalization Z         : {z:.6g} ({z_source})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic failure prefix parquet data from HIL interventions."
    )
    parser.add_argument("--input", required=True, type=Path, help="annotated dataset root")
    parser.add_argument("--output", required=True, type=Path, help="synthetic output root")
    parser.add_argument("--dry-run", action="store_true", help="scan and print summary only")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="skip unreadable parquet shards instead of failing",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace output root if it exists")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers for scan phase")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--c-fail-frac", type=float, default=DEFAULT_C_FAIL_FRAC)
    parser.add_argument("--c-fail", type=float, default=None, help="override raw failure penalty")
    parser.add_argument("--normalization-z", type=float, default=None, help="override reward scale Z")
    parser.add_argument(
        "--recompute-mc-return",
        action="store_true",
        help="recompute mc_return from synthetic rewards; by default source mc_return is copied unchanged",
    )
    parser.add_argument(
        "--source-episode-index",
        type=int,
        default=None,
        help="only create synthetic samples from this source episode_index",
    )
    parser.add_argument(
        "--max-synthetic-episodes",
        type=int,
        default=None,
        help="cap the number of synthetic episodes written after filtering",
    )
    parser.add_argument(
        "--drop-image-columns",
        action="store_true",
        help="omit source columns whose names start with observation.images",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="omit an additional source column from the output; may be passed multiple times",
    )
    return parser.parse_args()


def _build_read_columns(schema: pa.Schema, drop_image_columns: bool, drop_columns: list[str]) -> list[str] | None:
    drop = set(drop_columns)
    if drop_image_columns:
        drop.update(name for name in schema.names if name.startswith("observation.images"))

    protected = REQUIRED_COLUMNS | {"unnormalized_reward", "rl_mask"}
    protected_drops = sorted(drop & protected)
    if protected_drops:
        raise ValueError(f"cannot drop required/reward/index columns: {protected_drops}")

    if not drop:
        return None

    unknown = sorted(name for name in drop if name not in schema.names)
    if unknown:
        raise ValueError(f"cannot drop columns not present in source schema: {unknown}")

    return [name for name in schema.names if name not in drop]


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    files = _discover_parquet_files(input_root)
    if not files:
        raise FileNotFoundError(f"no parquet files found under {input_root / 'data'}")

    print(f"input    : {input_root}")
    print(f"output   : {output_root}")
    print(f"files    : {len(files)} parquet shard(s)")
    print(f"workers  : {args.workers}")
    print(f"version  : {SCRIPT_VERSION}")

    scan_results = _scan_files(files, args.workers)
    invalid = [r for r in scan_results if not r.valid]
    valid = [r for r in scan_results if r.valid]

    if invalid and not args.skip_invalid:
        details = "\n".join(f"  {r.path}: {r.error}" for r in invalid)
        raise RuntimeError(f"invalid parquet files found:\n{details}")
    if not valid:
        raise RuntimeError("no valid parquet files to process")

    reference_schema = valid[0].schema
    assert reference_schema is not None
    for result in valid[1:]:
        assert result.schema is not None
        if not _schemas_match(reference_schema, result.schema):
            raise ValueError(
                f"schema mismatch between {valid[0].path} and {result.path}; "
                "all input shards must have the same source schema"
            )

    all_episode_lengths = [
        episode.length for result in valid for episode in result.episodes.values()
    ]
    if not all_episode_lengths:
        raise RuntimeError("no source episodes found")

    start_episode_index = max(r.max_episode_index for r in valid) + 1
    start_global_index = max(r.max_global_index for r in valid) + 1
    t_max = max(all_episode_lengths)
    c_fail_source = "command line --c-fail"
    c_fail = args.c_fail
    if c_fail is None:
        c_fail, c_fail_source = _infer_c_fail(valid, t_max, args.c_fail_frac)

    read_columns = _build_read_columns(reference_schema, args.drop_image_columns, args.drop_column)
    if read_columns is not None:
        dropped = [name for name in reference_schema.names if name not in read_columns]
        print(f"drop     : omitting {len(dropped)} source column(s): {dropped}")
    if args.source_episode_index is not None:
        print(f"filter   : source_episode_index == {args.source_episode_index}")
    if args.max_synthetic_episodes is not None:
        print(f"filter   : max_synthetic_episodes == {args.max_synthetic_episodes}")

    plans = _build_synthetic_plans(
        valid,
        start_episode_index,
        start_global_index,
        source_episode_index=args.source_episode_index,
        max_synthetic_episodes=args.max_synthetic_episodes,
    )
    if not plans:
        raise RuntimeError("no inference -> teleop intervention transitions found")

    synthetic_raw_min = min(
        _episode_raw_mc_min(plan.cut.prefix_len, c_fail, args.gamma) for plan in plans
    )
    z_source = "command line --normalization-z"
    z = args.normalization_z
    if z is None:
        z, z_source = _infer_normalization_z(valid, c_fail, synthetic_raw_min)
    if z <= 0:
        raise ValueError(f"normalization Z must be positive, got {z}")

    _print_summary(
        valid,
        invalid,
        plans,
        t_max,
        c_fail,
        c_fail_source,
        z,
        z_source,
        synthetic_raw_min,
        start_episode_index,
        start_global_index,
        args.recompute_mc_return,
    )

    if args.dry_run:
        print("dry-run  : no files written")
        return

    output_file = _write_output(
        output_root,
        plans,
        c_fail,
        args.gamma,
        z,
        args.overwrite,
        read_columns,
        args.recompute_mc_return,
    )
    print(f"wrote    : {output_file}")
    _validate_output(output_file, start_episode_index, start_global_index, args.recompute_mc_return)
    print("validate : OK")


if __name__ == "__main__":
    main()
