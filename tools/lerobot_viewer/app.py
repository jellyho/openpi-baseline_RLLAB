"""LeRobot dataset web viewer.

A lightweight Flask app that scans a directory tree of LeRobot (v2.x) datasets
and serves an interactive web UI to inspect metadata, browse episodes, play the
camera videos, and plot state/action trajectories.

The dataset root is expected to contain one or more LeRobot datasets, each
identified by a ``meta/info.json`` file. Datasets can be nested arbitrarily
(e.g. ``<task>/<subset>/meta/info.json``); the path relative to the root (minus
``/meta``) is used as the dataset id.

Run:
    python app.py --root /data5/jellyho/PFR_RSS/dataset/phase1 --port 7800
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import socket
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Populated in main().
ROOT: Path = Path(".")
CACHE_DIR: Path = Path(".")
SERVER_IP: str = ""
SERVER_PORT: int = 0

# In-memory caches.
_DATASETS: dict[str, Path] = {}          # ds_id -> dataset dir
_OVERVIEW_CACHE: dict | None = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Discovery & metadata helpers
# --------------------------------------------------------------------------- #
def discover_datasets(root: Path) -> dict[str, Path]:
    """Find every LeRobot dataset (dir containing meta/info.json) under root."""
    datasets: dict[str, Path] = {}
    for info_path in sorted(root.rglob("meta/info.json")):
        ds_dir = info_path.parent.parent
        ds_id = ds_dir.relative_to(root).as_posix()
        datasets[ds_id] = ds_dir
    return datasets


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@functools.lru_cache(maxsize=64)
def load_info(ds_id: str) -> dict:
    return json.loads((_DATASETS[ds_id] / "meta" / "info.json").read_text())


@functools.lru_cache(maxsize=64)
def load_episodes(ds_id: str) -> list[dict]:
    return _read_jsonl(_DATASETS[ds_id] / "meta" / "episodes.jsonl")


@functools.lru_cache(maxsize=64)
def load_tasks(ds_id: str) -> dict[int, str]:
    rows = _read_jsonl(_DATASETS[ds_id] / "meta" / "tasks.jsonl")
    return {r["task_index"]: r["task"] for r in rows}


@functools.lru_cache(maxsize=64)
def load_episode_stats(ds_id: str) -> dict[int, dict]:
    rows = _read_jsonl(_DATASETS[ds_id] / "meta" / "episodes_stats.jsonl")
    return {r["episode_index"]: r["stats"] for r in rows}


def camera_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def video_path(ds_id: str, ep_idx: int, cam: str) -> Path:
    info = load_info(ds_id)
    chunk = ep_idx // info["chunks_size"]
    rel = info["video_path"].format(episode_chunk=chunk, video_key=cam, episode_index=ep_idx)
    return _DATASETS[ds_id] / rel


def data_path(ds_id: str, ep_idx: int) -> Path:
    info = load_info(ds_id)
    chunk = ep_idx // info["chunks_size"]
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
    return _DATASETS[ds_id] / rel


# --------------------------------------------------------------------------- #
# Overview aggregation (meta-only -> fast)
# --------------------------------------------------------------------------- #
def build_overview() -> dict:
    global _OVERVIEW_CACHE
    with _LOCK:
        if _OVERVIEW_CACHE is not None:
            return _OVERVIEW_CACHE

        datasets = []
        for ds_id in _DATASETS:
            info = load_info(ds_id)
            eps = load_episodes(ds_id)
            tasks = load_tasks(ds_id)
            fps = info.get("fps", 0) or 1
            lengths = [e.get("length", 0) for e in eps]

            cams = camera_keys(info)
            feats = info["features"]
            state_dim = feats.get("observation.state", {}).get("shape", [None])[0]
            action_dim = feats.get("action", {}).get("shape", [None])[0]

            # split task/subset from the id when it looks like "<task>/<subset>"
            parts = ds_id.split("/")
            group = parts[0] if len(parts) > 1 else ds_id
            subset = "/".join(parts[1:]) if len(parts) > 1 else ""

            # per-episode label (from episodes.jsonl "tasks") distribution
            labels: dict[str, int] = {}
            for e in eps:
                t = e.get("tasks", "")
                t = t if isinstance(t, str) else json.dumps(t)
                labels[t] = labels.get(t, 0) + 1

            datasets.append({
                "id": ds_id,
                "group": group,
                "subset": subset,
                "robot_type": info.get("robot_type"),
                "codebase_version": info.get("codebase_version"),
                "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes", len(eps)),
                "total_frames": info.get("total_frames", int(sum(lengths))),
                "duration_min": round(sum(lengths) / fps / 60, 2),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "cameras": cams,
                "resolution": _resolution(feats, cams),
                "task_instructions": list(tasks.values()),
                "episode_labels": labels,
                "length_stats": _length_stats(lengths, fps),
                "video_codec": _codec(feats, cams),
            })

        groups: dict[str, dict] = {}
        for d in datasets:
            g = groups.setdefault(d["group"], {
                "group": d["group"], "total_episodes": 0, "total_frames": 0,
                "duration_min": 0.0, "subsets": []})
            g["total_episodes"] += d["total_episodes"]
            g["total_frames"] += d["total_frames"]
            g["duration_min"] = round(g["duration_min"] + d["duration_min"], 2)
            g["subsets"].append(d["subset"] or d["id"])

        _OVERVIEW_CACHE = {
            "root": str(ROOT),
            "server_ip": SERVER_IP,
            "server_port": SERVER_PORT,
            "datasets": datasets,
            "groups": list(groups.values()),
            "totals": {
                "datasets": len(datasets),
                "episodes": sum(d["total_episodes"] for d in datasets),
                "frames": sum(d["total_frames"] for d in datasets),
                "duration_min": round(sum(d["duration_min"] for d in datasets), 1),
            },
        }
        return _OVERVIEW_CACHE


def _resolution(feats: dict, cams: list[str]) -> str | None:
    if not cams:
        return None
    shp = feats[cams[0]].get("shape")
    if shp and len(shp) >= 2:
        return f"{shp[1]}x{shp[0]}"
    return None


def _codec(feats: dict, cams: list[str]) -> str | None:
    if not cams:
        return None
    return feats[cams[0]].get("info", {}).get("video.codec")


def _length_stats(lengths: list[int], fps: float) -> dict:
    if not lengths:
        return {}
    a = np.asarray(lengths, dtype=float)
    # histogram in seconds for the dashboard
    secs = a / fps
    hist, edges = np.histogram(secs, bins=min(30, max(5, len(a) // 3)))
    return {
        "min": int(a.min()), "max": int(a.max()),
        "mean": round(float(a.mean()), 1), "median": int(np.median(a)),
        "sec_hist": {"counts": hist.tolist(),
                     "edges": [round(float(x), 1) for x in edges]},
    }


# --------------------------------------------------------------------------- #
# Commander-state (HIL intervention) cache — computed lazily per dataset
# --------------------------------------------------------------------------- #
def _commander_cache_file(ds_id: str) -> Path:
    return CACHE_DIR / (ds_id.replace("/", "__") + "__commander.json")


def compute_commander_stats(ds_id: str, force: bool = False) -> dict:
    """Per-episode commander_state fraction (inference/teleop/...). Cached to disk."""
    cache_file = _commander_cache_file(ds_id)
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text())

    info = load_info(ds_id)
    if "observation.commander_state" not in info["features"]:
        result = {"available": False}
        cache_file.write_text(json.dumps(result))
        return result

    per_ep: dict[str, dict] = {}
    states_total: dict[str, int] = {}
    for e in load_episodes(ds_id):
        idx = e["episode_index"]
        p = data_path(ds_id, idx)
        if not p.exists():
            continue
        col = pq.read_table(p, columns=["observation.commander_state"]).to_pandas()
        vc = col["observation.commander_state"].astype(str).value_counts().to_dict()
        n = int(sum(vc.values())) or 1
        per_ep[str(idx)] = {k: int(v) for k, v in vc.items()}
        for k, v in vc.items():
            states_total[k] = states_total.get(k, 0) + int(v)

    result = {"available": True, "per_episode": per_ep, "totals": states_total}
    cache_file.write_text(json.dumps(result))
    return result


# --------------------------------------------------------------------------- #
# Trajectory extraction
# --------------------------------------------------------------------------- #
def _segments(values: list[str]) -> list[dict]:
    """Run-length encode a categorical per-frame series into [start,end) segments."""
    segs = []
    if not values:
        return segs
    cur = values[0]
    start = 0
    for i in range(1, len(values)):
        if values[i] != cur:
            segs.append({"label": cur, "start": start, "end": i})
            cur, start = values[i], i
    segs.append({"label": cur, "start": start, "end": len(values)})
    return segs


def extract_trajectory(ds_id: str, ep_idx: int, max_points: int = 1500) -> dict:
    p = data_path(ds_id, ep_idx)
    if not p.exists():
        abort(404, f"episode parquet not found: {p}")
    info = load_info(ds_id)
    feats = info["features"]

    cols = ["timestamp", "frame_index"]
    for c in ("observation.state", "action", "observation.commander_state", "subtask"):
        if c in feats:
            cols.append(c)
    df = pq.read_table(p, columns=cols).to_pandas()
    n = len(df)

    # downsample indices for plotting
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    else:
        idx = np.arange(n)

    ts = df["timestamp"].to_numpy()[idx].astype(float).tolist()

    def stack(col):
        arr = np.stack(df[col].to_numpy())  # (n, dim)
        return arr[idx].astype(float).round(5).tolist()

    out = {
        "ds_id": ds_id,
        "episode_index": ep_idx,
        "num_frames": int(n),
        "fps": info.get("fps"),
        "duration_s": round(float(df["timestamp"].iloc[-1]), 2) if n else 0,
        "timestamps": [round(t, 3) for t in ts],
        "state_names": feats.get("observation.state", {}).get("names"),
        "action_names": feats.get("action", {}).get("names"),
    }
    if "observation.state" in feats:
        out["state"] = stack("observation.state")
    if "action" in feats:
        out["action"] = stack("action")

    # categorical timelines (full-resolution run-length encoded -> cheap)
    for col, key in (("observation.commander_state", "commander_segments"),
                     ("subtask", "subtask_segments")):
        if col in feats:
            vals = df[col].astype(str).tolist()
            out[key] = _segments(vals)
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/overview")
def api_overview():
    return jsonify(build_overview())


@app.route("/api/datasets")
def api_datasets():
    return jsonify(sorted(_DATASETS.keys()))


@app.route("/api/dataset")
def api_dataset():
    ds_id = request.args.get("ds", "")
    if ds_id not in _DATASETS:
        abort(404, "unknown dataset")
    info = load_info(ds_id)
    eps = load_episodes(ds_id)
    tasks = load_tasks(ds_id)
    stats = load_episode_stats(ds_id)
    cams = camera_keys(info)

    commander = None
    if _commander_cache_file(ds_id).exists():
        commander = json.loads(_commander_cache_file(ds_id).read_text())

    rows = []
    for e in eps:
        idx = e["episode_index"]
        row = {
            "episode_index": idx,
            "length": e.get("length"),
            "label": e.get("tasks"),
            "duration_s": round(e.get("length", 0) / (info.get("fps") or 1), 1),
        }
        st = stats.get(idx)
        if st and "action" in st:
            row["action_abs_max"] = round(float(np.max(np.abs(st["action"]["max"] +
                                                              st["action"]["min"]))), 3)
        if commander and commander.get("available"):
            ep_c = commander["per_episode"].get(str(idx), {})
            tot = sum(ep_c.values()) or 1
            row["teleop_frac"] = round(ep_c.get("teleop", 0) / tot, 3)
            row["intervention_frac"] = round(
                (tot - ep_c.get("inference", 0)) / tot, 3)
        rows.append(row)

    return jsonify({
        "id": ds_id,
        "info": {
            "fps": info.get("fps"), "robot_type": info.get("robot_type"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "cameras": cams, "resolution": _resolution(info["features"], cams),
        },
        "tasks": tasks, "episodes": rows,
        "commander_available": bool(commander and commander.get("available")),
        "commander_totals": (commander or {}).get("totals"),
    })


@app.route("/api/commander")
def api_commander():
    ds_id = request.args.get("ds", "")
    if ds_id not in _DATASETS:
        abort(404, "unknown dataset")
    force = request.args.get("force") == "1"
    return jsonify(compute_commander_stats(ds_id, force=force))


@app.route("/api/episode")
def api_episode():
    ds_id = request.args.get("ds", "")
    if ds_id not in _DATASETS:
        abort(404, "unknown dataset")
    ep = int(request.args.get("ep", 0))
    return jsonify(extract_trajectory(ds_id, ep))


@app.route("/api/video")
def api_video():
    ds_id = request.args.get("ds", "")
    if ds_id not in _DATASETS:
        abort(404, "unknown dataset")
    ep = int(request.args.get("ep", 0))
    cam = request.args.get("cam", "")
    if cam not in camera_keys(load_info(ds_id)):
        abort(404, "unknown camera")
    p = video_path(ds_id, ep, cam)
    if not p.exists():
        abort(404, f"video not found: {p}")
    # conditional=True enables HTTP Range support for seeking.
    return send_file(p, mimetype="video/mp4", conditional=True)


def get_lan_ip() -> str:
    """Best-effort primary (outbound) IP of this host, not 127.0.0.1."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no packets are actually sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def main():
    global ROOT, CACHE_DIR, _DATASETS, SERVER_IP, SERVER_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data5/jellyho/PFR_RSS/dataset/phase1",
                    help="directory containing LeRobot datasets")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7800)
    ap.add_argument("--cache", default=None, help="cache dir (default: <script>/.cache)")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    CACHE_DIR = Path(args.cache).resolve() if args.cache else Path(__file__).parent / ".cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _DATASETS = discover_datasets(ROOT)
    if not _DATASETS:
        raise SystemExit(f"No LeRobot datasets (meta/info.json) found under {ROOT}")

    SERVER_IP = get_lan_ip()
    SERVER_PORT = args.port
    lan_ip = SERVER_IP
    print(f"[viewer] root = {ROOT}")
    print(f"[viewer] found {len(_DATASETS)} datasets:")
    for k in _DATASETS:
        print(f"          - {k}")
    print(f"[viewer] serving on host {args.host}:{args.port}")
    print(f"[viewer]   server IP : http://{lan_ip}:{args.port}")
    print(f"[viewer]   localhost : http://localhost:{args.port}  (via SSH port-forward)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
