# LeRobot Dataset Web Viewer

A lightweight web viewer for inspecting LeRobot (v2.x) datasets: a metadata
dashboard, an episode browser, synced multi-camera video playback, and
state/action trajectory plots with a HIL `commander_state` timeline.

Built for `/data5/jellyho/PFR_RSS/dataset/phase1` but works on any directory
tree containing LeRobot datasets (anything with a `meta/info.json`).

## Run

```bash
cd tools/lerobot_viewer
./run.sh                                   # root=.../phase1, port=7800
./run.sh /path/to/other/dataset 7801       # custom root + port
```

Or directly:

```bash
/data5/jellyho/miniconda3/envs/openpi/bin/python app.py \
    --root /data5/jellyho/PFR_RSS/dataset/phase1 --port 7800
```

The server runs on the remote machine. From your laptop, forward the port:

```bash
ssh -N -L 7800:localhost:7800 jellyho@<host>
# then open http://localhost:7800
```

## Views

- **Overview** — totals (datasets / episodes / frames / hours), per-dataset
  cards grouped by task, episode/frame bar charts, episode-length stats,
  cameras + codec.
- **Dataset** — task instruction, fps/robot, camera/resolution, a sortable &
  filterable episode table, and a **Compute intervention stats** button that
  scans `observation.commander_state` across all episodes (cached to `.cache/`)
  and shows a teleop/inference donut + per-episode intervention %.
- **Episode** — 3 synced camera videos, a shared play/seek bar, a
  `commander_state` timeline ribbon (HIL takeover segments), and 14-dim
  state/action plots with a cursor synced to playback. Click a plot to seek.

## Notes

- **Video codec is AV1.** Modern Chrome / Edge / Firefox play it natively;
  Safari may not. If playback fails, use Chrome/Firefox (or ask for an
  on-the-fly ffmpeg transcode fallback to be added).
- Trajectories are downsampled to ~1500 points for plotting; categorical
  timelines (`commander_state`, `subtask`) are full-resolution run-length
  encoded.
- `subtask` is currently a `"TODO"` placeholder in the data, so its timeline
  only appears when more than one value is present.
- The intervention scan result is cached per dataset under `.cache/`. Re-run
  with `?force=1` on `/api/commander` to recompute.

## Layout

```
app.py              Flask backend (discovery, aggregation, parquet/video serving)
static/index.html   shell
static/app.js       SPA (overview / dataset / episode views)
static/style.css    styling
run.sh              launcher (uses the openpi conda env python)
.cache/             cached commander_state scans (gitignored)
```
