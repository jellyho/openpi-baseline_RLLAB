"use strict";

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //
const $ = (sel, root = document) => root.querySelector(sel);
const app = $("#app");

const fmt = (n) => (n ?? 0).toLocaleString("en-US");
const fmtf = (n, d = 1) => (n ?? 0).toLocaleString("en-US", { maximumFractionDigits: d });

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function tagClass(label = "") {
  const l = label.toLowerCase();
  if (l.includes("fail")) return "tag-fail";
  if (l.includes("hil") || l.includes("correct")) return "tag-hil";
  return "tag-expert";
}

// commander_state -> colour (HIL semantics)
const CMD_COLORS = {
  inference: "#4493f8",   // policy in control
  teleop: "#a371f7",      // human intervention
  pre_teleop: "#d29922",  // about to take over
  align: "#79c0ff",       // aligning before handoff
  record: "#3fb950",      // recording
  restore: "#db61a2",     // restoring scene
  idle: "#6e7681",
};
const cmdColor = (k) => CMD_COLORS[k] || "#6e7681";

const PLOT_LAYOUT = {
  paper_bgcolor: "#161b22", plot_bgcolor: "#161b22",
  font: { color: "#8b949e", size: 11 },
  margin: { l: 45, r: 10, t: 10, b: 30 },
  xaxis: { gridcolor: "#2a313c", zerolinecolor: "#2a313c", title: "time (s)" },
  yaxis: { gridcolor: "#2a313c", zerolinecolor: "#2a313c" },
  legend: { orientation: "h", font: { size: 9 }, y: -0.25 },
  showlegend: true,
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// arm-colored palette: left arm blue-ish, right arm orange-ish, gripper highlighted
function dimColor(i, dim) {
  const half = Math.floor(dim / 2);
  const isRight = i >= half;
  const within = isRight ? i - half : i;
  const isGripper = within === half - 1;
  const base = isRight ? [255, 159, 64] : [64, 147, 248];
  const shade = 0.55 + 0.45 * (within / Math.max(1, half - 1));
  if (isGripper) return isRight ? "#f85149" : "#3fb950";
  return `rgb(${base.map((c) => Math.round(c * shade)).join(",")})`;
}

// --------------------------------------------------------------------------- //
// router
// --------------------------------------------------------------------------- //
window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);

function route() {
  const h = location.hash.replace(/^#\/?/, "");
  const parts = h.split("/").filter(Boolean);
  if (parts[0] === "ds") {
    const ds = decodeURIComponent(parts.slice(1, parts.length - 1).join("/") || parts[1]);
    // episode route: ds/<id...>/ep/<n>
    const epIdx = parts.indexOf("ep");
    if (epIdx !== -1) {
      const dsId = decodeURIComponent(parts.slice(1, epIdx).join("/"));
      renderEpisode(dsId, parseInt(parts[epIdx + 1], 10));
    } else {
      const dsId = decodeURIComponent(parts.slice(1).join("/"));
      renderDataset(dsId);
    }
  } else {
    renderOverview();
  }
}

function crumbs(items) {
  $("#crumbs").innerHTML = items
    .map((it, i) => (i === items.length - 1
      ? `<b>${it.t}</b>`
      : `<a href="${it.h}">${it.t}</a>`))
    .join(" &nbsp;›&nbsp; ");
}

// --------------------------------------------------------------------------- //
// OVERVIEW
// --------------------------------------------------------------------------- //
async function renderOverview() {
  crumbs([{ t: "Overview" }]);
  app.innerHTML = `<div class="loading">Loading overview…</div>`;
  let ov;
  try { ov = await getJSON("/api/overview"); }
  catch (e) { app.innerHTML = `<div class="err">${e.message}</div>`; return; }

  const srv = ov.server_ip ? `http://${ov.server_ip}:${ov.server_port}` : "";
  $("#rootPath").innerHTML = (srv ? `<span title="server address">🌐 ${srv}</span><br>` : "")
    + `<span title="dataset root">${ov.root}</span>`;
  const t = ov.totals;

  let html = `
    <div class="cards">
      <div class="card"><div class="big">${fmt(t.datasets)}</div><div class="lbl">Datasets</div></div>
      <div class="card"><div class="big">${fmt(t.episodes)}</div><div class="lbl">Episodes</div></div>
      <div class="card"><div class="big">${fmt(t.frames)}</div><div class="lbl">Frames</div></div>
      <div class="card"><div class="big">${fmtf(t.duration_min / 60, 1)} h</div><div class="lbl">Total duration</div></div>
    </div>
    <div class="grid2">
      <div class="panel"><h3>Episodes per dataset</h3><div id="chartEp" style="height:300px"></div></div>
      <div class="panel"><h3>Frames per dataset</h3><div id="chartFr" style="height:300px"></div></div>
    </div>
  `;

  // group datasets
  const byGroup = {};
  ov.datasets.forEach((d) => { (byGroup[d.group] ??= []).push(d); });

  for (const [g, subs] of Object.entries(byGroup)) {
    const ep = subs.reduce((a, d) => a + d.total_episodes, 0);
    const fr = subs.reduce((a, d) => a + d.total_frames, 0);
    const mn = subs.reduce((a, d) => a + d.duration_min, 0);
    const inst = subs[0].task_instructions?.[0] || "";
    html += `<div class="group"><h2>${g}
        <small>${fmt(ep)} eps · ${fmt(fr)} frames · ${fmtf(mn, 0)} min</small></h2>`;
    if (inst) html += `<div class="muted" style="margin:-6px 0 6px">📋 ${inst}</div>`;
    html += `<div class="subset-row" style="cursor:default;background:transparent;border:none;color:var(--muted);font-size:12px">
        <div>subset</div><div class="num">episodes</div><div class="num">frames</div>
        <div class="num">duration</div><div class="num">len (med)</div><div>cameras / codec</div></div>`;
    for (const d of subs) {
      const ls = d.length_stats || {};
      const medS = ls.median ? (ls.median / d.fps).toFixed(0) + "s" : "—";
      const cams = d.cameras.map((c) => `<span class="chip cam">${c.replace("observation.images.", "")}</span>`).join(" ");
      html += `<div class="subset-row" onclick="location.hash='#/ds/${encodeURIComponent(d.id)}'">
          <div class="name ${tagClass(d.subset)}">${d.subset || d.id}</div>
          <div class="num">${fmt(d.total_episodes)}</div>
          <div class="num">${fmt(d.total_frames)}</div>
          <div class="num">${fmtf(d.duration_min, 0)}m</div>
          <div class="num">${medS}</div>
          <div class="sub">${cams} <span class="chip">${d.video_codec || "?"}</span></div>
        </div>`;
    }
    html += `</div>`;
  }
  app.innerHTML = html;

  // charts: stacked by group color, one bar per dataset
  const names = ov.datasets.map((d) => d.id);
  const colorFor = (id) => {
    if (id.includes("fail")) return "#f85149";
    if (id.includes("hil") || id.includes("success")) return "#a371f7";
    return "#3fb950";
  };
  const colors = ov.datasets.map((d) => colorFor(d.id));
  Plotly.newPlot("chartEp",
    [{ type: "bar", x: names, y: ov.datasets.map((d) => d.total_episodes), marker: { color: colors } }],
    { ...PLOT_LAYOUT, showlegend: false, margin: { l: 50, r: 10, t: 10, b: 110 },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "", tickangle: -35, tickfont: { size: 9 } },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "episodes" } }, PLOT_CONFIG);
  Plotly.newPlot("chartFr",
    [{ type: "bar", x: names, y: ov.datasets.map((d) => d.total_frames), marker: { color: colors } }],
    { ...PLOT_LAYOUT, showlegend: false, margin: { l: 60, r: 10, t: 10, b: 110 },
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "", tickangle: -35, tickfont: { size: 9 } },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "frames" } }, PLOT_CONFIG);
}

// --------------------------------------------------------------------------- //
// DATASET DETAIL
// --------------------------------------------------------------------------- //
let _tableState = { rows: [], sort: "episode_index", asc: true };

async function renderDataset(dsId) {
  crumbs([{ t: "Overview", h: "#/" }, { t: dsId }]);
  app.innerHTML = `<div class="loading">Loading ${dsId}…</div>`;
  let d;
  try { d = await getJSON(`/api/dataset?ds=${encodeURIComponent(dsId)}`); }
  catch (e) { app.innerHTML = `<div class="err">${e.message}</div>`; return; }

  const inf = d.info;
  const cams = inf.cameras.map((c) => `<span class="chip cam">${c.replace("observation.images.", "")}</span>`).join(" ");
  const inst = Object.values(d.tasks)[0] || "—";

  app.innerHTML = `
    <h2>${dsId}</h2>
    <div class="panel kv">
      <div class="k">Task instruction</div><div>${inst}</div>
      <div class="k">Robot / fps</div><div>${inf.robot_type} · ${inf.fps} fps</div>
      <div class="k">Episodes / frames</div><div>${fmt(inf.total_episodes)} · ${fmt(inf.total_frames)}</div>
      <div class="k">Cameras</div><div>${cams} <span class="muted">@ ${inf.resolution}</span></div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0">Human intervention (commander_state)</h3>
      <div id="cmdBox"><button class="btn" id="cmdBtn">Compute intervention stats</button>
        <span class="muted" style="margin-left:10px">scans every episode parquet (cached after first run)</span></div>
    </div>

    <div class="panel">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <h3 style="margin:0">Episodes (${d.episodes.length})</h3>
        <input type="text" id="epFilter" placeholder="filter by label…" style="margin-left:auto" />
      </div>
      <div style="overflow:auto;max-height:65vh"><table id="epTable"></table></div>
    </div>
  `;

  _tableState = { rows: d.episodes, sort: "episode_index", asc: true, dsId, commander: d.commander_available };
  renderEpTable();
  $("#epFilter").addEventListener("input", renderEpTable);

  if (d.commander_available && d.commander_totals) {
    renderCommanderDonut(d.commander_totals);
  } else {
    $("#cmdBtn").addEventListener("click", () => computeCommander(dsId));
  }
}

function renderEpTable() {
  const st = _tableState;
  const q = ($("#epFilter")?.value || "").toLowerCase();
  let rows = st.rows.filter((r) => !q || String(r.label || "").toLowerCase().includes(q));
  rows = rows.slice().sort((a, b) => {
    const va = a[st.sort], vb = b[st.sort];
    const c = (va > vb ? 1 : va < vb ? -1 : 0);
    return st.asc ? c : -c;
  });

  const hasCmd = st.commander;
  const cols = [
    ["episode_index", "ep", true],
    ["length", "frames", true],
    ["duration_s", "dur (s)", true],
    ["label", "label", false],
  ];
  if (hasCmd) {
    cols.push(["teleop_frac", "teleop %", true]);
    cols.push(["intervention_frac", "interv. %", true]);
  }

  const th = cols.map(([k, lbl, num]) =>
    `<th class="${num ? "num " : ""}${st.sort === k ? "sorted " + (st.asc ? "asc" : "") : ""}"
        onclick="sortBy('${k}')">${lbl}</th>`).join("");

  const body = rows.map((r) => {
    const tds = cols.map(([k, , num]) => {
      let v = r[k];
      if (k === "label") return `<td class="${tagClass(v)}">${v ?? ""}</td>`;
      if ((k === "teleop_frac" || k === "intervention_frac")) {
        v = v == null ? "—" : (v * 100).toFixed(0) + "%";
      }
      return `<td class="${num ? "num" : ""}">${v ?? ""}</td>`;
    }).join("");
    return `<tr style="cursor:pointer" onclick="location.hash='#/ds/${encodeURIComponent(st.dsId)}/ep/${r.episode_index}'">${tds}</tr>`;
  }).join("");

  $("#epTable").innerHTML = `<thead><tr>${th}</tr></thead><tbody>${body}</tbody>`;
}
window.sortBy = (k) => {
  const st = _tableState;
  st.asc = st.sort === k ? !st.asc : true;
  st.sort = k;
  renderEpTable();
};

async function computeCommander(dsId) {
  const box = $("#cmdBox");
  box.innerHTML = `<span class="spinner">Scanning episodes… (this can take a minute for large datasets)</span>`;
  try {
    const c = await getJSON(`/api/commander?ds=${encodeURIComponent(dsId)}`);
    if (!c.available) { box.innerHTML = `<span class="muted">No commander_state in this dataset.</span>`; return; }
    box.innerHTML = `<div id="cmdDonut" style="height:240px"></div>`;
    renderCommanderDonut(c.totals);
    // merge per-episode fractions into the table
    _tableState.commander = true;
    _tableState.rows.forEach((r) => {
      const e = c.per_episode[String(r.episode_index)] || {};
      const tot = Object.values(e).reduce((a, b) => a + b, 0) || 1;
      r.teleop_frac = (e.teleop || 0) / tot;
      r.intervention_frac = (tot - (e.inference || 0)) / tot;
    });
    renderEpTable();
  } catch (e) { box.innerHTML = `<span class="err">${e.message}</span>`; }
}

function renderCommanderDonut(totals) {
  const box = $("#cmdBox");
  if (!$("#cmdDonut")) box.innerHTML = `<div id="cmdDonut" style="height:240px"></div>`;
  const keys = Object.keys(totals);
  const total = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
  const teleop = totals.teleop || 0;
  Plotly.newPlot("cmdDonut", [{
    type: "pie", hole: 0.6, labels: keys, values: keys.map((k) => totals[k]),
    marker: { colors: keys.map(cmdColor) }, textinfo: "label+percent",
    sort: false,
  }], {
    ...PLOT_LAYOUT, showlegend: true, margin: { l: 10, r: 10, t: 10, b: 10 },
    annotations: [{ text: `${(teleop / total * 100).toFixed(1)}%<br>teleop`, showarrow: false, font: { size: 16, color: "#e6edf3" } }],
  }, PLOT_CONFIG);
}

// --------------------------------------------------------------------------- //
// EPISODE VIEWER
// --------------------------------------------------------------------------- //
let _epState = null;

async function renderEpisode(dsId, ep) {
  crumbs([{ t: "Overview", h: "#/" },
          { t: dsId, h: `#/ds/${encodeURIComponent(dsId)}` },
          { t: `episode ${ep}` }]);
  app.innerHTML = `<div class="loading">Loading episode ${ep}…</div>`;

  let traj, dsInfo;
  try {
    [traj, dsInfo] = await Promise.all([
      getJSON(`/api/episode?ds=${encodeURIComponent(dsId)}&ep=${ep}`),
      getJSON(`/api/dataset?ds=${encodeURIComponent(dsId)}`),
    ]);
  } catch (e) { app.innerHTML = `<div class="err">${e.message}</div>`; return; }

  const cams = dsInfo.info.cameras;
  const eps = dsInfo.episodes.map((r) => r.episode_index).sort((a, b) => a - b);
  const pos = eps.indexOf(ep);
  const prev = pos > 0 ? eps[pos - 1] : null;
  const next = pos < eps.length - 1 ? eps[pos + 1] : null;
  const nav = (e, lbl) => e == null
    ? `<button class="btn ghost" disabled>${lbl}</button>`
    : `<button class="btn ghost" onclick="location.hash='#/ds/${encodeURIComponent(dsId)}/ep/${e}'">${lbl}</button>`;

  app.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
      <h2 style="margin:0">${dsId} · episode ${ep}</h2>
      <span class="muted">${traj.num_frames} frames · ${traj.duration_s}s · ${traj.fps} fps</span>
      <span style="margin-left:auto;display:flex;gap:8px">${nav(prev, "‹ prev")}${nav(next, "next ›")}</span>
    </div>

    <div class="videos">
      ${cams.map((c) => `
        <div class="vidwrap">
          <div class="cap">${c.replace("observation.images.", "")}</div>
          <video data-cam="${c}" preload="auto" playsinline muted
                 src="/api/video?ds=${encodeURIComponent(dsId)}&ep=${ep}&cam=${encodeURIComponent(c)}"></video>
        </div>`).join("")}
    </div>

    <div class="controls">
      <button class="btn" id="playBtn">▶ Play</button>
      <input type="range" id="seek" min="0" max="1000" value="0" step="1" />
      <span class="time" id="timeLbl">0.00 / ${traj.duration_s}s</span>
      <label class="muted" style="font-size:12px"><input type="checkbox" id="loopChk" checked> loop</label>
    </div>

    ${traj.commander_segments ? `
      <h3>commander_state</h3>
      <div class="legend" id="cmdLegend"></div>
      <div class="timeline" id="cmdTimeline"></div>` : ""}
    ${(traj.subtask_segments && traj.subtask_segments.length > 1) ? `
      <h3>subtask</h3>
      <div class="timeline" id="subTimeline"></div>` : ""}

    <div class="grid2" style="margin-top:16px">
      <div class="panel"><h3 style="margin-top:0">observation.state</h3><div id="statePlot" style="height:300px"></div></div>
      <div class="panel"><h3 style="margin-top:0">action</h3><div id="actionPlot" style="height:300px"></div></div>
    </div>
  `;

  buildPlots(traj);
  buildTimelines(traj);
  setupSync(traj);
}

function buildPlots(traj) {
  const mk = (key, names) => {
    if (!traj[key]) return;
    const dim = traj[key][0].length;
    const x = traj.timestamps;
    const traces = [];
    for (let i = 0; i < dim; i++) {
      traces.push({
        type: "scattergl", mode: "lines",
        name: (names && names[i]) ? names[i] : `${key}[${i}]`,
        x, y: traj[key].map((row) => row[i]),
        line: { width: 1.2, color: dimColor(i, dim) },
      });
    }
    const div = key === "state" ? "statePlot" : "actionPlot";
    Plotly.newPlot(div, traces, {
      ...PLOT_LAYOUT,
      shapes: [cursorShape(0)],
    }, PLOT_CONFIG);
  };
  mk("state", traj.state_names);
  mk("action", traj.action_names);
}

function cursorShape(t) {
  return { type: "line", x0: t, x1: t, yref: "paper", y0: 0, y1: 1,
           line: { color: "#fff", width: 1.5, dash: "dot" } };
}

function buildTimelines(traj) {
  const dur = traj.duration_s || 1;
  const fps = traj.fps || 60;
  const fillSegs = (elId, segs, colorFn) => {
    const el = document.getElementById(elId);
    if (!el || !segs) return;
    const total = segs[segs.length - 1].end;
    el.innerHTML = segs.map((s) => {
      const left = (s.start / total) * 100;
      const w = ((s.end - s.start) / total) * 100;
      return `<div class="seg" title="${s.label} (${s.start}-${s.end})"
                style="left:${left}%;width:${w}%;background:${colorFn(s.label)}"></div>`;
    }).join("") + `<div class="cursor" style="left:0%"></div>`;
  };
  fillSegs("cmdTimeline", traj.commander_segments, cmdColor);
  // subtask: cycle a palette
  const pal = ["#4493f8", "#3fb950", "#d29922", "#a371f7", "#f85149", "#79c0ff"];
  const seen = {};
  let ci = 0;
  fillSegs("subTimeline", traj.subtask_segments, (l) => (seen[l] ??= pal[ci++ % pal.length]));

  // legend for commander
  if (traj.commander_segments) {
    const labels = [...new Set(traj.commander_segments.map((s) => s.label))];
    $("#cmdLegend").innerHTML = labels.map((l) =>
      `<span><i style="background:${cmdColor(l)}"></i>${l}</span>`).join("");
  }
}

function setupSync(traj) {
  const videos = [...document.querySelectorAll("video[data-cam]")];
  const master = videos[0];
  const seek = $("#seek");
  const timeLbl = $("#timeLbl");
  const playBtn = $("#playBtn");
  const loopChk = $("#loopChk");
  const dur = traj.duration_s || (traj.num_frames / (traj.fps || 60));
  let dragging = false;

  const setAll = (t) => videos.forEach((v) => { if (Math.abs(v.currentTime - t) > 0.05) v.currentTime = t; });

  playBtn.onclick = () => {
    if (master.paused) { videos.forEach((v) => v.play()); playBtn.textContent = "❚❚ Pause"; }
    else { videos.forEach((v) => v.pause()); playBtn.textContent = "▶ Play"; }
  };

  seek.addEventListener("input", () => {
    dragging = true;
    const t = (seek.value / 1000) * dur;
    setAll(t);
    updateCursor(t);
  });
  seek.addEventListener("change", () => { dragging = false; });

  master.addEventListener("ended", () => {
    if (loopChk.checked) { setAll(0); videos.forEach((v) => v.play()); }
    else playBtn.textContent = "▶ Play";
  });

  let raf;
  const tick = () => {
    const t = master.currentTime;
    if (!dragging) {
      seek.value = Math.min(1000, (t / dur) * 1000);
      updateCursor(t);
    }
    timeLbl.textContent = `${t.toFixed(2)} / ${dur.toFixed(2)}s`;
    raf = requestAnimationFrame(tick);
  };
  cancelAnimationFrame(window.__epRaf);
  raf = requestAnimationFrame(tick);
  window.__epRaf = raf;

  function updateCursor(t) {
    const frac = Math.max(0, Math.min(1, t / dur));
    ["statePlot", "actionPlot"].forEach((id) => {
      if (document.getElementById(id))
        Plotly.relayout(id, { "shapes[0].x0": t, "shapes[0].x1": t });
    });
    document.querySelectorAll(".timeline .cursor").forEach((c) => {
      c.style.left = (frac * 100) + "%";
    });
  }

  // click on plots to seek
  ["statePlot", "actionPlot"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.on("plotly_click", (d) => {
      if (d.points && d.points[0]) { setAll(d.points[0].x); updateCursor(d.points[0].x); }
    });
  });
}
