import html
import os
import threading
import time
from urllib.parse import quote

import cv2
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory

import exam_camera
import exam_detection
import exam_reporting
import exam_state
from exam_camera import cleanup_removed_feeds, read_feed_frame, release_all_captures, scan_cameras
from exam_config import REPORT_DIR, ensure_dirs
from exam_detection import close_detectors, detect_on_frame, ensure_detectors, init_feed_state
from exam_reporting import generate_report, record_incident, save_snapshot
from exam_state import add_event, get_candidate_meta, init_state, parse_candidate_map


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class StreamlitShim:
    def __init__(self):
        self.session_state = SessionState()


st = StreamlitShim()
exam_camera.st = st
exam_detection.st = st
exam_reporting.st = st
exam_state.st = st

app = Flask(__name__)
state_lock = threading.RLock()
bg_stop = threading.Event()
bg_thread = None


def close_resources() -> None:
    release_all_captures()
    close_detectors()


def status_issues(sig: dict) -> list[str]:
    issues = []
    if "Possible" in sig.get("mobile", "") or "Mobile risk" in sig.get("mobile", ""):
        issues.append("Phone usage suspected")
    if "Talking" in sig.get("talking", ""):
        issues.append("Talking detected")
    if "Paper" in sig.get("paper", ""):
        issues.append("Paper usage suspected")
    if "Repeated" in sig.get("head_turn", ""):
        issues.append("Head turning detected")
    if not issues:
        issues.append("No issues detected")
    return issues


def status_theme(sig: dict) -> tuple[str, str, str]:
    severity = sig.get("severity", "NORMAL")
    if severity in ("ALERT", "HIGH ALERT"):
        if "Possible" in sig.get("mobile", "") or "Mobile risk" in sig.get("mobile", ""):
            return "POSSIBLE MOBILE USAGE", "#f5a623", "#ff7f11"
        return "SUSPECTED BEHAVIOR", "#ff4d4d", "#e53935"
    if severity == "WARNING":
        return "WARNING", "#f5a623", "#f5a623"
    if severity == "OFFLINE":
        return "OFFLINE", "#b5bcc7", "#7f8ea3"
    return "NORMAL", "#53d769", "#2e7d32"


def connected_cameras(feed_sources: list[str]) -> int:
    return sum(1 for src in feed_sources if st.session_state.feed_status.get(src) == "Connected")


def update_frames(feed_sources: list[str]) -> None:
    for src in feed_sources:
        init_feed_state(src)
        st.session_state.feed_frames[src] = read_feed_frame(src)


def process_detection(feed_sources: list[str]) -> None:
    # Keep detections active for live-preview mode so UI status reflects behavior.
    if not (st.session_state.running or st.session_state.live_preview):
        return
    now = time.time()
    detection_interval = 0.22
    if now - st.session_state.last_detect_ts < detection_interval:
        return
    st.session_state.last_detect_ts = now

    ensure_detectors()
    st.session_state.tick += 1
    n = len(feed_sources)
    if n == 0:
        return

    start = st.session_state.rr_index % n
    batch = min(st.session_state.analysis_batch_size, n)
    indexes = [(start + i) % n for i in range(batch)]
    st.session_state.rr_index = (start + batch) % n
    active_alerts = 0

    for idx in indexes:
        src = feed_sources[idx]
        frame = st.session_state.feed_frames.get(src)
        if frame is None:
            continue
        if st.session_state.feed_status.get(src) != "Connected":
            st.session_state.feed_signals[src] = {
                "mobile": "No mobile signal",
                "talking": "No talking signal",
                "paper": "No paper signal",
                "head_turn": "No head-turn signal",
                "severity": "OFFLINE",
            }
            continue

        signal_text = detect_on_frame(src, frame)
        if signal_text.get("severity") not in ("ALERT", "HIGH ALERT"):
            st.session_state.feed_risk_scores[src] = max(0.0, st.session_state.feed_risk_scores.get(src, 0.0) - 0.35)
            continue

        active_alerts += 1
        st.session_state.feed_risk_scores[src] = min(100.0, st.session_state.feed_risk_scores.get(src, 0.0) + 1.8)
        last_incident = st.session_state.last_incident_ts.get(src, 0.0)
        if time.time() - last_incident > 3:
            snap = save_snapshot(frame, src)
            st.session_state.last_snapshot_ts[src] = time.time()
            record_incident(src, signal_text, snap)
            st.session_state.last_incident_ts[src] = time.time()
            add_event(f"Snapshot captured for feed {src}")
            add_event(f"Incident logged on feed {src}")

    if active_alerts > 0:
        st.session_state.risk_score = min(100.0, st.session_state.risk_score + (1.6 * active_alerts))
    else:
        st.session_state.risk_score = max(0.0, st.session_state.risk_score - 0.6)


def background_loop() -> None:
    while not bg_stop.is_set():
        with state_lock:
            feed_sources = list(st.session_state.feed_list)
            active = st.session_state.running or st.session_state.live_preview
            if active:
                update_frames(feed_sources)
            process_detection(feed_sources)
        time.sleep(0.06 if st.session_state.running else 0.12)


def ensure_started() -> None:
    global bg_thread
    with state_lock:
        if "running" not in st.session_state:
            init_state()
            ensure_dirs()
    if bg_thread is None or not bg_thread.is_alive():
        bg_stop.clear()
        bg_thread = threading.Thread(target=background_loop, daemon=True)
        bg_thread.start()


def snapshot_state() -> dict:
    with state_lock:
        feeds = list(st.session_state.feed_list)
        cam_rows = []
        for i, src in enumerate(feeds):
            sig = st.session_state.feed_signals.get(
                src,
                {"severity": "NORMAL", "mobile": "No mobile signal", "talking": "No talking signal", "paper": "No paper signal", "head_turn": "No head-turn signal"},
            )
            status_text, status_color, border_color = status_theme(sig)
            meta = get_candidate_meta(src)
            cam_rows.append(
                {
                    "index": i + 1,
                    "source": src,
                    "candidate": meta.get("candidate", f"Student_{i+1}"),
                    "status_text": status_text,
                    "status_color": status_color,
                    "border_color": border_color,
                    "issues": status_issues(sig)[:3],
                    "overlay": " | ".join([x for x in status_issues(sig) if x != "No issues detected"][:2]),
                    "frame_url": "/frame?source=" + quote(src, safe=""),
                }
            )

        reports = []
        if st.session_state.report_ready:
            for source in feeds[:4]:
                rep = st.session_state.per_camera_reports.get(source)
                if rep:
                    reports.append({"label": f"Download Report CAM {source}", "file": rep["txt_name"]})

        return {
            "running": st.session_state.running,
            "connected": connected_cameras(feeds),
            "feed_count": len(feeds),
            "risk_score": int(st.session_state.risk_score),
            "incidents": len(st.session_state.incidents),
            "feeds_raw": st.session_state.feeds_raw,
            "candidate_meta_raw": st.session_state.candidate_meta_raw,
            "analysis_batch_size": st.session_state.analysis_batch_size,
            "grid_cols": st.session_state.grid_cols,
            "live_preview": st.session_state.live_preview,
            "rows": cam_rows,
            "events": st.session_state.events[:12],
            "reports": reports,
        }


PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Exam Proctor - Multi Feed (Flask)</title>
  <style>
    :root {
      --bg-main: #0d1b34; --line: #223f6b; --text: #e9f0ff; --muted: #b5c3de;
    }
    body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; color: var(--text);
      background: radial-gradient(1200px 700px at 10% -10%, #1e3761 0%, #0d1b34 50%, #0a1730 100%); }
    .wrap { max-width: 1500px; margin: 0 auto; padding: 12px; }
    .top-title { background: linear-gradient(180deg, #18345f, #132a4c); border: 1px solid var(--line); border-radius: 14px; text-align: center; padding: 12px; font-size: 2.2rem; font-weight: 800; }
    .top-strip { margin-top: 10px; background: linear-gradient(180deg, #142b50, #112544); border: 1px solid var(--line); border-radius: 12px; padding: 12px 18px; display: flex; justify-content: space-between; font-size: 1.25rem; font-weight: 700; }
    .panel { margin-top: 12px; background: linear-gradient(180deg, #10213f, #0f203c); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
    .cfg { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    textarea, select { width: 100%; box-sizing: border-box; border-radius: 8px; border: 1px solid #355584; background: #0e2344; color: #e9f0ff; padding: 8px; }
    .btns { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    button { border: 1px solid #355584; border-radius: 8px; background: #11315f; color: #fff; padding: 8px 10px; cursor: pointer; }
    .main { margin-top: 12px; display: grid; grid-template-columns: 1fr 3fr; gap: 12px; }
    .status-board { background: linear-gradient(180deg, #112544, #0d1d37); border: 1px solid var(--line); border-radius: 10px; padding: 14px; min-height: 740px; }
    .status-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 10px; }
    .cam-entry { border-top: 1px solid #274a78; padding: 10px 0; }
    .cam-name { font-size: 1.1rem; font-weight: 700; }
    .cam-state { font-size: 1.1rem; font-weight: 800; margin: 3px 0 6px 0; }
    .issue-line { color: var(--muted); font-size: 0.95rem; margin: 2px 0; }
    .feeds { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .feed-card { position: relative; background: #0f2342; border: 5px solid #2e7d32; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.35); }
    .feed-img { width: 100%; height: 340px; object-fit: cover; display: block; }
    .overlay-note { position: absolute; left: 16px; top: 14px; background: rgba(195, 23, 23, 0.82); color: #fff; font-weight: 800; font-size: 1rem; border-radius: 7px; padding: 6px 12px; }
    .feed-footer { background: #0d1f3a; font-size: 1.1rem; font-weight: 700; text-align: center; padding: 7px 10px; }
    .action-row { margin-top: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .compact-note { margin-left: auto; font-size: 1rem; }
    .logs { margin-top: 12px; background: #0f203c; border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
    .reports { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    @media (max-width: 980px) {
      .cfg { grid-template-columns: 1fr; }
      .main { grid-template-columns: 1fr; }
      .feeds { grid-template-columns: 1fr; }
      .compact-note { margin-left: 0; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top-title">AI Exam Monitoring Dashboard</div>
    <div class="top-strip">
      <div>Connected Cameras: <span id="connected" style="color:#58e676">0</span></div>
      <div>Monitoring Status: <span id="monitorState" style="color:#ffbe55">STOPPED</span></div>
    </div>

    <div class="panel">
      <h3 style="margin-top:0">Configuration</h3>
      <div class="cfg">
        <div>
          <label>Feed sources (one per line, index or rtsp/http)</label>
          <textarea id="feedsRaw" rows="7"></textarea>
          <div class="btns">
            <button onclick="applyConfig()">Apply Feed List</button>
            <button onclick="scanCams()">Scan Local Cameras</button>
          </div>
        </div>
        <div>
          <label>Camera metadata (camera,candidate,resume_link)</label>
          <textarea id="metaRaw" rows="7"></textarea>
          <div class="btns">
            <select id="gridCols"><option>2</option><option selected>3</option><option>4</option></select>
            <select id="batchSize"><option>1</option><option selected>2</option><option>3</option><option>4</option></select>
            <label><input id="livePreview" type="checkbox" checked /> Live preview when stopped</label>
          </div>
        </div>
      </div>
    </div>

    <div class="main">
      <div class="status-board">
        <div class="status-title">Camera Status</div>
        <div id="statusBoard"></div>
      </div>
      <div class="panel">
        <div id="feedsGrid" class="feeds"></div>
      </div>
    </div>

    <div class="panel">
      <div class="action-row">
        <button id="startStopBtn" onclick="toggleRun()">Start Monitoring</button>
        <button onclick="generateReport()">Generate Report</button>
        <button onclick="resetRisk()">Reset Risk</button>
        <div class="compact-note"><b>Risk:</b> <span id="risk">0</span> | <b>Incidents:</b> <span id="incidents">0</span></div>
      </div>
      <div class="reports" id="reports"></div>
    </div>

    <div class="logs">
      <h3 style="margin-top:0">Recent Events</h3>
      <div id="events"></div>
    </div>
  </div>

  <script>
    async function post(url, body={}) {
      await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    }

    async function applyConfig() {
      await post("/api/config", {
        feeds_raw: document.getElementById("feedsRaw").value,
        candidate_meta_raw: document.getElementById("metaRaw").value,
        grid_cols: parseInt(document.getElementById("gridCols").value, 10),
        analysis_batch_size: parseInt(document.getElementById("batchSize").value, 10),
        live_preview: document.getElementById("livePreview").checked
      });
      await refresh();
    }

    async function scanCams() { await post("/api/scan_cameras"); await refresh(); }
    async function toggleRun() {
      const btn = document.getElementById("startStopBtn");
      await post(btn.dataset.running === "1" ? "/api/stop" : "/api/start");
      await refresh();
    }
    async function generateReport() { await post("/api/generate_report"); await refresh(); }
    async function resetRisk() { await post("/api/reset_risk"); await refresh(); }

    function esc(s) {
      return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
    }

    async function refresh() {
      const res = await fetch("/api/state");
      const data = await res.json();

      document.getElementById("connected").textContent = data.connected;
      document.getElementById("monitorState").textContent = data.running ? "ACTIVE" : "STOPPED";
      document.getElementById("monitorState").style.color = data.running ? "#58e676" : "#ffbe55";
      const btn = document.getElementById("startStopBtn");
      btn.textContent = data.running ? "Stop Monitoring" : "Start Monitoring";
      btn.dataset.running = data.running ? "1" : "0";
      document.getElementById("risk").textContent = data.risk_score;
      document.getElementById("incidents").textContent = data.incidents;

      document.getElementById("feedsRaw").value = data.feeds_raw;
      document.getElementById("metaRaw").value = data.candidate_meta_raw;
      document.getElementById("gridCols").value = String(data.grid_cols);
      document.getElementById("batchSize").value = String(data.analysis_batch_size);
      document.getElementById("livePreview").checked = !!data.live_preview;

      const statusBoard = document.getElementById("statusBoard");
      statusBoard.innerHTML = data.rows.map(r => `
        <div class="cam-entry">
          <div class="cam-name">Camera ${r.index} - ${esc(r.candidate)}</div>
          <div class="cam-state" style="color:${r.status_color}">STATUS: ${esc(r.status_text)}</div>
          ${r.issues.map(i => `<div class="issue-line">- ${esc(i)}</div>`).join("")}
        </div>
      `).join("");

      renderFeeds(data.rows);

      const events = document.getElementById("events");
      events.innerHTML = data.events.length ? data.events.map(e => `<div>- ${esc(e)}</div>`).join("") : "<div>No events yet</div>";

      const reports = document.getElementById("reports");
      reports.innerHTML = data.reports.map(r => `<a href="/download/${encodeURIComponent(r.file)}"><button>${esc(r.label)}</button></a>`).join("");
    }

    function feedCardId(source) {
      return "feed_" + encodeURIComponent(source).replaceAll("%", "_");
    }

    function renderFeeds(rows) {
      const grid = document.getElementById("feedsGrid");
      const wanted = new Set(rows.map(r => r.source));

      for (const node of Array.from(grid.children)) {
        if (!wanted.has(node.dataset.source)) {
          node.remove();
        }
      }

      for (const r of rows) {
        const id = feedCardId(r.source);
        let card = document.getElementById(id);
        if (!card) {
          card = document.createElement("div");
          card.id = id;
          card.className = "feed-card";
          card.dataset.source = r.source;

          const img = document.createElement("img");
          img.className = "feed-img";
          img.src = "/stream?source=" + encodeURIComponent(r.source);
          card.appendChild(img);

          const overlay = document.createElement("div");
          overlay.className = "overlay-note";
          overlay.style.display = "none";
          card.appendChild(overlay);

          const footer = document.createElement("div");
          footer.className = "feed-footer";
          card.appendChild(footer);

          grid.appendChild(card);
        }

        card.style.borderColor = r.border_color;
        const overlay = card.querySelector(".overlay-note");
        if (r.overlay) {
          overlay.textContent = "- " + r.overlay;
          overlay.style.display = "block";
        } else {
          overlay.textContent = "";
          overlay.style.display = "none";
        }
        const footer = card.querySelector(".feed-footer");
        footer.textContent = "Cam " + r.index + " | Status: " + r.status_text;
      }
    }

    setInterval(refresh, 700);
    refresh();
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    ensure_started()
    return render_template_string(PAGE_HTML)


@app.route("/api/state")
def api_state():
    ensure_started()
    return jsonify(snapshot_state())


@app.route("/api/config", methods=["POST"])
def api_config():
    ensure_started()
    payload = request.get_json(silent=True) or {}
    with state_lock:
        st.session_state.feeds_raw = str(payload.get("feeds_raw", st.session_state.feeds_raw))
        parsed = [x.strip() for x in st.session_state.feeds_raw.splitlines() if x.strip()]
        st.session_state.feed_list = parsed or ["0"]
        st.session_state.candidate_meta_raw = str(payload.get("candidate_meta_raw", st.session_state.candidate_meta_raw))
        st.session_state.candidate_map = parse_candidate_map(st.session_state.candidate_meta_raw)
        st.session_state.grid_cols = int(payload.get("grid_cols", st.session_state.grid_cols))
        st.session_state.analysis_batch_size = int(payload.get("analysis_batch_size", st.session_state.analysis_batch_size))
        st.session_state.live_preview = bool(payload.get("live_preview", st.session_state.live_preview))
        cleanup_removed_feeds()
        add_event(f"Feed list updated: {st.session_state.feed_list}")
    return jsonify({"ok": True})


@app.route("/api/scan_cameras", methods=["POST"])
def api_scan_cameras():
    ensure_started()
    found = scan_cameras()
    with state_lock:
        st.session_state.feed_list = found or ["0"]
        st.session_state.feeds_raw = "\n".join(st.session_state.feed_list)
        cleanup_removed_feeds()
        add_event(f"Local camera scan: {st.session_state.feed_list}")
    return jsonify({"ok": True, "found": found})


@app.route("/api/start", methods=["POST"])
def api_start():
    ensure_started()
    with state_lock:
        st.session_state.running = True
        add_event("Monitoring started")
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ensure_started()
    with state_lock:
        st.session_state.running = False
        close_resources()
        generate_report()
        add_event("Monitoring stopped")
        add_event("Per-camera reports generated")
    return jsonify({"ok": True})


@app.route("/api/generate_report", methods=["POST"])
def api_generate_report():
    ensure_started()
    with state_lock:
        generate_report()
        add_event("Final report generated")
    return jsonify({"ok": True})


@app.route("/api/reset_risk", methods=["POST"])
def api_reset_risk():
    ensure_started()
    with state_lock:
        st.session_state.risk_score = 0.0
        st.session_state.incidents = []
        st.session_state.feed_risk_scores = {}
        add_event("Risk and incidents reset")
    return jsonify({"ok": True})


@app.route("/frame")
def frame():
    ensure_started()
    source = request.args.get("source", "").strip()
    if not source:
        return Response(status=400)
    with state_lock:
        frame_bgr = st.session_state.feed_frames.get(source)
        if frame_bgr is None:
            frame_bgr = exam_camera.offline_frame("No Frame")
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return Response(status=500)
    return Response(encoded.tobytes(), mimetype="image/jpeg")


@app.route("/stream")
def stream():
    ensure_started()
    source = request.args.get("source", "").strip()
    if not source:
        return Response(status=400)

    def gen():
        while True:
            with state_lock:
                frame_bgr = st.session_state.feed_frames.get(source)
                if frame_bgr is None:
                    frame_bgr = exam_camera.offline_frame("No Frame")
            ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                data = encoded.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                )
            time.sleep(0.08)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/download/<path:name>")
def download(name: str):
    ensure_started()
    return send_from_directory(REPORT_DIR, name, as_attachment=True)


@app.route("/health")
def health():
    ensure_started()
    return jsonify({"ok": True})


def main():
    ensure_started()
    app.run(host="0.0.0.0", port=8502, debug=False, threaded=True)


if __name__ == "__main__":
    main()
