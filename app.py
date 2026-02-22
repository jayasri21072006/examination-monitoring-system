import base64
import html
import os
import time


import cv2
import streamlit as st


from exam_camera import cleanup_removed_feeds, read_feed_frame, release_all_captures, scan_cameras
from exam_config import ensure_dirs
from exam_detection import close_detectors, detect_on_frame, ensure_detectors, init_feed_state
from exam_reporting import generate_report, record_incident, save_snapshot
from exam_state import add_event, get_candidate_meta, init_state, parse_candidate_map




def close_resources() -> None:
    release_all_captures()
    close_detectors()




def connected_cameras(feed_sources: list[str]) -> int:
    return sum(1 for src in feed_sources if st.session_state.feed_status.get(src) == "Connected")




def status_issues(sig: dict) -> list[str]:
    issues = []
    if "Possible" in sig.get("mobile", ""):
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
    if severity == "ALERT":
        if "Possible" in sig.get("mobile", ""):
            return "POSSIBLE MOBILE USAGE", "#f5a623", "#ff7f11"
        return "SUSPECTED BEHAVIOR", "#ff4d4d", "#e53935"
    if severity == "OFFLINE":
        return "OFFLINE", "#b5bcc7", "#7f8ea3"
    return "NORMAL", "#53d769", "#2e7d32"




def frame_to_data_uri(frame_bgr) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"




def render_style() -> None:
    st.markdown(
        """
<style>
:root {
  --bg-main: #0d1b34;
  --bg-card: #13294d;
  --bg-panel: #102341;
  --line: #223f6b;
  --text: #e9f0ff;
  --muted: #b5c3de;
  --green: #53d769;
  --red: #ff4d4d;
  --amber: #f5a623;
}
.stApp {
  background: radial-gradient(1200px 700px at 10% -10%, #1e3761 0%, #0d1b34 50%, #0a1730 100%);
}
[data-testid="stSidebar"] { display: none; }
.main .block-container {
  max-width: 1500px;
  padding-top: 1.0rem;
}
.top-title {
  background: linear-gradient(180deg, #18345f, #132a4c);
  border: 1px solid var(--line);
  border-radius: 14px;
  text-align: center;
  padding: 12px;
  color: var(--text);
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.2px;
}
.top-strip {
  margin-top: 10px;
  background: linear-gradient(180deg, #142b50, #112544);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 18px;
  color: var(--text);
  display: flex;
  justify-content: space-between;
  font-size: 1.45rem;
  font-weight: 700;
}
.main-panel {
  margin-top: 12px;
  background: linear-gradient(180deg, #10213f, #0f203c);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px;
}
.status-board {
  background: linear-gradient(180deg, #112544, #0d1d37);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px;
  color: var(--text);
  min-height: 740px;
}
.status-title {
  font-size: 1.95rem;
  font-weight: 800;
  margin-bottom: 10px;
}
.cam-entry {
  border-top: 1px solid #274a78;
  padding: 10px 0;
}
.cam-name {
  font-size: 1.25rem;
  font-weight: 700;
}
.cam-state {
  font-size: 1.25rem;
  font-weight: 800;
  margin: 3px 0 6px 0;
}
.issue-line {
  color: var(--muted);
  font-size: 1.08rem;
  margin: 2px 0;
}
.feed-card {
  position: relative;
  background: #0f2342;
  border: 5px solid #2e7d32;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 8px 20px rgba(0,0,0,0.35);
  margin-bottom: 12px;
}
.feed-card.alert { border-color: #e53935; }
.feed-card.mobile { border-color: #ff9800; }
.feed-img {
  width: 100%;
  height: 340px;
  object-fit: cover;
  display: block;
}
.overlay-note {
  position: absolute;
  left: 16px;
  top: 14px;
  background: rgba(195, 23, 23, 0.82);
  color: #fff;
  font-weight: 800;
  font-size: 1.45rem;
  border-radius: 7px;
  padding: 6px 12px;
}
.feed-footer {
  background: #0d1f3a;
  color: #e8f1ff;
  font-size: 1.55rem;
  font-weight: 700;
  text-align: center;
  padding: 7px 10px;
}
.action-row {
  margin-top: 12px;
  background: linear-gradient(180deg, #10213f, #0f203c);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
}
.compact-note {
  color: #d9e6ff;
  font-size: 1rem;
}
</style>
""",
        unsafe_allow_html=True,
    )




def render_controls() -> None:
    with st.expander("Configuration", expanded=False):
        a, b = st.columns(2)
        with a:
            new_raw = st.text_area("Feed sources (one per line, index or rtsp/http)", value=st.session_state.feeds_raw, height=120)
            st.session_state.feeds_raw = new_raw
            if st.button("Apply Feed List", use_container_width=True):
                parsed = [x.strip() for x in st.session_state.feeds_raw.splitlines() if x.strip()]
                st.session_state.feed_list = parsed or ["0"]
                cleanup_removed_feeds()
                add_event(f"Feed list updated: {st.session_state.feed_list}")
            if st.button("Scan Local Cameras", use_container_width=True):
                found = scan_cameras()
                st.session_state.feed_list = found or ["0"]
                st.session_state.feeds_raw = "\n".join(st.session_state.feed_list)
                cleanup_removed_feeds()
                add_event(f"Local camera scan: {st.session_state.feed_list}")
        with b:
            meta_raw = st.text_area(
                "Camera metadata (camera,candidate,resume_link)",
                value=st.session_state.candidate_meta_raw,
                height=120,
                help="One line per camera. Example: 0,John Doe,https://resume.link/john",
            )
            st.session_state.candidate_meta_raw = meta_raw
            st.session_state.candidate_map = parse_candidate_map(meta_raw)
            st.session_state.grid_cols = st.selectbox("Grid Columns", [2, 3, 4], index=[2, 3, 4].index(st.session_state.grid_cols))
            st.session_state.analysis_batch_size = st.selectbox(
                "Feeds analyzed per cycle", [1, 2, 3, 4], index=[1, 2, 3, 4].index(st.session_state.analysis_batch_size)
            )
            st.session_state.live_preview = st.toggle("Live preview when stopped", value=st.session_state.live_preview)




def render_top(feed_sources: list[str]) -> None:
    connected = connected_cameras(feed_sources)
    monitor_state = "ACTIVE" if st.session_state.running else "STOPPED"
    monitor_color = "#58e676" if st.session_state.running else "#ffbe55"
    st.markdown("<div class='top-title'>AI Exam Monitoring Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='top-strip'>"
            f"<div>Connected Cameras: <span style='color:#58e676'>{connected}</span></div>"
            f"<div>Monitoring Status: <span style='color:{monitor_color}'>{monitor_state}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )




def render_status_board(feed_sources: list[str]) -> None:
    blocks = ["<div class='status-board'><div class='status-title'>Camera Status</div>"]
    for i, src in enumerate(feed_sources):
        sig = st.session_state.feed_signals.get(
            src,
            {"severity": "NORMAL", "mobile": "No mobile signal", "talking": "No talking signal", "paper": "No paper signal", "head_turn": "No head-turn signal"},
        )
        meta = get_candidate_meta(src)
        status_text, status_color, _ = status_theme(sig)
        issues = status_issues(sig)
        safe_name = html.escape(meta.get("candidate", f"Student_{i+1}"))
        blocks.append("<div class='cam-entry'>")
        blocks.append(f"<div class='cam-name'>Camera {i + 1} - {safe_name}</div>")
        blocks.append(f"<div class='cam-state' style='color:{status_color}'>STATUS: {html.escape(status_text)}</div>")
        for item in issues[:3]:
            blocks.append(f"<div class='issue-line'>- {html.escape(item)}</div>")
        blocks.append("</div>")
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)




def render_feed_cards(feed_sources: list[str]) -> None:
    cols = st.columns(2)
    for idx, src in enumerate(feed_sources):
        frame = st.session_state.feed_frames.get(src)
        if frame is None:
            continue
        sig = st.session_state.feed_signals.get(src, {})
        status_text, _status_color, border_color = status_theme(sig)
        issues = [x for x in status_issues(sig) if x != "No issues detected"]
        overlay = html.escape(" | ".join(issues[:2])) if issues else ""
        css_class = "feed-card"
        if "MOBILE" in status_text:
            css_class += " mobile"
        elif status_text == "SUSPECTED BEHAVIOR":
            css_class += " alert"


        uri = frame_to_data_uri(frame)
        footer_status = "Normal" if status_text == "NORMAL" else status_text.title()
        card_html = [
            f"<div class='{css_class}' style='border-color:{border_color}'>",
            f"<img class='feed-img' src='{uri}'/>",
        ]
        if overlay:
            card_html.append(f"<div class='overlay-note'>- {overlay}</div>")
        card_html.append(f"<div class='feed-footer'>Cam {idx + 1} | Status: {html.escape(footer_status)}</div>")
        card_html.append("</div>")
        with cols[idx % 2]:
            st.markdown("".join(card_html), unsafe_allow_html=True)




def process_detection(feed_sources: list[str]) -> None:
    if not st.session_state.running:
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
        frame = st.session_state.feed_frames[src]
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
        if signal_text["severity"] != "ALERT":
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




def update_frames(feed_sources: list[str]) -> None:
    for src in feed_sources:
        init_feed_state(src)
        st.session_state.feed_frames[src] = read_feed_frame(src)




def render_actions(feed_sources: list[str]) -> None:
    st.markdown("<div class='action-row'>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns([1.2, 1, 1, 1])
    with a1:
        if st.session_state.running:
            if st.button("Stop Monitoring", use_container_width=True):
                st.session_state.running = False
                close_resources()
                generate_report()
                add_event("Monitoring stopped")
                add_event("Per-camera reports generated")
        else:
            if st.button("Start Monitoring", use_container_width=True):
                st.session_state.running = True
                add_event("Monitoring started")
    with a2:
        if st.button("Generate Report", use_container_width=True):
            generate_report()
            add_event("Final report generated")
    with a3:
        if st.button("Reset Risk", use_container_width=True):
            st.session_state.risk_score = 0.0
            st.session_state.incidents = []
            st.session_state.feed_risk_scores = {}
            add_event("Risk and incidents reset")
    with a4:
        st.markdown(f"<div class='compact-note'><b>Risk:</b> {int(st.session_state.risk_score)}<br><b>Incidents:</b> {len(st.session_state.incidents)}</div>", unsafe_allow_html=True)


    if st.session_state.report_ready:
        dcols = st.columns(max(1, min(len(feed_sources), 4)))
        for i, source in enumerate(feed_sources[:4]):
            report = st.session_state.per_camera_reports.get(source)
            if report is None:
                continue
            with dcols[i % len(dcols)]:
                st.download_button(
                    f"Download Report CAM {source}",
                    report["txt"],
                    report["txt_name"],
                    "text/plain",
                    use_container_width=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)




def render_logs() -> None:
    if st.session_state.events:
        with st.expander("Recent Events", expanded=False):
            for event in st.session_state.events[:12]:
                st.write(f"- {event}")




def main() -> None:
    st.set_page_config(page_title="AI Exam Proctor - Multi Feed", layout="wide")
    init_state()
    ensure_dirs()
    render_style()
    render_controls()


    feed_sources = st.session_state.feed_list
    update_frames(feed_sources)
    process_detection(feed_sources)


    render_top(feed_sources)
    st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
    left, right = st.columns([1, 3])
    with left:
        render_status_board(feed_sources)
    with right:
        render_feed_cards(feed_sources)
    st.markdown("</div>", unsafe_allow_html=True)


    render_actions(feed_sources)
    render_logs()


    if st.session_state.running or st.session_state.live_preview:
        time.sleep(0.09 if st.session_state.running else 0.18)
        st.rerun()




if __name__ == "__main__":
    main()



