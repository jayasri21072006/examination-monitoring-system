import time
from datetime import datetime

import streamlit as st


def init_state() -> None:
    defaults = {
        "running": False,
        "session_started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feeds_raw": "0",
        "feed_list": ["0"],
        "candidate_meta_raw": "",
        "candidate_map": {},
        "captures": {},
        "cam_workers": {},
        "feed_frames": {},
        "feed_status": {},
        "feed_signals": {},
        "feed_risk_scores": {},
        "feed_counters": {},
        "cam_retry_after": {},
        "cam_last_ok": {},
        "events": [],
        "incidents": [],
        "report_csv": "",
        "report_txt": "",
        "per_camera_reports": {},
        "report_ready": False,
        "risk_score": 0.0,
        "tick": 0,
        "rr_index": 0,
        "analysis_batch_size": 2,
        "face_mesh": None,
        "hands": None,
        "last_snapshot_ts": {},
        "last_incident_ts": {},
        "last_detect_ts": 0.0,
        "grid_cols": 3,
        "live_preview": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_event(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state.events.insert(0, f"[{ts}] {message}")
    st.session_state.events = st.session_state.events[:30]


def parse_candidate_map(raw_text: str) -> dict:
    mapping = {}
    for line in raw_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        source = parts[0]
        candidate = parts[1] if len(parts) > 1 and parts[1] else f"Candidate {source}"
        resume = parts[2] if len(parts) > 2 else ""
        mapping[source] = {"candidate": candidate, "resume": resume}
    return mapping


def get_candidate_meta(source: str) -> dict:
    return st.session_state.candidate_map.get(source, {"candidate": f"Candidate {source}", "resume": ""})
