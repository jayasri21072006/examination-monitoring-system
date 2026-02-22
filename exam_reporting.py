import csv
import io
import os
from datetime import datetime

import cv2
import streamlit as st

from exam_config import REPORT_DIR, SNAPSHOT_DIR, ensure_dirs
from exam_state import get_candidate_meta


def save_snapshot(frame_bgr, source: str) -> str:
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"feed_{source.replace(':', '_')}_{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path


def record_incident(source: str, signal_text: dict, snapshot: str) -> None:
    meta = get_candidate_meta(source)
    st.session_state.incidents.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feed": source,
            "candidate": meta["candidate"],
            "resume": meta["resume"],
            "severity": signal_text["severity"],
            "mobile": signal_text["mobile"],
            "talking": signal_text["talking"],
            "paper": signal_text["paper"],
            "head_turn": signal_text["head_turn"],
            "risk_score": int(st.session_state.risk_score),
            "snapshot": snapshot,
        }
    )
    st.session_state.incidents = st.session_state.incidents[-1000:]


def save_face_profile(source: str):
    frame = st.session_state.feed_frames.get(source)
    if frame is None:
        return ""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = source.replace(":", "_").replace("/", "_").replace("\\", "_")
    path = os.path.join(SNAPSHOT_DIR, f"camera_{safe_source}_face_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def _count_behaviors(rows: list[dict]) -> dict:
    return {
        "mobile": sum(1 for r in rows if r.get("mobile", "").startswith("Possible")),
        "talking": sum(1 for r in rows if r.get("talking", "").startswith("Talking")),
        "paper": sum(1 for r in rows if r.get("paper", "").startswith("Paper")),
        "head_turn": sum(1 for r in rows if r.get("head_turn", "").startswith("Repeated")),
    }


def generate_report() -> None:
    ensure_dirs()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "timestamp",
            "feed",
            "candidate",
            "resume",
            "severity",
            "mobile",
            "talking",
            "paper",
            "head_turn",
            "risk_score",
            "snapshot",
        ]
    )
    for row in st.session_state.incidents:
        writer.writerow(
            [
                row["timestamp"],
                row["feed"],
                row.get("candidate", ""),
                row.get("resume", ""),
                row["severity"],
                row["mobile"],
                row["talking"],
                row["paper"],
                row["head_turn"],
                row["risk_score"],
                row["snapshot"],
            ]
        )
    st.session_state.report_csv = buf.getvalue()

    summary = [
        "AI Exam Monitoring - Final Report",
        f"Session started: {st.session_state.session_started_at}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total feeds: {len(st.session_state.feed_list)}",
        f"Total incidents: {len(st.session_state.incidents)}",
        f"Final risk score: {int(st.session_state.risk_score)}",
        "Final decision should be made by the invigilator.",
    ]
    st.session_state.report_txt = "\n".join(summary)
    with open(os.path.join(REPORT_DIR, "exam_incidents.csv"), "w", newline="", encoding="utf-8") as f:
        f.write(st.session_state.report_csv)
    with open(os.path.join(REPORT_DIR, "exam_final_report.txt"), "w", encoding="utf-8") as f:
        f.write(st.session_state.report_txt)

    by_feed = {}
    for row in st.session_state.incidents:
        by_feed.setdefault(row["feed"], []).append(row)

    per_camera_reports = {}
    for source in st.session_state.feed_list:
        rows = by_feed.get(source, [])
        meta = get_candidate_meta(source)
        behavior_counts = _count_behaviors(rows)
        profile_image = save_face_profile(source)
        feed_buf = io.StringIO()
        feed_writer = csv.writer(feed_buf)
        feed_writer.writerow(
            [
                "timestamp",
                "feed",
                "candidate",
                "resume",
                "severity",
                "mobile",
                "talking",
                "paper",
                "head_turn",
                "risk_score",
                "snapshot",
                "profile_face",
            ]
        )
        for row in rows:
            feed_writer.writerow(
                [
                    row["timestamp"],
                    row["feed"],
                    row.get("candidate", ""),
                    row.get("resume", ""),
                    row["severity"],
                    row["mobile"],
                    row["talking"],
                    row["paper"],
                    row["head_turn"],
                    row["risk_score"],
                    row["snapshot"],
                    profile_image,
                ]
            )

        feed_summary = [
            "AI Exam Monitoring - Camera Report",
            f"Session started: {st.session_state.session_started_at}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Camera source: {source}",
            f"Candidate: {meta['candidate']}",
            f"Resume: {meta['resume'] or 'N/A'}",
            f"Captured face image: {profile_image or 'Not available'}",
            f"Incidents on this camera: {len(rows)}",
            f"Mobile usage alerts: {behavior_counts['mobile']}",
            f"Talking alerts: {behavior_counts['talking']}",
            f"Paper alerts: {behavior_counts['paper']}",
            f"Head-turn alerts: {behavior_counts['head_turn']}",
            "Final decision should be made by the invigilator.",
        ]
        safe_source = source.replace(":", "_").replace("/", "_").replace("\\", "_")
        csv_name = f"camera_{safe_source}_incidents.csv"
        txt_name = f"camera_{safe_source}_report.txt"
        csv_content = feed_buf.getvalue()
        txt_content = "\n".join(feed_summary)
        with open(os.path.join(REPORT_DIR, csv_name), "w", newline="", encoding="utf-8") as f:
            f.write(csv_content)
        with open(os.path.join(REPORT_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write(txt_content)
        per_camera_reports[source] = {
            "candidate": meta["candidate"],
            "csv_name": csv_name,
            "txt_name": txt_name,
            "csv": csv_content,
            "txt": txt_content,
            "face_image": profile_image,
            "behavior_counts": behavior_counts,
        }

    st.session_state.per_camera_reports = per_camera_reports
    st.session_state.report_ready = True
