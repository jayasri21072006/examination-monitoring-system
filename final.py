import html
import io
import os
import datetime
import threading
import time
from urllib.parse import quote

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

import exam_camera
import exam_detection
import exam_reporting
import exam_state
from exam_camera import cleanup_removed_feeds, read_feed_frame, release_all_captures, scan_cameras
from exam_config import ensure_dirs
from exam_detection import close_detectors, detect_on_frame, ensure_detectors, init_feed_state
from exam_reporting import record_incident, save_snapshot
from exam_state import add_event, get_candidate_meta, init_state, parse_candidate_map


# ── shim so sub-modules keep working ────────────────────────────────────────
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
exam_camera.st   = st
exam_detection.st = st
exam_reporting.st = st
exam_state.st    = st

app        = Flask(__name__)
state_lock = threading.RLock()
bg_stop    = threading.Event()
bg_thread  = None


# ─────────────────────────────────────────────────────────────────────────────
#  PDF REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_exam_pdf(source, candidate, incidents, snapshot_paths, events, risk_score):
    """
    Generate a PDF report for one camera / candidate.
    Returns raw PDF bytes.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=f"Exam Report – {candidate}",
    )
    W, _ = A4
    styles = getSampleStyleSheet()
    now_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")

    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=22,
                              textColor=colors.HexColor("#0b1e35"),
                              spaceAfter=6, alignment=TA_CENTER)
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=11,
                              textColor=colors.HexColor("#2d5f9a"),
                              spaceAfter=4, alignment=TA_CENTER)
    sec_s   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
                              textColor=colors.HexColor("#0b1e35"),
                              spaceBefore=14, spaceAfter=4)
    small   = ParagraphStyle("Sm", parent=styles["Normal"], fontSize=9,
                              textColor=colors.HexColor("#444"))
    foot_s  = ParagraphStyle("F", parent=small, alignment=TA_CENTER,
                              textColor=colors.HexColor("#888"))
    normal  = styles["Normal"]

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("AI Exam Monitoring System", title_s))
    story.append(Paragraph("Proctoring Incident Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#2d5f9a")))
    story.append(Spacer(1, 0.3*cm))

    info_data = [
        ["Candidate",        candidate or "—"],
        ["Camera / Source",  str(source)],
        ["Report Generated", now_str],
        ["Total Incidents",  str(len(incidents))],
        ["Final Risk Score", f"{int(risk_score)} / 100"],
    ]
    it = Table(info_data, colWidths=[5*cm, 12*cm])
    it.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), colors.HexColor("#e8f0fa")),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#adc1df")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.white, colors.HexColor("#f4f8ff")]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    story.append(it)
    story.append(Spacer(1, 0.4*cm))

    # ── Risk bar ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Score", sec_s))
    bar_pct  = min(100, max(0, int(risk_score)))
    bar_col  = (colors.HexColor("#e53935") if bar_pct >= 60 else
                colors.HexColor("#f5a623") if bar_pct >= 30 else
                colors.HexColor("#2e7d32"))
    usable_w = W - 4*cm
    fill_w   = max(0.01, usable_w * bar_pct / 100)
    empty_w  = max(0.01, usable_w - fill_w)
    bar = Table([[""]], colWidths=[fill_w])
    bar.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bar_col),
        ("ROWHEIGHT",     (0,0),(-1,-1), 18),
        ("GRID",          (0,0),(-1,-1), 0, colors.white),
    ]))
    outer = Table([[bar, ""]], colWidths=[fill_w, empty_w])
    outer.setStyle(TableStyle([
        ("BACKGROUND",    (1,0),(1,0), colors.HexColor("#dde4ee")),
        ("ROWHEIGHT",     (0,0),(-1,-1), 18),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#adc1df")),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(outer)
    story.append(Paragraph(f"Risk level: {bar_pct}/100", small))
    story.append(Spacer(1, 0.3*cm))

    # ── Snapshots ─────────────────────────────────────────────────────────────
    valid_imgs = []
    for sp in (snapshot_paths or [])[:8]:
        try:
            if isinstance(sp, str) and sp and os.path.isfile(sp):
                valid_imgs.append(RLImage(sp, width=8*cm, height=6*cm, kind="proportional"))
            elif sp is not None and not isinstance(sp, str):
                # numpy frame
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, sp)
                tmp.close()
                valid_imgs.append(RLImage(tmp.name, width=8*cm, height=6*cm, kind="proportional"))
        except Exception:
            pass

    if valid_imgs:
        story.append(Paragraph("Captured Snapshots", sec_s))
        for i in range(0, len(valid_imgs), 2):
            pair = valid_imgs[i:i+2]
            while len(pair) < 2:
                pair.append("")
            pt = Table([pair], colWidths=[9*cm, 9*cm])
            pt.setStyle(TableStyle([
                ("ALIGN",         (0,0),(-1,-1), "CENTER"),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
                ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#adc1df")),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ]))
            story.append(pt)
            story.append(Spacer(1, 0.2*cm))

    # ── Incident table ────────────────────────────────────────────────────────
    story.append(Paragraph("Incident Log", sec_s))
    if incidents:
        header = ["#", "Time", "Severity", "Detections"]
        rows   = [header]
        sev_hi = []
        for idx, inc in enumerate(incidents, 1):
            ts = inc.get("ts", inc.get("timestamp", ""))
            if isinstance(ts, (int, float)):
                ts = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            sigs     = inc.get("signals", inc.get("signal_text", {}))
            severity = sigs.get("severity", "—") if isinstance(sigs, dict) else "—"
            dets = []
            if isinstance(sigs, dict):
                for k in ("mobile", "talking", "paper", "head_turn"):
                    v = sigs.get(k, "")
                    if v and "No " not in v and "NORMAL" not in v.upper():
                        dets.append(v)
            sev_hi.append(severity)
            rows.append([str(idx), str(ts), severity,
                         Paragraph("; ".join(dets) if dets else "—", small)])

        col_w = [1*cm, 3*cm, 3.5*cm, usable_w - 7.5*cm]
        inc_t = Table(rows, colWidths=col_w, repeatRows=1)
        sev_colors = {
            "HIGH ALERT": colors.HexColor("#ffd6d6"),
            "ALERT":      colors.HexColor("#ffe8cc"),
            "WARNING":    colors.HexColor("#fffbe0"),
        }
        cmds = [
            ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#2d5f9a")),
            ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#adc1df")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f4f8ff")]),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ]
        for i, sev in enumerate(sev_hi, 1):
            if sev in sev_colors:
                cmds.append(("BACKGROUND", (0,i), (-1,i), sev_colors[sev]))
        inc_t.setStyle(TableStyle(cmds))
        story.append(inc_t)
    else:
        story.append(Paragraph("No incidents recorded for this camera.", normal))

    story.append(Spacer(1, 0.4*cm))

    # ── Event log ─────────────────────────────────────────────────────────────
    story.append(Paragraph("System Event Log", sec_s))
    for ev in (events or [])[:40]:
        story.append(Paragraph(f"• {ev}", small))
    if not events:
        story.append(Paragraph("No events recorded.", normal))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#adc1df")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by AI Exam Monitoring System  •  {now_str}", foot_s))

    doc.build(story)
    return buf.getvalue()


def generate_all_pdf_reports():
    """Build one PDF per camera; store bytes in session_state.report_files."""
    feeds         = list(st.session_state.feed_list)
    all_incidents = st.session_state.incidents
    all_events    = list(st.session_state.events)
    risk_score    = st.session_state.risk_score

    st.session_state.per_camera_reports = {}
    st.session_state.report_files       = {}

    for i, src in enumerate(feeds):
        meta      = get_candidate_meta(src)
        candidate = meta.get("candidate", f"Student_{i+1}")

        cam_incidents = [
            inc for inc in all_incidents
            if str(inc.get("source", "")) == str(src)
        ]
        snap_paths = [
            inc.get("snapshot") or inc.get("snapshot_path", "")
            for inc in cam_incidents
            if inc.get("snapshot") or inc.get("snapshot_path", "")
        ]
        # fallback: latest live frame
        if not snap_paths:
            lf = st.session_state.feed_frames.get(src)
            if lf is not None:
                snap_paths = [lf]

        try:
            pdf_bytes = build_exam_pdf(
                source         = src,
                candidate      = candidate,
                incidents      = cam_incidents,
                snapshot_paths = snap_paths,
                events         = all_events,
                risk_score     = risk_score,
            )
        except Exception as exc:
            add_event(f"PDF build failed for {src}: {exc}")
            pdf_bytes = b""

        pdf_name = f"report_cam{i+1}_{candidate.replace(' ','_')}.pdf"
        st.session_state.report_files[pdf_name] = {
            "bytes":    pdf_bytes,
            "mimetype": "application/pdf",
        }
        st.session_state.per_camera_reports[src] = {
            "pdf_name":  pdf_name,
            "candidate": candidate,
        }
        add_event(f"Report ready: {pdf_name}")

    st.session_state.report_ready = True


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def close_resources():
    release_all_captures()
    close_detectors()


def status_issues(sig):
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


def status_theme(sig):
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


def connected_cameras(feed_sources):
    return sum(1 for src in feed_sources
               if st.session_state.feed_status.get(src) == "Connected")


def update_frames(feed_sources):
    for src in feed_sources:
        init_feed_state(src)
        st.session_state.feed_frames[src] = read_feed_frame(src)


def process_detection(feed_sources):
    if not (st.session_state.running or st.session_state.live_preview):
        return
    now = time.time()
    if now - st.session_state.last_detect_ts < 0.22:
        return
    st.session_state.last_detect_ts = now

    ensure_detectors()
    st.session_state.tick += 1
    n = len(feed_sources)
    if n == 0:
        return

    start  = st.session_state.rr_index % n
    batch  = min(st.session_state.analysis_batch_size, n)
    indexes = [(start + i) % n for i in range(batch)]
    st.session_state.rr_index = (start + batch) % n
    active_alerts = 0

    for idx in indexes:
        src   = feed_sources[idx]
        frame = st.session_state.feed_frames.get(src)
        if frame is None:
            continue
        if st.session_state.feed_status.get(src) != "Connected":
            st.session_state.feed_signals[src] = {
                "mobile": "No mobile signal", "talking": "No talking signal",
                "paper":  "No paper signal",  "head_turn": "No head-turn signal",
                "severity": "OFFLINE",
            }
            continue

        signal_text = detect_on_frame(src, frame)
        if signal_text.get("severity") not in ("ALERT", "HIGH ALERT"):
            st.session_state.feed_risk_scores[src] = max(
                0.0, st.session_state.feed_risk_scores.get(src, 0.0) - 0.35)
            continue

        active_alerts += 1
        st.session_state.feed_risk_scores[src] = min(
            100.0, st.session_state.feed_risk_scores.get(src, 0.0) + 1.8)
        last_incident = st.session_state.last_incident_ts.get(src, 0.0)
        if time.time() - last_incident > 3:
            snap = save_snapshot(frame, src)
            st.session_state.last_snapshot_ts[src]  = time.time()
            record_incident(src, signal_text, snap)
            st.session_state.last_incident_ts[src]  = time.time()
            add_event(f"Snapshot captured for feed {src}")
            add_event(f"Incident logged on feed {src}")

    if active_alerts > 0:
        st.session_state.risk_score = min(100.0, st.session_state.risk_score + 1.6 * active_alerts)
    else:
        st.session_state.risk_score = max(0.0, st.session_state.risk_score - 0.6)


def background_loop():
    while not bg_stop.is_set():
        with state_lock:
            feed_sources = list(st.session_state.feed_list)
            active = st.session_state.running or st.session_state.live_preview
            if active:
                update_frames(feed_sources)
            process_detection(feed_sources)
        time.sleep(0.06 if st.session_state.running else 0.12)


def ensure_started():
    global bg_thread
    with state_lock:
        if "running" not in st.session_state:
            init_state()
            ensure_dirs()
    if bg_thread is None or not bg_thread.is_alive():
        bg_stop.clear()
        bg_thread = threading.Thread(target=background_loop, daemon=True)
        bg_thread.start()


def snapshot_state():
    with state_lock:
        feeds    = list(st.session_state.feed_list)
        cam_rows = []
        for i, src in enumerate(feeds):
            sig = st.session_state.feed_signals.get(src, {
                "severity": "NORMAL",
                "mobile": "No mobile signal", "talking": "No talking signal",
                "paper":  "No paper signal",  "head_turn": "No head-turn signal",
            })
            status_text, status_color, border_color = status_theme(sig)
            meta = get_candidate_meta(src)
            cam_rows.append({
                "index":        i + 1,
                "source":       src,
                "candidate":    meta.get("candidate", f"Student_{i+1}"),
                "status_text":  status_text,
                "status_color": status_color,
                "border_color": border_color,
                "issues":       status_issues(sig)[:3],
                "overlay":      " | ".join(
                    [x for x in status_issues(sig) if x != "No issues detected"][:2]
                ),
                "frame_url":    "/frame?source=" + quote(src, safe=""),
            })

        # ── FIXED: safe .get() with fallback ──────────────────────────────
        reports = []
        if st.session_state.report_ready:
            for source in feeds:
                rep      = st.session_state.per_camera_reports.get(source)
                pdf_name = (rep or {}).get("pdf_name") if rep else None
                if pdf_name:
                    i_label  = feeds.index(source) + 1
                    candidate = (rep or {}).get("candidate", f"CAM {i_label}")
                    reports.append({
                        "label": f"Download Report – {candidate} (CAM {i_label})",
                        "file":  pdf_name,
                    })

        return {
            "running":             st.session_state.running,
            "connected":           connected_cameras(feeds),
            "feed_count":          len(feeds),
            "risk_score":          int(st.session_state.risk_score),
            "incidents":           len(st.session_state.incidents),
            "feeds_raw":           st.session_state.feeds_raw,
            "candidate_meta_raw":  st.session_state.candidate_meta_raw,
            "analysis_batch_size": st.session_state.analysis_batch_size,
            "grid_cols":           st.session_state.grid_cols,
            "live_preview":        st.session_state.live_preview,
            "rows":                cam_rows,
            "events":              st.session_state.events[:12],
            "reports":             reports,
            "report_ready":        st.session_state.report_ready,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  HTML  (adds a "Report Ready" banner + auto-shows download buttons)
# ─────────────────────────────────────────────────────────────────────────────

PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Exam Proctor - Multi Feed</title>
  <style>
    :root {
      --bg-main:#07121f;--bg-deep:#0b1e35;--panel:rgba(14,34,58,0.82);
      --panel-2:rgba(19,45,73,0.9);--line:#2f4f78;--text:#edf4ff;
      --muted:#adc1df;--gold:#f8c85a;--cyan:#4de2ff;--green:#51f09d;
      --danger:#ff6a6a;
    }
    body{margin:0;font-family:"Trebuchet MS","Segoe UI",Tahoma,sans-serif;
      color:var(--text);
      background:radial-gradient(1200px 600px at 0% -20%,rgba(77,226,255,.16),rgba(7,18,31,0)),
        radial-gradient(900px 500px at 95% 0%,rgba(248,200,90,.13),rgba(7,18,31,0)),
        linear-gradient(160deg,#07121f 0%,#0b1e35 45%,#091a2d 100%);
      min-height:100vh;}
    .wrap{max-width:1500px;margin:0 auto;padding:16px;}
    .top-title{background:linear-gradient(110deg,rgba(248,200,90,.2),rgba(77,226,255,.15));
      border:1px solid #44628c;border-radius:18px;text-align:center;padding:16px;
      font-size:2.2rem;letter-spacing:.5px;font-weight:900;
      box-shadow:0 12px 26px rgba(0,0,0,.35);backdrop-filter:blur(4px);}
    .top-strip{margin-top:12px;background:var(--panel);border:1px solid var(--line);
      border-radius:14px;padding:14px 18px;display:flex;justify-content:space-between;
      font-size:1.15rem;font-weight:800;}
    .panel{margin-top:14px;background:var(--panel);border:1px solid var(--line);
      border-radius:14px;padding:14px;box-shadow:0 10px 24px rgba(0,0,0,.32);
      backdrop-filter:blur(4px);}
    button{border:0;border-radius:11px;
      background:linear-gradient(90deg,#214979,#2d5f9a);
      color:#fff;padding:10px 14px;font-weight:700;cursor:pointer;
      transition:transform .18s,box-shadow .18s,filter .18s;}
    button:hover{transform:translateY(-1px);box-shadow:0 10px 20px rgba(0,0,0,.24);filter:brightness(1.08);}
    .btn-stop{background:linear-gradient(90deg,#7a2121,#c0392b);}
    .btn-stop:hover{filter:brightness(1.1);}
    .main{margin-top:14px;display:grid;grid-template-columns:1fr 3fr;gap:14px;}
    .status-board{background:var(--panel-2);border:1px solid var(--line);
      border-radius:12px;padding:14px;min-height:740px;}
    .status-title{font-size:1.35rem;font-weight:900;margin-bottom:10px;color:var(--gold);}
    .cam-entry{border-top:1px solid #2f507a;padding:10px 0;animation:rise .35s ease;}
    .cam-name{font-size:1rem;font-weight:700;}
    .cam-state{font-size:1rem;font-weight:900;margin:3px 0 6px;}
    .issue-line{color:var(--muted);font-size:.92rem;margin:2px 0;}
    .feeds{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
    .feed-card{position:relative;background:#0f2342;border:4px solid #2e7d32;
      border-radius:12px;overflow:hidden;box-shadow:0 12px 24px rgba(0,0,0,.34);
      animation:rise .35s ease;transition:transform .2s,box-shadow .2s;}
    .feed-card:hover{transform:translateY(-2px);box-shadow:0 14px 30px rgba(0,0,0,.42);}
    .feed-img{width:100%;height:340px;object-fit:cover;display:block;background:#091628;}
    .overlay-note{position:absolute;left:14px;top:12px;
      background:linear-gradient(90deg,rgba(195,23,23,.9),rgba(255,106,106,.85));
      color:#fff;font-weight:800;font-size:.95rem;border-radius:8px;padding:6px 11px;}
    .feed-footer{background:#0c1e37;font-size:1rem;font-weight:800;text-align:center;
      padding:8px 10px;border-top:1px solid #244367;}
    .action-row{margin-top:2px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
    .compact-note{margin-left:auto;font-size:1rem;padding:8px 12px;border-radius:10px;
      border:1px solid #3b5e89;background:rgba(8,24,44,.7);}
    .logs{margin-top:14px;background:var(--panel);border:1px solid var(--line);
      border-radius:14px;padding:12px;}

    /* ── Report ready banner ───────────────────────────────────────────── */
    #reportBanner{display:none;margin-top:14px;
      background:linear-gradient(110deg,rgba(81,240,157,.18),rgba(77,226,255,.12));
      border:2px solid #51f09d;border-radius:16px;padding:18px 22px;}
    #reportBanner h2{margin:0 0 10px;color:#51f09d;font-size:1.5rem;}
    #reportBanner p{margin:0 0 14px;color:var(--muted);}
    .dl-btn{display:inline-block;margin:4px 6px 4px 0;
      background:linear-gradient(90deg,#1a6b3a,#27ae60);
      color:#fff;padding:11px 18px;border-radius:11px;font-weight:800;
      font-size:1rem;text-decoration:none;
      transition:transform .18s,box-shadow .18s,filter .18s;}
    .dl-btn:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(0,0,0,.3);filter:brightness(1.1);}

    #monitorState.active{color:var(--green)!important;text-shadow:0 0 10px rgba(81,240,157,.45);}
    @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
    @media(max-width:980px){.main{grid-template-columns:1fr}.feeds{grid-template-columns:1fr}.compact-note{margin-left:0}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top-title">AI Exam Monitoring Dashboard</div>
  <div class="top-strip">
    <div>Connected Cameras: <span id="connected" style="color:#58e676">0</span></div>
    <div>Monitoring Status: <span id="monitorState" style="color:#ffbe55">STOPPED</span></div>
  </div>

  <!-- Report Ready Banner (shown after Stop) -->
  <div id="reportBanner">
    <h2>&#10003; Session Complete — Reports Ready</h2>
    <p>Monitoring has stopped. Click a button below to download the PDF report for each student.</p>
    <div id="reportLinks"></div>
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
      <div class="compact-note">
        <b>Risk:</b> <span id="risk">0</span> |
        <b>Incidents:</b> <span id="incidents">0</span>
      </div>
    </div>
  </div>

  <div class="logs">
    <h3 style="margin-top:0">Recent Events</h3>
    <div id="events"></div>
  </div>
</div>

<script>
async function post(url,body={}){
  await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
}

async function toggleRun(){
  const btn=document.getElementById("startStopBtn");
  await post(btn.dataset.running==="1"?"/api/stop":"/api/start");
  await refresh();
}
async function generateReport(){await post("/api/generate_report");await refresh();}
async function resetRisk(){await post("/api/reset_risk");await refresh();}

function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");}

function renderReportBanner(data){
  const banner=document.getElementById("reportBanner");
  const links=document.getElementById("reportLinks");
  if(data.report_ready && data.reports && data.reports.length>0){
    links.innerHTML=data.reports.map(r=>
      `<a class="dl-btn" href="/download/${encodeURIComponent(r.file)}" download="${esc(r.file)}">
         &#8659; ${esc(r.label)}
       </a>`
    ).join("");
    banner.style.display="block";
    // scroll into view once
    if(banner.dataset.shown!=="1"){
      banner.scrollIntoView({behavior:"smooth",block:"start"});
      banner.dataset.shown="1";
    }
  } else {
    banner.style.display="none";
    banner.dataset.shown="0";
  }
}

async function refresh(){
  const res=await fetch("/api/state");
  const data=await res.json();

  document.getElementById("connected").textContent=data.connected;
  const monitor=document.getElementById("monitorState");
  monitor.textContent=data.running?"ACTIVE":"STOPPED";
  monitor.classList.toggle("active",!!data.running);
  monitor.style.color=data.running?"#58e676":"#ffbe55";

  const btn=document.getElementById("startStopBtn");
  btn.textContent=data.running?"Stop Monitoring":"Start Monitoring";
  btn.className=data.running?"btn-stop":"";
  btn.dataset.running=data.running?"1":"0";

  document.getElementById("risk").textContent=data.risk_score;
  document.getElementById("incidents").textContent=data.incidents;

  document.getElementById("statusBoard").innerHTML=data.rows.map(r=>`
    <div class="cam-entry">
      <div class="cam-name">Camera ${r.index} – ${esc(r.candidate)}</div>
      <div class="cam-state" style="color:${r.status_color}">STATUS: ${esc(r.status_text)}</div>
      ${r.issues.map(i=>`<div class="issue-line">- ${esc(i)}</div>`).join("")}
    </div>`).join("");

  renderFeeds(data.rows);
  renderReportBanner(data);

  document.getElementById("events").innerHTML=
    data.events.length?data.events.map(e=>`<div>- ${esc(e)}</div>`).join(""):"<div>No events yet</div>";
}

function feedCardId(source){return "feed_"+encodeURIComponent(source).replaceAll("%","_");}

function renderFeeds(rows){
  const grid=document.getElementById("feedsGrid");
  const wanted=new Set(rows.map(r=>r.source));
  for(const node of Array.from(grid.children)){
    if(!wanted.has(node.dataset.source))node.remove();
  }
  for(const r of rows){
    const id=feedCardId(r.source);
    let card=document.getElementById(id);
    if(!card){
      card=document.createElement("div");
      card.id=id;card.className="feed-card";card.dataset.source=r.source;
      const img=document.createElement("img");
      img.className="feed-img";
      img.src="/stream?source="+encodeURIComponent(r.source);
      card.appendChild(img);
      const overlay=document.createElement("div");
      overlay.className="overlay-note";overlay.style.display="none";
      card.appendChild(overlay);
      const footer=document.createElement("div");
      footer.className="feed-footer";
      card.appendChild(footer);
      grid.appendChild(card);
    }
    card.style.borderColor=r.border_color;
    const overlay=card.querySelector(".overlay-note");
    if(r.overlay){overlay.textContent="- "+r.overlay;overlay.style.display="block";}
    else{overlay.textContent="";overlay.style.display="none";}
    card.querySelector(".feed-footer").textContent=
      "Cam "+r.index+" | Status: "+r.status_text;
  }
}

setInterval(refresh,700);
refresh();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────────────────────────────────────

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
        st.session_state.candidate_meta_raw = str(
            payload.get("candidate_meta_raw", st.session_state.candidate_meta_raw))
        st.session_state.candidate_map = parse_candidate_map(st.session_state.candidate_meta_raw)
        st.session_state.grid_cols = int(payload.get("grid_cols", st.session_state.grid_cols))
        st.session_state.analysis_batch_size = int(
            payload.get("analysis_batch_size", st.session_state.analysis_batch_size))
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
        st.session_state.running      = True
        st.session_state.report_ready = False   # reset banner when re-starting
        add_event("Monitoring started")
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ensure_started()
    with state_lock:
        st.session_state.running = False
        close_resources()
        generate_all_pdf_reports()          # ← builds PDFs for every camera
        add_event("Monitoring stopped")
        add_event("Per-camera PDF reports generated")
    return jsonify({"ok": True})


@app.route("/api/generate_report", methods=["POST"])
def api_generate_report():
    ensure_started()
    with state_lock:
        generate_all_pdf_reports()
        add_event("Manual report generation complete")
    return jsonify({"ok": True})


@app.route("/api/reset_risk", methods=["POST"])
def api_reset_risk():
    ensure_started()
    with state_lock:
        st.session_state.risk_score        = 0.0
        st.session_state.incidents         = []
        st.session_state.feed_risk_scores  = {}
        st.session_state.report_ready      = False
        st.session_state.report_files      = {}
        st.session_state.per_camera_reports = {}
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
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + encoded.tobytes() + b"\r\n")
            time.sleep(0.08)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/download/<path:name>")
def download(name: str):
    ensure_started()
    with state_lock:
        file_obj = st.session_state.report_files.get(name)
        if not file_obj:
            return Response(status=404)
        data     = file_obj.get("bytes", b"")
        mimetype = file_obj.get("mimetype", "application/pdf")
    return Response(
        data, mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.route("/health")
def health():
    ensure_started()
    return jsonify({"ok": True})


def main():
    ensure_started()
    app.run(host="0.0.0.0", port=8502, debug=False, threaded=True)


if __name__ == "__main__":
    main()
