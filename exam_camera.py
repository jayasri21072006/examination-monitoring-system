import time
import threading

import cv2
import numpy as np
import streamlit as st

from exam_config import MAX_SCAN_INDEX


def source_to_capture_arg(source: str):
    source = source.strip()
    return int(source) if source.isdigit() else source


def is_network_source(source: str) -> bool:
    s = source.strip().lower()
    return s.startswith(("rtsp://", "http://", "https://"))


def _try_open(arg, backend: int):
    cap = cv2.VideoCapture(arg, backend)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    return cap


def _warmup_read(cap, attempts: int = 6, delay: float = 0.05) -> bool:
    # Some cameras need a few frames before read() returns valid data.
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        time.sleep(delay)
    return False


def scan_cameras() -> list[str]:
    found = []
    for idx in range(MAX_SCAN_INDEX):
        cap = _try_open(idx, cv2.CAP_DSHOW)
        if cap is None:
            cap = _try_open(idx, cv2.CAP_MSMF)
        if cap is None:
            cap = _try_open(idx, cv2.CAP_ANY)
        if cap is not None:
            if _warmup_read(cap):
                found.append(str(idx))
            cap.release()
    return found


def offline_frame(text: str) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (120, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return frame


def cleanup_removed_feeds() -> None:
    active = set(st.session_state.feed_list)
    for source in list(st.session_state.cam_workers.keys()):
        if source in active:
            continue
        _stop_worker(source)


def release_all_captures() -> None:
    for source in list(st.session_state.cam_workers.keys()):
        _stop_worker(source)
    st.session_state.cam_workers = {}
    st.session_state.captures = {}


def _open_capture(source: str):
    arg = source_to_capture_arg(source)
    if isinstance(arg, int):
        # DSHOW is often more stable with USB webcams; fall back as needed.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif is_network_source(source):
        # Network streams are more reliable through FFmpeg / generic backend.
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    cap = None
    for backend in backends:
        cap = _try_open(arg, backend)
        if cap is not None:
            break
    if cap is None:
        cap = cv2.VideoCapture(arg)
    return cap


def _worker_loop(source: str, state: dict) -> None:
    cap = None
    read_fail_streak = 0
    while not state["stop_event"].is_set():
        if cap is None:
            cap = _open_capture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 12)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                cap = None
                with state["lock"]:
                    state["status"] = "Unavailable"
                time.sleep(1.2)
                continue
            if not _warmup_read(cap):
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                with state["lock"]:
                    state["status"] = "Warming up"
                time.sleep(0.4)
                continue
            read_fail_streak = 0

        ok, frame = cap.read()
        if ok and frame is not None:
            read_fail_streak = 0
            with state["lock"]:
                state["frame"] = frame
                state["status"] = "Connected"
                state["last_ok"] = time.time()
            # Avoid CPU spin when camera delivers frames very quickly.
            time.sleep(0.01)
            continue

        read_fail_streak += 1
        with state["lock"]:
            state["status"] = "Reconnecting"
        # Tolerate short transient failures before reopening the device.
        if read_fail_streak < 3:
            time.sleep(0.03)
            continue
        try:
            cap.release()
        except Exception:
            pass
        cap = None
        read_fail_streak = 0
        time.sleep(0.2)

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _ensure_worker(source: str) -> dict:
    existing = st.session_state.cam_workers.get(source)
    if existing is not None and existing["thread"].is_alive():
        return existing

    state = {
        "frame": None,
        "status": "Reconnecting",
        "last_ok": 0.0,
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
    }
    t = threading.Thread(target=_worker_loop, args=(source, state), daemon=True)
    state["thread"] = t
    st.session_state.cam_workers[source] = state
    t.start()
    return state


def _stop_worker(source: str) -> None:
    state = st.session_state.cam_workers.pop(source, None)
    if state is None:
        return
    state["stop_event"].set()
    thread = state.get("thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout=0.4)


def read_feed_frame(source: str) -> np.ndarray:
    state = _ensure_worker(source)

    with state["lock"]:
        frame = None if state["frame"] is None else state["frame"].copy()
        status = state["status"]
        last_ok = state["last_ok"]

    st.session_state.feed_status[source] = status
    st.session_state.cam_last_ok[source] = last_ok

    # If no new frame, keep showing last good frame
    if frame is None:
        last_frame = st.session_state.get("last_good_frame_" + source)
        if last_frame is not None:
            return last_frame
        return np.zeros((360, 640, 3), dtype=np.uint8)

    # If frame exists, flip and store it
    flipped = cv2.flip(frame, 1)
    st.session_state["last_good_frame_" + source] = flipped
    return flipped
