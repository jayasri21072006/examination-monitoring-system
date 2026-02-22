import time
import threading

import cv2
import numpy as np
import streamlit as st

from exam_config import MAX_SCAN_INDEX


def source_to_capture_arg(source: str):
    source = source.strip()
    return int(source) if source.isdigit() else source


def scan_cameras() -> list[str]:
    found = []
    for idx in range(MAX_SCAN_INDEX):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            read_ok, _ = cap.read()
            if read_ok:
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
    # MSMF is usually smoother on newer Windows camera stacks; fallback to DSHOW.
    cap = cv2.VideoCapture(arg, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(arg, cv2.CAP_DSHOW)
    return cap


def _worker_loop(source: str, state: dict) -> None:
    cap = None
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

        ok, frame = cap.read()
        if ok and frame is not None:
            with state["lock"]:
                state["frame"] = frame
                state["status"] = "Connected"
                state["last_ok"] = time.time()
            # Avoid CPU spin when camera delivers frames very quickly.
            time.sleep(0.01)
            continue

        with state["lock"]:
            state["status"] = "Reconnecting"
        try:
            cap.release()
        except Exception:
            pass
        cap = None
        time.sleep(0.15)

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