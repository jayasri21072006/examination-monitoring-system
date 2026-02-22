import cv2
import mediapipe as mp
import streamlit as st
from ultralytics import YOLO
import torch


# ---------------- INITIALIZATION ---------------- #

def ensure_detectors() -> None:
    if st.session_state.face_mesh is None:
        st.session_state.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    if st.session_state.hands is None:
        st.session_state.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    if "yolo_model" not in st.session_state:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO("best.pt")
        model.to(device)
        st.session_state.yolo_model = model


def close_detectors() -> None:
    if st.session_state.face_mesh is not None:
        st.session_state.face_mesh.close()
        st.session_state.face_mesh = None

    if st.session_state.hands is not None:
        st.session_state.hands.close()
        st.session_state.hands = None


# ---------------- FEED STATE ---------------- #

def init_feed_state(source: str) -> None:
    if source not in st.session_state.feed_counters:
        st.session_state.feed_counters[source] = {
            "mobile": 0,
            "talking": 0,
            "paper": 0,
            "head_turn": 0,
        }

    if source not in st.session_state.feed_signals:
        st.session_state.feed_signals[source] = {
            "mobile": "No mobile signal",
            "talking": "No talking signal",
            "paper": "No paper signal",
            "head_turn": "No head-turn signal",
            "severity": "NORMAL",
        }

    if source not in st.session_state.feed_risk_scores:
        st.session_state.feed_risk_scores[source] = 0.0


def smooth_counter(source: str, key: str, active: bool, up: int = 1, down: int = 1) -> int:
    current = st.session_state.feed_counters[source][key]
    if active:
        current += up
    else:
        current = max(0, current - down)
    st.session_state.feed_counters[source][key] = current
    return current


# ---------------- MAIN DETECTION ---------------- #

def detect_on_frame(source: str, frame_bgr):
    init_feed_state(source)

    # ---------------- FRAME SKIP (ANTI FREEZE FIX) ---------------- #
    frame_key = f"{source}_frame_count"
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0

    st.session_state[frame_key] += 1

    # Run heavy detection only every 5 frames
    if st.session_state[frame_key] % 5 != 0:
        return st.session_state.feed_signals[source]

    # --------------------------------------------------------------- #

    resized = cv2.resize(frame_bgr, (0, 0), fx=0.7, fy=0.7)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    face_res = st.session_state.face_mesh.process(rgb)
    hands_res = st.session_state.hands.process(rgb)

    signals = {
        "mobile": False,
        "talking": False,
        "paper": False,
        "head_turn": False,
    }

    final_score = 0.0
    downward_alert = False

    face_lm = None
    if face_res.multi_face_landmarks:
        face_lm = face_res.multi_face_landmarks[0].landmark

    # ---------------- FACE LOGIC ---------------- #

    if face_lm is not None:
        nose = face_lm[1]
        left_face = face_lm[234]
        right_face = face_lm[454]

        face_width = max(1e-6, right_face.x - left_face.x)

        # Head Turn
        left_gap = nose.x - left_face.x
        right_gap = right_face.x - nose.x
        turn_score = abs(left_gap - right_gap) / face_width
        signals["head_turn"] = turn_score > 0.24

        # Talking
        upper_lip = face_lm[13]
        lower_lip = face_lm[14]
        left_eye = face_lm[33]
        right_eye = face_lm[263]

        mouth_open = abs(lower_lip.y - upper_lip.y)
        eye_width = max(1e-6, abs(right_eye.x - left_eye.x))
        talk_ratio = mouth_open / eye_width
        signals["talking"] = talk_ratio > 0.27

        # ---------------- DOWNWARD GAZE ---------------- #

        down_key = f"{source}_down_count"
        if down_key not in st.session_state:
            st.session_state[down_key] = 0

        eye_center_y = (left_eye.y + right_eye.y) / 2
        looking_down = eye_center_y > nose.y + (0.18 * face_width)

        if looking_down:
            st.session_state[down_key] += 1
        else:
            st.session_state[down_key] = max(0, st.session_state[down_key] - 1)

        downward_alert = st.session_state[down_key] > 6

        # ---------------- HAND LOGIC ---------------- #

        hand_near_head = False
        hand_low_hold = False

        if hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:

                xs = [p.x for p in hand.landmark]
                ys = [p.y for p in hand.landmark]

                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)

                near_face = (
                    abs(cx - nose.x) < (face_width * 1.7)
                    and abs(cy - nose.y) < (face_width * 1.9)
                )

                hand_w = max(xs) - min(xs)
                hand_h = max(ys) - min(ys)
                vertical_hand = hand_h > (hand_w * 1.1)

                if near_face and vertical_hand:
                    hand_near_head = True

                avg_y = sum(ys) / len(ys)
                if avg_y > nose.y + (0.25 * face_width):
                    hand_low_hold = True

                index_tip = hand.landmark[8]
                prev_key = f"{source}_prev_index_y"

                if prev_key not in st.session_state:
                    st.session_state[prev_key] = index_tip.y

                movement = abs(index_tip.y - st.session_state[prev_key])
                if movement > 0.02:
                    signals["mobile"] = True

                st.session_state[prev_key] = index_tip.y

        # ---------------- RULE SCORE ---------------- #

        rule_mobile = hand_near_head or (downward_alert and hand_low_hold)
        rule_score = 0.8 if rule_mobile else 0.0

        # ---------------- YOLO DETECTION ---------------- #

        yolo_conf = 0.0
        model = st.session_state.yolo_model

        # 🔥 Reduced from 640 → 416 (Performance Boost)
        small_frame = cv2.resize(frame_bgr, (416, 416))
        results = model(small_frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if "mobile" in class_name.lower() or "phone" in class_name.lower():
                yolo_conf = max(yolo_conf, conf)

        # ---------------- FUSION ---------------- #

        rule_weight = 0.6
        model_weight = 0.4

        final_score = (rule_weight * rule_score) + (model_weight * yolo_conf)
        st.session_state.feed_risk_scores[source] = final_score

        signals["mobile"] = signals["mobile"] or (final_score > 0.55)

    # ---------------- PAPER DETECTION ---------------- #

    h, _ = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.45):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paper_like = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            _, _, cw, ch = cv2.boundingRect(approx)
            ratio = cw / max(1, ch)
            if 0.6 < ratio < 1.8:
                paper_like = True
                break

    signals["paper"] = paper_like

    # ---------------- SMOOTHING ---------------- #

    mobile_count = smooth_counter(source, "mobile", signals["mobile"])
    talking_count = smooth_counter(source, "talking", signals["talking"])
    paper_count = smooth_counter(source, "paper", signals["paper"])
    turn_count = smooth_counter(source, "head_turn", signals["head_turn"])

    mobile_alert = mobile_count >= 3
    talking_alert = talking_count >= 5
    paper_alert = paper_count >= 4
    turn_alert = turn_count >= 5

    # ---------------- RISK SCORE ---------------- #

    risk_score = (
        0.5 * st.session_state.feed_risk_scores[source]
        + 0.2 * int(downward_alert)
        + 0.15 * int(talking_alert)
        + 0.1 * int(paper_alert)
        + 0.05 * int(turn_alert)
    )

    if risk_score > 0.75:
        severity = "HIGH ALERT"
    elif risk_score > 0.5:
        severity = "ALERT"
    elif risk_score > 0.3:
        severity = "WARNING"
    else:
        severity = "NORMAL"

    feed_text = {
        "mobile": f"Mobile risk {final_score:.2f}" if mobile_alert else "No mobile signal",
        "talking": "Talking detected" if talking_alert else "No talking signal",
        "paper": "Paper detected in desk zone" if paper_alert else "No paper signal",
        "head_turn": "Repeated head turning" if turn_alert else "No head-turn signal",
        "severity": severity,
    }

    st.session_state.feed_signals[source] = feed_text
    return feed_text