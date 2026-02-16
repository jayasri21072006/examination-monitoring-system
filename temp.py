import cv2, time, threading, io
import numpy as np
import mediapipe as mp
import streamlit as st
import sounddevice as sd
from collections import defaultdict, deque
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("Exam Integrity Monitoring", layout="wide")
st.title("🛡️ Exam Integrity Monitoring – Behavior-Aware System")

event_log = []
active_flags = set()  # (student_id, event_type)
students, student_counter = {}, 1

# =====================================================
# CAMERA THREAD
# =====================================================
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.running = False
        self.cap.release()

camera = Camera()

# =====================================================
# AUDIO THREAD (SAFE)
# =====================================================
speech_energy = deque(maxlen=10)

def audio_worker():
    while True:
        try:
            audio = sd.rec(int(0.25 * 44100), 44100, 1, blocking=True)
            speech_energy.append(np.linalg.norm(audio))
        except:
            speech_energy.append(0)

threading.Thread(target=audio_worker, daemon=True).start()

# =====================================================
# MEDIAPIPE
# =====================================================
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2)

# =====================================================
# TRACKERS
# =====================================================
head_turn = defaultdict(lambda: deque(maxlen=25))
down_glance = defaultdict(lambda: deque(maxlen=25))
mouth_motion = defaultdict(lambda: deque(maxlen=15))

phone_near = defaultdict(lambda: deque(maxlen=20))
screen_light = defaultdict(lambda: deque(maxlen=20))
scroll_motion = defaultdict(lambda: deque(maxlen=15))

# =====================================================
# STUDENT ID
# =====================================================
def get_student_id(x):
    global student_counter
    for sid, px in students.items():
        if abs(px - x) < 0.05:
            students[sid] = x
            return sid
    sid = f"Student-{student_counter}"
    students[sid] = x
    student_counter += 1
    return sid

# =====================================================
# EVENT LOGGER (ONCE ONLY)
# =====================================================
def log_event(sid, event_type, message):
    key = (sid, event_type)
    if key in active_flags:
        return
    active_flags.add(key)
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {sid} – {message}"
    event_log.append(entry)
    st.sidebar.error(entry)

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(40, 800, "Exam Integrity Monitoring Report")
    y = 760
    for e in event_log:
        if y < 50:
            c.showPage()
            y = 760
        c.drawString(40, y, e)
        y -= 18
    c.save()
    buffer.seek(0)
    return buffer

# =====================================================
# UI
# =====================================================
run = st.checkbox("▶ Start Monitoring")
frame_slot = st.empty()
st.sidebar.title("🚨 Confirmed Events")

# =====================================================
# MAIN LOOP
# =====================================================
while run:
    frame = camera.frame
    if frame is None:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_res = face_mesh.process(rgb)
    hand_res = hands.process(rgb)

    if face_res.multi_face_landmarks:
        for face in face_res.multi_face_landmarks:
            nose = face.landmark[1]
            mouth = face.landmark[13]
            sid = get_student_id(nose.x)

            h, w, _ = frame.shape
            cx, cy = int(nose.x * w), int(nose.y * h)
            cv2.putText(frame, sid, (cx - 40, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ================= SPEAKING =================
            mouth_motion[sid].append(abs(mouth.y - nose.y))
            if len(speech_energy) > 0:
                avg_audio = sum(speech_energy) / len(speech_energy)
                if avg_audio > 0.05 and sum(mouth_motion[sid]) > 0.3:
                    log_event(sid, "speech", "Verbal Communication Detected")

            # ================= HEAD TURN =================
            head_turn[sid].append(nose.x)
            if len(head_turn[sid]) == 25 and max(head_turn[sid]) - min(head_turn[sid]) > 0.18:
                log_event(sid, "head_turn", "Repeated Head Turning")

            # ================= BIT PAPER =================
            down_glance[sid].append(nose.y > 0.7)
            if sum(down_glance[sid]) > 18:
                log_event(sid, "bit_paper", "Unauthorized Reference Material Usage")

            # ================= HAND & MOBILE BEHAVIOR =================
            prev_y = None
            if hand_res.multi_hand_landmarks:
                for hand in hand_res.multi_hand_landmarks:
                    wrist = hand.landmark[0]

                    phone_near[sid].append(abs(wrist.x - nose.x) < 0.1)

                    if prev_y is not None:
                        scroll_motion[sid].append(abs(wrist.y - prev_y) > 0.015)
                    prev_y = wrist.y

            screen_light[sid].append(np.mean(gray) > 160)

            # ================= MOBILE DECISION (ETHICAL) =================
            mobile_score = 0
            if sum(phone_near[sid]) > 10:
                mobile_score += 1
            if sum(screen_light[sid]) > 10:
                mobile_score += 1
            if sum(scroll_motion[sid]) > 6 or sum(down_glance[sid]) > 15:
                mobile_score += 1

            if mobile_score >= 2:
                log_event(
                    sid,
                    "mobile",
                    "Mobile-like Device Interaction (Behavior Confirmed)"
                )

    frame_slot.image(rgb, channels="RGB")
    time.sleep(0.02)

camera.stop()

# =====================================================
# DOWNLOAD REPORT
# =====================================================
st.markdown("---")
if st.button("📄 Download Exam Report"):
    pdf = generate_pdf()
    st.download_button(
        "⬇ Download PDF",
        data=pdf,
        file_name="exam_integrity_report.pdf",
        mime="application/pdf"
    )
