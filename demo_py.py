import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam error")
    exit()

# ---------- MEDIAPIPE ----------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=1)
face = mp_face.FaceMesh(max_num_faces=1)

# ---------- VARIABLES ----------
motion_points = deque(maxlen=30)
energy_window = deque(maxlen=15)

baseline_samples = []
baseline_ready = False
baseline_energy = 0

stress_frames = 0
freeze_frames = 0
last_bbox_area = None

help_timer = 0
sos_timer = 0

start_time = time.time()

# ---------- FUNCTIONS ----------
def motion_energy(lm, w, h):
    cx = int(lm[0].x * w)
    cy = int(lm[0].y * h)
    motion_points.append((cx, cy))
    if len(motion_points) < 2:
        return 0
    dx = motion_points[-1][0] - motion_points[-2][0]
    dy = motion_points[-1][1] - motion_points[-2][1]
    return math.sqrt(dx*dx + dy*dy)

def bbox_area(lm, w, h):
    xs = [int(l.x*w) for l in lm]
    ys = [int(l.y*h) for l in lm]
    return (max(xs)-min(xs))*(max(ys)-min(ys))

def neon_line(img, p1, p2, color):
    cv2.line(img, p1, p2, color, 8)
    cv2.line(img, p1, p2, (255,255,255), 2)

def detect_hand_gesture(lm):
    fingers = []
    for tip in [8, 12, 16, 20]:
        fingers.append(lm[tip].y < lm[tip-2].y)
    if all(fingers):
        return "HELP"
    if not any(fingers):
        return "SOS"
    return None

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose.process(rgb)
    hand_res = hands.process(rgb)
    face_res = face.process(rgb)

    canvas = np.zeros_like(frame)
    state = "NORMAL"
    reason = "Stable behavior"
    color = (0,255,255)

    # -------- POSE LOGIC --------
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        energy = motion_energy(lm, w, h)
        energy_window.append(energy)
        avg_energy = np.mean(energy_window)

        area = bbox_area(lm, w, h)

        if not baseline_ready:
            baseline_samples.append(avg_energy)
            if time.time() - start_time > 10:
                baseline_energy = np.mean(baseline_samples) + 8
                baseline_ready = True

        if baseline_ready:
            if avg_energy > baseline_energy * 1.4:
                stress_frames += 1
            else:
                stress_frames = max(0, stress_frames-1)

            if stress_frames > 8:
                state = "STRESS"
                color = (0,165,255)
                reason = "Repeated unstable movement"

            if stress_frames > 15:
                state = "DANGER"
                color = (0,0,255)
                reason = "Sustained panic motion"

            if stress_frames > 10 and avg_energy < 3:
                freeze_frames += 1
                if freeze_frames > 10:
                    state = "FREEZE RISK"
                    color = (255,0,255)
                    reason = "Freeze after stress"
            else:
                freeze_frames = 0

            if last_bbox_area and area > last_bbox_area * 1.4 and stress_frames > 8:
                state = "PROXIMITY THREAT"
                color = (0,0,200)
                reason = "Rapid approach"

        last_bbox_area = area

        for c in mp_pose.POSE_CONNECTIONS:
            p1, p2 = lm[c[0]], lm[c[1]]
            neon_line(canvas,
                (int(p1.x*w), int(p1.y*h)),
                (int(p2.x*w), int(p2.y*h)),
                color)

    # -------- HAND GESTURE --------
    if hand_res.multi_hand_landmarks:
        for hand in hand_res.multi_hand_landmarks:
            gesture = detect_hand_gesture(hand.landmark)
            mp.solutions.drawing_utils.draw_landmarks(
                canvas, hand, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec((0,255,0),2)
            )

            if gesture == "HELP":
                help_timer += 1
                if help_timer > 30:
                    state = "HELP SIGNAL"
                    color = (0,255,0)
                    reason = "Open palm help gesture"
            elif gesture == "SOS":
                sos_timer += 1
                if sos_timer > 40:
                    state = "SOS TRIGGERED"
                    color = (0,0,255)
                    reason = "Closed fist SOS"
            else:
                help_timer = sos_timer = 0

    # -------- FACE --------
    if face_res.multi_face_landmarks:
        for flm in face_res.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                canvas, flm, mp_face.FACEMESH_CONTOURS,
                mp.solutions.drawing_utils.DrawingSpec((0,255,255),1)
            )

    # -------- TEXT --------
    cv2.putText(canvas, f"STATE: {state}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(canvas, f"REASON: {reason}", (30,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Neon Women Safety – Stable System", canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()