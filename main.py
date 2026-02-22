import streamlit as st
import time
import random
import csv
import os
from datetime import datetime

# =====================================================
# CONFIGURATION (Policy Aligned)
# =====================================================
REFRESH_INTERVAL = 3  # seconds
ESCALATION_THRESHOLD = 85
LOG_DIRECTORY = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "exam_events.csv")

SEAT_IDS = [
    "Seat-A1", "Seat-A2", "Seat-B1", "Seat-B2"
]

BEHAVIOR_CATEGORIES = {
    "NORMAL": 0,
    "REPEATED_HEAD_TURNING": 10,
    "SLEEPING_OR_DISENGAGED": 15,
    "GROUP_DISCUSSION": 20,
    "MOBILE_PHONE_USAGE": 30,
    "COPYING_BEHAVIOR": 35,
    "ARGUMENT_WITH_INVIGILATOR": 25
}

# =====================================================
# INITIALIZATION
# =====================================================
st.set_page_config(
    page_title="Examination Surveillance Dashboard",
    layout="wide"
)

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "seat_id",
            "risk_score",
            "observed_behavior",
            "system_action"
        ])

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def log_event(seat_id, risk, behavior, action):
    with open(LOG_FILE_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            seat_id,
            int(risk),
            behavior,
            action
        ])

def simulate_behavior():
    """Simulates realistic exam hall activity"""
    weighted = (
        ["NORMAL"] * 6 +
        ["REPEATED_HEAD_TURNING"] * 2 +
        ["SLEEPING_OR_DISENGAGED"] +
        ["GROUP_DISCUSSION"] +
        ["MOBILE_PHONE_USAGE"] +
        ["COPYING_BEHAVIOR"]
    )
    return random.choice(weighted)

def update_risk(previous_risk, behavior):
    if behavior == "NORMAL":
        return max(0, previous_risk - 5)
    return min(100, previous_risk + BEHAVIOR_CATEGORIES[behavior])

def derive_status(risk):
    if risk < 30:
        return "NORMAL"
    elif risk < 60:
        return "OBSERVATION_REQUIRED"
    elif risk < 85:
        return "SUSPICIOUS_ACTIVITY"
    else:
        return "ESCALATION_RECOMMENDED"

# =====================================================
# SESSION STATE
# =====================================================
if "risk_scores" not in st.session_state:
    st.session_state.risk_scores = {seat: 0 for seat in SEAT_IDS}

# =====================================================
# UI HEADER
# =====================================================
st.title("AI-Assisted Examination Monitoring System")
st.caption(
    "Decision-Support System | Passive Surveillance | Policy-Compliant"
)

st.sidebar.header("System Control")
monitoring_active = st.sidebar.checkbox(
    "Enable Monitoring", value=True
)

st.sidebar.markdown("---")
st.sidebar.write(
    "This system does not replace invigilators.\n"
    "It provides analytical support through CCTV feeds."
)

# =====================================================
# MAIN DASHBOARD LOOP
# =====================================================
dashboard = st.empty()

while monitoring_active:
    with dashboard.container():
        st.subheader("Live Examination Hall Overview")

        columns = st.columns(len(SEAT_IDS))

        for index, seat in enumerate(SEAT_IDS):
            with columns[index]:
                observed_behavior = simulate_behavior()
                previous_risk = st.session_state.risk_scores[seat]
                current_risk = update_risk(previous_risk, observed_behavior)
                st.session_state.risk_scores[seat] = current_risk

                status = derive_status(current_risk)

                st.metric(
                    label=seat,
                    value=f"{current_risk} %",
                    delta=observed_behavior.replace("_", " ")
                )

                st.write(f"Status: **{status}**")

                # Logging rules
                if current_risk >= 60:
                    log_event(
                        seat,
                        current_risk,
                        observed_behavior,
                        "LOGGED_FOR_REVIEW"
                    )

                if current_risk >= ESCALATION_THRESHOLD:
                    st.warning("Escalation recommended to supervisory authority")
                    log_event(
                        seat,
                        current_risk,
                        observed_behavior,
                        "ESCALATION_RECOMMENDED"
                    )

        st.markdown("---")
        st.caption(
            "Escalation is triggered only after sustained abnormal behavior. "
            "Final action rests with examination authorities."
        )

    time.sleep(REFRESH_INTERVAL)
