import streamlit as st
import cv2
from ultralytics import YOLO
import pygame
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YOLO Crowd Monitor", layout="wide")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- PATH ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sound_path = PROJECT_ROOT / "alert_sound.mp3"

# ---------------- SESSION ----------------
if "sound_on" not in st.session_state:
    st.session_state.sound_on = False

# ---------------- PYGAME INIT ----------------
pygame.mixer.init()
if sound_path.exists():
    pygame.mixer.music.load(str(sound_path))

# ---------------- BLINK ----------------
def blinking_banner(text, color):
    st.markdown(
        f"""
        <style>
        @keyframes blink {{
            0% {{opacity:1;}}
            50% {{opacity:0;}}
            100% {{opacity:1;}}
        }}
        .alert {{
            background:{color};
            color:white;
            padding:20px;
            text-align:center;
            font-size:26px;
            font-weight:bold;
            border-radius:10px;
            animation:blink 1s infinite;
        }}
        </style>
        <div class="alert">{text}</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- UI ----------------
st.title("🎥 YOLO Live Crowd Monitoring")

st.sidebar.header("⚙️ Threshold Settings")
HIGH_THRESHOLD = st.sidebar.slider("High", 1, 20, 3)
CRITICAL_THRESHOLD = st.sidebar.slider("Critical", 1, 50, 6)

start = st.button("▶️ Start Camera")
frame_window = st.image([])

# ---------------- MAIN ----------------
if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        results = model(frame)

        count = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

        # ---------------- ALERT ----------------
        if count >= CRITICAL_THRESHOLD:
            status = "🚨 CRITICAL"
            color = (0,0,255)

            blinking_banner("🚨 CRITICAL CROWD ALERT 🚨", "red")

            # 🔊 PLAY (LOOP)
            if not st.session_state.sound_on:
                pygame.mixer.music.play(-1)
                st.session_state.sound_on = True

        elif count >= HIGH_THRESHOLD:
            status = "⚠️ HIGH"
            color = (0,165,255)

            blinking_banner("⚠️ HIGH CROWD ALERT ⚠️", "orange")

            # 🔇 STOP
            if st.session_state.sound_on:
                pygame.mixer.music.stop()
                st.session_state.sound_on = False

        else:
            status = "✅ SAFE"
            color = (0,255,0)

            # 🔇 STOP
            if st.session_state.sound_on:
                pygame.mixer.music.stop()
                st.session_state.sound_on = False

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Count: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Status: {status}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()