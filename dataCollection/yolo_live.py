import cv2
from ultralytics import YOLO
import pygame
import os

# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- ALERT SETTINGS ----------------
HIGH_THRESHOLD = 1
CRITICAL_THRESHOLD = 2

# ---------------- SOUND SETUP ----------------
pygame.mixer.init()

sound_path = os.path.join(os.path.dirname(__file__), "..", "alert_sound.mp3")
pygame.mixer.music.load(sound_path)

is_playing = False


def run_yolo():
    global is_playing

    cap = cv2.VideoCapture(0)  # webcam

    if not cap.isOpened():
        print("Camera not opening")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- YOLO DETECTION ----------------
        results = model(frame)

        count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # ---------------- ALERT LOGIC ----------------
        alert_text = "SAFE"
        color = (0,255,0)

        if count >= CRITICAL_THRESHOLD:
            alert_text = "CRITICAL 🚨"
            color = (0,0,255)

            # 🔊 PLAY SOUND LOOP
            if not is_playing:
                pygame.mixer.music.play(-1)  # loop
                is_playing = True

        elif count >= HIGH_THRESHOLD:
            alert_text = "HIGH ⚠️"
            color = (0,165,255)

            # sound band
            if is_playing:
                pygame.mixer.music.stop()
                is_playing = False

        else:
            alert_text = "SAFE"
            color = (0,255,0)

            # sound band
            if is_playing:
                pygame.mixer.music.stop()
                is_playing = False

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Count: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Status: {alert_text}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("YOLO Crowd Monitor", frame)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_yolo()