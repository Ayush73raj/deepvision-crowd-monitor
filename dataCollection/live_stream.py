import sys
import os

# Fix path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import numpy as np

# Import model
from src.simple_cnn.model_simplecnn import SimpleCNN

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("models/simple_cnn/best_simplecnn.pth", map_location="cpu"))
model.eval()


def run_webcam():
    cap = cv2.VideoCapture(0)  # 0 = webcam

    if not cap.isOpened():
        print("Camera not opening")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # STEP 1: Resize
        frame_resized = cv2.resize(frame, (640, 480))

        # STEP 2: Convert to model input
        img = frame_resized / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img).unsqueeze(0).float()

        # STEP 3: Predict
        with torch.no_grad():
            output = model(img)

            # FIX negative values
            output = torch.relu(output)

            count = output.sum().item()

        # STEP 4: Show count
        cv2.putText(frame_resized,
                    f"Count: {int(count)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        # STEP 5: Show video
        cv2.imshow("Live Crowd Monitor", frame_resized)

        # STEP 6: Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()