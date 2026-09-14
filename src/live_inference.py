"""
Step 3: Live Inference (Prettified)
------------------------------------
Runs your trained classifier live against your webcam feed, with a
styled HUD overlay: a translucent info panel, color-coded gesture
labels, a live confidence bar, and clean custom landmark drawing.
"""

import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np
import time

MODEL_PATH = "../models/gesture_classifier.pkl"

# One accent color per gesture (BGR, since OpenCV) -- tweak to taste
GESTURE_COLORS = {
    "fist":       (60, 60, 220),    # red
    "open_palm":  (60, 200, 60),    # green
    "thumbs_up":  (0, 200, 255),    # amber
    "peace_sign": (220, 130, 60),   # blue
    "point":      (200, 60, 200),   # magenta
}
DEFAULT_COLOR = (200, 200, 200)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS


def landmarks_to_row(hand_landmarks):
    """Wrist-relative, scale-normalized coordinates (shape, not position)."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wx, wy, wz = pts[0]
    rel = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    scale = (rel[9][0] ** 2 + rel[9][1] ** 2 + rel[9][2] ** 2) ** 0.5
    scale = scale if scale > 1e-6 else 1.0
    row = []
    for x, y, z in rel:
        row.extend([x / scale, y / scale, z / scale])
    return row


def draw_landmarks_styled(frame, hand_landmarks, color):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # bones
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # joints -- fingertips slightly bigger
    tip_ids = {4, 8, 12, 16, 20}
    for i, p in enumerate(pts):
        r = 6 if i in tip_ids else 4
        cv2.circle(frame, p, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, color, 2, cv2.LINE_AA)


def draw_panel(frame, label, confidence, color, fps):
    h, w, _ = frame.shape
    panel_w, panel_h = 320, 130
    x0, y0 = 20, 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), color, 2, cv2.LINE_AA)

    # gesture label
    cv2.putText(frame, label.upper(), (x0 + 16, y0 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, color, 2, cv2.LINE_AA)

    # confidence bar
    bar_x, bar_y, bar_w, bar_h = x0 + 16, y0 + 60, panel_w - 32, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_w + 8, bar_y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # fps, small and unobtrusive
    cv2.putText(frame, f"{fps:.0f} FPS", (x0 + 16, y0 + panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    clf = joblib.load(MODEL_PATH)
    feature_names = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running live gesture recognition. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        label, confidence, color = "No hand detected", 0.0, DEFAULT_COLOR

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            row = landmarks_to_row(hand_landmarks)
            X = pd.DataFrame([row], columns=feature_names)

            pred = clf.predict(X)[0]
            confidence = clf.predict_proba(X).max()
            label = pred
            color = GESTURE_COLORS.get(pred, DEFAULT_COLOR)

            draw_landmarks_styled(frame, hand_landmarks, color)

        draw_panel(frame, label, confidence, color, fps)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()