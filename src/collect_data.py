"""
Step 1: Data Collection
------------------------
Records hand landmark coordinates from your webcam for a set of gestures
you define. MediaPipe detects 21 landmark points per hand (fingertips,
knuckles, wrist, etc.) -- we save these coordinates, labeled with
whichever gesture you're currently holding, to build a training dataset.

Controls while running:
- Hold a gesture steady, then press the NUMBER key matching that gesture
  (shown on screen) to record one sample.
- Press 'q' to quit and save everything collected so far.

Aim for at least 50-100 samples per gesture for a reasonably reliable
classifier -- vary your hand's distance/angle slightly between samples
so the model doesn't overfit to one exact position.
"""

import cv2
import mediapipe as mp
import csv
import os

# Define your gesture classes here -- change these to whatever you want
GESTURES = {
    ord('1'): "fist",
    ord('2'): "open_palm",
    ord('3'): "thumbs_up",
    ord('4'): "peace_sign",
    ord('5'): "point",
}

OUTPUT_PATH = "../data/gestures.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

def landmarks_to_row(hand_landmarks):
    """Convert landmarks to wrist-relative, scale-normalized coordinates.
    This makes the features describe hand SHAPE, not position in frame."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist_x, wrist_y, wrist_z = pts[0]

    # Translate: make wrist the origin
    rel = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in pts]

    # Scale: normalize by distance from wrist to middle-finger MCP (landmark 9)
    # -- a stable reference for "hand size" regardless of distance from camera
    scale = (rel[9][0]**2 + rel[9][1]**2 + rel[9][2]**2) ** 0.5
    scale = scale if scale > 1e-6 else 1.0

    row = []
    for x, y, z in rel:
        row.extend([x / scale, y / scale, z / scale])
    return row


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera permissions/index.")
        return

    csv_file = open(OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header + ["label"])

    counts = {name: 0 for name in GESTURES.values()}

    print("Recording controls:")
    for key, name in GESTURES.items():
        print(f"  Press '{chr(key)}' for gesture: {name}")
    print("  Press 'q' to quit and save.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show instructions and running counts on screen
        y = 30
        for key, name in GESTURES.items():
            cv2.putText(frame, f"[{chr(key)}] {name}: {counts[name]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        cv2.imshow("Data Collection - press number keys to record", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in GESTURES and results.multi_hand_landmarks:
            row = landmarks_to_row(results.multi_hand_landmarks[0])
            label = GESTURES[key]
            writer.writerow(row + [label])
            counts[label] += 1
            print(f"Recorded sample for '{label}' (total: {counts[label]})")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\nData saved to", OUTPUT_PATH)
    print("Counts:", counts)


if __name__ == "__main__":
    main()