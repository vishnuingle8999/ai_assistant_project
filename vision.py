import time
import cv2
import numpy as np
from ultralytics import YOLO


class ObstacleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the lightweight YOLO nano model
        self.model = YOLO(model_path)

        # Memory dictionaries to track object state across frames
        self.last_spoken = {}  # Time when the object was last announced
        self.spoken_y2 = {}    # Distance (y2) when the object was last announced
        self.prev_y2 = {}      # Distance (y2) in the previous frame to check speed
        self.last_seen = {}    # Time when the object was last visible on screen

        self.cooldown = 2.5    # Minimum seconds between alerts for the same object

        # Warm up the model once at startup so the first camera frame is instant
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=320, device="cpu", verbose=False)

    def analyze_frame(self, frame):
        if frame is None:
            return []

        results = self.model.predict(
            frame,
            imgsz=320,      # Low resolution for <40ms CPU speed
            conf=0.50,      # Ignore weak/uncertain detections
            iou=0.45,       # Remove duplicate overlapping boxes
            max_det=5,      # Process at most 5 objects per frame
            device="cpu",   # Lock to CPU
            verbose=False
        )[0]

        now = time.monotonic()

        # 1. MEMORY CLEANUP: If an object hasn't been seen for 3.5 seconds
        # (user turned around or walked past it), forget its stationary lock.
        for k in list(self.last_seen.keys()):
            if (now - self.last_seen[k]) > 3.5:
                self.spoken_y2.pop(k, None)
                self.prev_y2.pop(k, None)
                self.last_seen.pop(k, None)

        threats = []
        current_y2 = {}

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxyn[0])

            # Ignore far away objects (bottom edge above 0.55)
            if y2 <= 0.55:
                continue

            # Map horizontal position
            center_x = (x1 + x2) * 0.5
            if center_x < 0.30:
                position = "on left"
            elif center_x > 0.70:
                position = "on right"
            else:
                position = "ahead"

            # Ignore side objects unless they are very close (y2 > 0.75)
            if position != "ahead" and y2 < 0.75:
                continue

            name = self.model.names[int(box.cls[0])]
            key = f"{name}_{position}"
            self.last_seen[key] = now

            # 2. APPROACHING CHECK: Did the object move closer since the last frame?
            # An increase of > 0.04 in y2 means it is actively moving toward the user.
            delta_frame = y2 - self.prev_y2.get(key, y2)
            is_approaching = delta_frame > 0.04
            current_y2[key] = max(y2, current_y2.get(key, 0.0))

            # 3. STATIONARY FILTER: If we already announced this object, check if
            # the distance changed. If it hasn't moved at least 6% closer (0.06)
            # and isn't actively approaching, stay silent!
            if key in self.spoken_y2:
                moved_closer = (y2 - self.spoken_y2[key]) >= 0.06
                if not moved_closer and not is_approaching:
                    continue

            # 4. COOLDOWN CHECK: Prevent rapid voice spam
            is_urgent = (y2 > 0.80) or (is_approaching and position == "ahead")
            wait_time = 1.2 if is_urgent else self.cooldown
            if (now - self.last_spoken.get(key, 0.0)) < wait_time:
                continue

            # Save valid active threat: prioritize approaching objects first, then closest y2
            threats.append((is_approaching, y2, is_urgent, key, name, position))

        # Update previous frame positions for the next loop
        self.prev_y2 = current_y2

        if not threats:
            return []

        # Pick the #1 highest-priority threat (Approaching > Closest to feet)
        threats.sort(reverse=True)
        is_approaching, y2, is_urgent, key, name, position = threats[0]

        # Lock this object's current distance and time so it won't repeat if stationary
        self.last_spoken[key] = now
        self.spoken_y2[key] = y2

        # Format clear voice alert
        if is_urgent and is_approaching:
            return [f"Stop! {name} approaching {position}"]
        if is_urgent:
            return [f"Stop! {name} {position}"]
        if is_approaching:
            return [f"Approaching {name} {position}"]
        return [f"{name} {position}"]


# --- LIVE DEVICE CAMERA TEST ---
if __name__ == "__main__":
    detector = ObstacleDetector()
    cap = cv2.VideoCapture(0)
    print("Camera active. Press 'q' on the video window to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or (cv2.waitKey(1) & 0xFF == ord("q")):
            break

        warnings = detector.analyze_frame(frame)
        if warnings:
            print("VOICE:", warnings[0])

        cv2.imshow("Stationary-Aware Vision Test", frame)

    cap.release()
    cv2.destroyAllWindows()