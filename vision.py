import os
import time
import cv2
import numpy as np
from ultralytics import YOLO


class ObstacleDetector:
    def __init__(self, model_path="yolov8n.pt", hazard_path="hazards.pt"):
        self.model = YOLO(model_path)

        self.hazard_model = None
        if os.path.exists(hazard_path):
            self.hazard_model = YOLO(hazard_path)
            print(f"Loaded custom hazard model: {hazard_path}")
        else:
            print(f"Warning: {hazard_path} not found. Running general model only.")

        self.last_spoken = {}
        self.spoken_y2 = {}
        self.prev_y2 = {}
        self.last_seen = {}
        self.cooldown = 2.5

        # Warm up models.
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=320, device="cpu", verbose=False)

        if self.hazard_model is not None:
            dummy_hazard = np.zeros((256, 256, 3), dtype=np.uint8)
            self.hazard_model.predict(
                dummy_hazard, imgsz=256, device="cpu", verbose=False
            )

    def analyze_frame(self, frame, include_boxes=False):
        """
        Default mode returns a list of warnings, preserving the original
        camera-test behavior.

        include_boxes=True returns:
        {
            "warnings": [...],
            "boxes": [
                {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.7,
                 "name": "person", "is_hazard": False}
            ]
        }

        Coordinates are normalized between 0 and 1.
        """
        if frame is None:
            return {"warnings": [], "boxes": []} if include_boxes else []

        # General object detection.
        results = self.model.predict(
            frame,
            imgsz=320,
            conf=0.50,
            iou=0.45,
            max_det=5,
            device="cpu",
            verbose=False
        )[0]

        raw_detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxyn[0])
            name = self.model.names[int(box.cls[0])].lower()
            raw_detections.append((x1, y1, x2, y2, name, False))

        # Custom ground-hazard detection.
        if self.hazard_model is not None:
            hazard_results = self.hazard_model.predict(
                frame,
                imgsz=256,
                conf=0.45,
                iou=0.45,
                max_det=3,
                device="cpu",
                verbose=False
            )[0]

            for box in hazard_results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxyn[0])
                name = self.hazard_model.names[int(box.cls[0])].lower()

                if name == "object":
                    continue

                raw_detections.append(
                    (x1, y1, x2, y2, name, True)
                )

        # Prepare every detected box for the browser overlay.
        detected_boxes = []

        for x1, y1, x2, y2, name, is_hazard in raw_detections:
            detected_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "name": name,
                "is_hazard": is_hazard
            })

        now = time.monotonic()

        # Remove stale object history.
        for key in list(self.last_seen.keys()):
            if now - self.last_seen[key] > 3.5:
                self.spoken_y2.pop(key, None)
                self.prev_y2.pop(key, None)
                self.last_seen.pop(key, None)

        threats = []
        current_y2 = {}

        for x1, y1, x2, y2, name, is_hazard in raw_detections:
            min_distance_y2 = 0.50 if is_hazard else 0.55

            if y2 <= min_distance_y2:
                continue

            center_x = (x1 + x2) * 0.5

            if center_x < 0.30:
                position = "on left"
            elif center_x > 0.70:
                position = "on right"
            else:
                position = "ahead"

            # Ignore side objects unless they appear close.
            if position != "ahead" and y2 < 0.75:
                continue

            key = f"{name}_{position}"
            self.last_seen[key] = now

            delta_frame = y2 - self.prev_y2.get(key, y2)
            is_approaching = delta_frame > 0.04

            current_y2[key] = max(
                y2, current_y2.get(key, 0.0)
            )

            # Avoid repeating stationary-object warnings.
            if key in self.spoken_y2:
                moved_closer = (
                    y2 - self.spoken_y2[key]
                ) >= 0.06

                if not moved_closer and not is_approaching:
                    continue

            urgent_threshold = 0.72 if is_hazard else 0.80

            is_urgent = (
                y2 > urgent_threshold
                or (is_approaching and position == "ahead")
            )

            wait_time = 1.2 if is_urgent else self.cooldown

            if now - self.last_spoken.get(key, 0.0) < wait_time:
                continue

            threats.append((
                is_hazard or is_approaching,
                y2,
                is_urgent,
                is_hazard,
                is_approaching,
                key,
                name,
                position
            ))

        self.prev_y2 = current_y2

        if not threats:
            if include_boxes:
                return {
                    "warnings": [],
                    "boxes": detected_boxes
                }
            return []

        # Select the highest-priority threat.
        threats.sort(reverse=True)

        (
            _,
            y2,
            is_urgent,
            is_hazard,
            is_approaching,
            key,
            name,
            position
        ) = threats[0]

        self.last_spoken[key] = now
        self.spoken_y2[key] = y2

        # Generate the warning in English.
        if is_hazard and is_urgent:
            message = f"Caution! {name} {position}"
        elif is_urgent and is_approaching:
            message = f"Stop! {name} approaching {position}"
        elif is_urgent:
            message = f"Stop! {name} {position}"
        elif is_approaching:
            message = f"Approaching {name} {position}"
        else:
            message = f"{name} {position}"

        warnings = [message]

        if include_boxes:
            return {
                "warnings": warnings,
                "boxes": detected_boxes
            }

        return warnings


# Optional standalone camera test.
if __name__ == "__main__":
    detector = ObstacleDetector()
    cap = cv2.VideoCapture(0)

    print("Camera active. Press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            break

        warnings = detector.analyze_frame(frame)

        if warnings:
            print("VOICE:", warnings[0])

        cv2.imshow("Obstacle Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
