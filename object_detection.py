"""
object_detection.py
--------------------
YOLOv8 (Ultralytics) wrapper used for general object detection:
person, car, bike, bus, truck, dog, cat, bag, mobile phone, etc.

The base "yolov8n.pt" checkpoint (auto-downloaded by ultralytics on
first run) is trained on the 80-class COCO dataset. COCO does NOT
include "helmet", "fire", "smoke", "knife" or "gun" - those are
handled separately:
  - fire / smoke  -> fire_detection.py   (colour/heuristic based)
  - knife / gun   -> weapon_detection.py (plugs in a custom model if
                     you provide one; see README.md "Custom Models")

If ultralytics / the model weights are not available (e.g. offline
first install), this module degrades gracefully and simply returns no
detections instead of crashing the whole app.
"""

import os
import cv2

import config

_YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    _YOLO_AVAILABLE = False


class ObjectDetector:
    def __init__(self, model_path=None, confidence=None):
        self.confidence = confidence or config.DEFAULT_YOLO_CONFIDENCE
        self.model = None
        self.class_names = {}

        if not _YOLO_AVAILABLE:
            print("[object_detection] ultralytics not installed - object detection disabled.")
            return

        model_path = model_path or config.DEFAULT_YOLO_MODEL
        try:
            # If a custom local weights file exists, use it; otherwise fall
            # back to the standard "yolov8n.pt" name so ultralytics can
            # auto-download it.
            source = model_path if os.path.isfile(model_path) else "yolov8n.pt"
            self.model = YOLO(source)
            self.class_names = self.model.names
        except Exception as e:
            print(f"[object_detection] Failed to load YOLO model: {e}")
            self.model = None

    def is_ready(self):
        return self.model is not None

    def detect(self, frame):
        """
        Runs YOLO on a frame.
        Returns list of dicts: {label, confidence, box:(x1,y1,x2,y2)}
        """
        if self.model is None:
            return []

        results = self.model.predict(
            frame, conf=self.confidence, verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.class_names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": (x1, y1, x2, y2),
                })
        return detections

    @staticmethod
    def draw_detections(frame, detections, color=(0, 200, 0)):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f'{det["label"]} {det["confidence"]*100:.0f}%'
            box_color = (0, 0, 255) if det["label"] in config.CRITICAL_CLASSES else color
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        return frame
