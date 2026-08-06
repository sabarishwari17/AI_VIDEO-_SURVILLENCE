"""
weapon_detection.py
--------------------
Knife / gun detection.

The stock YOLOv8n COCO checkpoint does NOT include "knife" or "gun"
classes, so out of the box this module runs in "disabled / awaiting
custom model" mode and will simply report no weapons.

To enable real weapon detection:
  1. Train (or download) a YOLOv8 model with "knife" and "gun" (or
     "pistol"/"weapon") classes - Roboflow Universe has several public
     weapon-detection datasets you can train against with
     `yolo train data=weapons.yaml model=yolov8n.pt`.
  2. Save the resulting best.pt as models/weapon_yolov8.pt
  3. Restart the app - this module will auto-detect the file and start
     using it.

This module deliberately reuses object_detection.ObjectDetector rather
than re-implementing YOLO inference, so behaviour (confidence
thresholding, box drawing) stays consistent across the app.
"""

import os
import config
from object_detection import ObjectDetector

WEAPON_MODEL_PATH = os.path.join(config.MODELS_FOLDER, "weapon_yolov8.pt")
WEAPON_LABELS = {"knife", "gun", "pistol", "weapon", "rifle"}


class WeaponDetector:
    def __init__(self, confidence=0.5):
        self.enabled = os.path.isfile(WEAPON_MODEL_PATH)
        self.detector = ObjectDetector(model_path=WEAPON_MODEL_PATH, confidence=confidence) \
            if self.enabled else None
        if not self.enabled:
            print("[weapon_detection] No custom weapon model found at "
                  f"{WEAPON_MODEL_PATH} - weapon detection is INACTIVE. "
                  "See README.md 'Custom Models' to enable it.")

    def is_ready(self):
        return self.enabled and self.detector is not None and self.detector.is_ready()

    def detect(self, frame):
        """Returns list of detections with label in WEAPON_LABELS."""
        if not self.is_ready():
            return []
        all_dets = self.detector.detect(frame)
        return [d for d in all_dets if d["label"].lower() in WEAPON_LABELS]
