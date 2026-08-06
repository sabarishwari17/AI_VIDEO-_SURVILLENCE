"""
fire_detection.py
------------------
Lightweight fire & smoke detection.

Fire: HSV colour-thresholding tuned for orange/red/yellow flame
colours + a minimum "hot area" size, which is a common, dependency-free
approach that works reasonably well for close/medium range indoor or
yard cameras without needing a trained model.

Smoke: greyish, low-saturation, textureless moving regions - detected
by combining motion (from motion_detection.py) with a desaturated
colour mask. Smoke detection from a single RGB camera is inherently
less reliable than fire detection; treat it as a secondary heuristic
and expect a higher false-positive rate.

For production-grade accuracy, replace `detect_fire()` with inference
from a custom-trained YOLOv8 fire/smoke model (see README.md
"Custom Models") - the function signature is designed to be a drop-in
swap.
"""

import cv2
import numpy as np


def detect_fire(frame, min_area=1200):
    """
    Returns (fire_detected: bool, boxes: list[(x,y,w,h)], confidence: float)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 180])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 120, 180])
    upper2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    total_fire_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        boxes.append(cv2.boundingRect(c))
        total_fire_area += area

    frame_area = frame.shape[0] * frame.shape[1]
    confidence = min(0.99, total_fire_area / frame_area * 8) if boxes else 0.0

    return (len(boxes) > 0), boxes, round(confidence, 2)


def detect_smoke(frame, motion_boxes, min_area=2500):
    """
    Heuristic smoke detector: looks for low-saturation, mid-brightness
    grey regions overlapping areas where motion was already detected.
    Returns (smoke_detected, boxes, confidence)
    """
    if not motion_boxes:
        return False, [], 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 90])
    upper = np.array([180, 60, 220])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    boxes = []
    for (x, y, w, h) in motion_boxes:
        region = mask[y:y + h, x:x + w]
        if region.size == 0:
            continue
        grey_ratio = np.count_nonzero(region) / region.size
        if grey_ratio > 0.45 and (w * h) > min_area:
            boxes.append((x, y, w, h))

    confidence = 0.5 if boxes else 0.0
    return (len(boxes) > 0), boxes, confidence


def draw_boxes(frame, boxes, label, color):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
