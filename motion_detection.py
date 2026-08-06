"""
motion_detection.py
--------------------
Simple, fast background-subtraction based motion detector using OpenCV.

Used as the first-stage trigger: motion detection runs on every frame
(cheap) and only when motion is found do we run the more expensive
YOLO / face detection passes on that frame (see app.py process loop).
"""

import cv2
import numpy as np

import config


class MotionDetector:
    def __init__(self, threshold=None, min_area=None):
        self.threshold = threshold or config.DEFAULT_MOTION_THRESHOLD
        self.min_area = min_area or config.DEFAULT_MOTION_MIN_AREA
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=False
        )

    def detect(self, frame):
        """
        Returns (motion_detected: bool, boxes: list[(x,y,w,h)], mask)
        """
        fg_mask = self.back_sub.apply(frame)
        fg_mask = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        return (len(boxes) > 0), boxes, fg_mask

    @staticmethod
    def draw_boxes(frame, boxes, color=(0, 255, 255), label="Motion"):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
