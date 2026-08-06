"""
face_detection.py
------------------
Fast face detection using OpenCV's built-in Haar Cascade classifier
(no extra model download required - ships inside opencv-python).
"""

import cv2


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Returns list of (x, y, w, h) face boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [tuple(f) for f in faces]

    @staticmethod
    def draw_boxes(frame, faces, color=(255, 200, 0), labels=None):
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if labels and i < len(labels):
                cv2.putText(frame, labels[i], (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
