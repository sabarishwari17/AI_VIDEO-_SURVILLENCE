"""
face_recognition_module.py
---------------------------
Face RECOGNITION (as opposed to face_detection.py, which only finds
face boxes). Named with a "_module" suffix so it never shadows the
third-party "face_recognition" PyPI package.

Uses OpenCV's LBPH (Local Binary Patterns Histograms) recognizer
(cv2.face.LBPHFaceRecognizer_create), which ships in
opencv-contrib-python. This avoids the dlib/cmake build headaches of
the popular "face_recognition" library while still giving solid
accuracy for a home/office surveillance use case.

Workflow
--------
1. Enroll a new authorized person -> capture N face crops from the
   webcam, store them under known_faces/<name>/, train/update the
   LBPH model, and record the person in the `known_faces` DB table.
2. At recognition time -> detect faces (face_detection.py), crop +
   resize + grayscale each face, run recognizer.predict() which
   returns (label_id, distance). Low distance = confident match.
3. Any face that does not match a known label within the confidence
   threshold is treated as an "Unknown" / intruder face (see
   alert.py / app.py process loop).
"""

import os
import cv2
import numpy as np

import config
import database

FACE_SIZE = (200, 200)
# LBPH "distance": lower = more confident match. Empirically tuned.
UNKNOWN_DISTANCE_THRESHOLD = 70


class FaceRecognizer:
    def __init__(self):
        self.available = hasattr(cv2, "face")
        if not self.available:
            print("[face_recognition_module] cv2.face not available - "
                  "install opencv-contrib-python to enable face recognition.")
            self.recognizer = None
        else:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_names = {}   # label_id -> name
        self.model_path = os.path.join(config.MODELS_FOLDER, "lbph_model.yml")
        self._load()

    def _load(self):
        if not self.available:
            return
        if os.path.isfile(self.model_path):
            try:
                self.recognizer.read(self.model_path)
            except Exception as e:
                print(f"[face_recognition_module] Could not load model: {e}")
        for row in database.get_known_faces():
            self.label_names[row["label_id"]] = row["name"]

    def is_trained(self):
        return self.available and len(self.label_names) > 0

    @staticmethod
    def _prep(face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, FACE_SIZE)
        gray = cv2.equalizeHist(gray)
        return gray

    def recognize(self, face_img):
        """
        Returns (name, distance) or ("Unknown", distance)
        """
        if not self.available or not self.is_trained():
            return "Unknown", 999.0
        gray = self._prep(face_img)
        label_id, distance = self.recognizer.predict(gray)
        if distance <= UNKNOWN_DISTANCE_THRESHOLD and label_id in self.label_names:
            return self.label_names[label_id], distance
        return "Unknown", distance

    def enroll(self, name, face_images):
        """
        Train (or update) the recognizer with a new person.
        `face_images` is a list of BGR/gray face crops (numpy arrays).
        """
        if not self.available:
            raise RuntimeError("cv2.face module not available - install opencv-contrib-python")

        label_id = database.next_label_id()
        person_dir = os.path.join(config.KNOWN_FACES_FOLDER, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        samples, labels = [], []
        for i, img in enumerate(face_images):
            gray = self._prep(img)
            samples.append(gray)
            labels.append(label_id)
            cv2.imwrite(os.path.join(person_dir, f"{i:03d}.jpg"), gray)

        if len(samples) == 0:
            raise ValueError("No face samples captured")

        # Retrain from scratch on ALL known faces (existing + new) so the
        # LBPH model stays consistent. This is fast enough for a handful
        # of enrolled people (typical home/office use case).
        all_samples, all_labels = list(samples), list(labels)
        for row in database.get_known_faces():
            existing_dir = os.path.join(
                config.KNOWN_FACES_FOLDER, row["name"].replace(" ", "_")
            )
            if not os.path.isdir(existing_dir):
                continue
            for fname in os.listdir(existing_dir):
                img = cv2.imread(os.path.join(existing_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    all_samples.append(cv2.resize(img, FACE_SIZE))
                    all_labels.append(row["label_id"])

        self.recognizer.train(all_samples, np.array(all_labels))
        self.recognizer.save(self.model_path)

        database.add_known_face(name, label_id, len(samples))
        self.label_names[label_id] = name
        return label_id
