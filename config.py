"""
config.py
---------
Central configuration for the AI Smart Video Surveillance System.

Static, deployment-level settings (secret key, folder paths, default
camera source) live here as plain Python constants.

Runtime-editable settings (email, Telegram token, WhatsApp number,
camera URL, recording quality, YOLO confidence, detection threshold)
are stored in the SQLite `settings` table and are edited from the
Settings page in the dashboard (see database.py -> get_setting /
set_setting and app.py -> /settings route). This module simply
provides the defaults used the first time the database is created.
"""

import os

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RECORDINGS_FOLDER = os.path.join(BASE_DIR, "recordings")
SNAPSHOTS_FOLDER = os.path.join(BASE_DIR, "snapshots")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
KNOWN_FACES_FOLDER = os.path.join(BASE_DIR, "known_faces")

DATABASE_PATH = os.path.join(DATABASE_FOLDER, "surveillance.db")
EVENTS_CSV_PATH = os.path.join(LOGS_FOLDER, "events.csv")

for folder in (
    UPLOAD_FOLDER,
    RECORDINGS_FOLDER,
    SNAPSHOTS_FOLDER,
    LOGS_FOLDER,
    DATABASE_FOLDER,
    MODELS_FOLDER,
    KNOWN_FACES_FOLDER,
):
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------
SECRET_KEY = os.environ.get("SURVEILLANCE_SECRET_KEY", "change-this-secret-key-in-production")
DEBUG = os.environ.get("SURVEILLANCE_DEBUG", "True") == "True"
HOST = os.environ.get("SURVEILLANCE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SURVEILLANCE_PORT", "5000"))

# ---------------------------------------------------------------------
# Default admin account (created once, on first run, if no users exist)
# Change the password immediately after first login!
# ---------------------------------------------------------------------
DEFAULT_ADMIN_USERNAME = os.environ.get("SURVEILLANCE_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("SURVEILLANCE_ADMIN_PASS", "admin123")

# ---------------------------------------------------------------------
# Camera defaults
#   source can be:
#     0, 1, 2 ...            -> laptop webcam / USB camera index
#     "rtsp://user:pass@ip/" -> RTSP camera
#     "http://ip:port/video" -> IP camera (MJPEG)
# ---------------------------------------------------------------------
DEFAULT_CAMERA_SOURCE = os.environ.get("SURVEILLANCE_CAMERA_SOURCE", "0")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 20

# ---------------------------------------------------------------------
# Detection defaults (overridable at runtime via Settings page)
# ---------------------------------------------------------------------
DEFAULT_YOLO_MODEL = os.path.join(MODELS_FOLDER, "yolov8n.pt")  # auto-downloaded by ultralytics
DEFAULT_YOLO_CONFIDENCE = 0.45
DEFAULT_MOTION_THRESHOLD = 25          # pixel intensity diff threshold
DEFAULT_MOTION_MIN_AREA = 900          # min contour area (pixels) to count as motion

# Classes from the base COCO YOLOv8 model we care about for this project.
# (fire / smoke / knife / gun / helmet require a custom-trained model -
#  see README.md "Custom Models")
COCO_CLASSES_OF_INTEREST = {
    "person", "car", "motorbike", "bus", "truck", "dog", "cat",
    "backpack", "handbag", "cell phone",
}

# Classes that should always trigger a HIGH priority alert + recording
CRITICAL_CLASSES = {"knife", "gun", "fire", "smoke", "weapon"}

# ---------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------
RECORDING_FOURCC = "mp4v"
RECORDING_QUALITY_PRESETS = {
    "low": (426, 240),
    "medium": (640, 480),
    "high": (1280, 720),
}
DEFAULT_RECORDING_QUALITY = "medium"
POST_EVENT_RECORD_SECONDS = 8   # keep recording N seconds after last detection

# ---------------------------------------------------------------------
# Alert cooldown - avoid spamming email/telegram every frame
# ---------------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = 60
