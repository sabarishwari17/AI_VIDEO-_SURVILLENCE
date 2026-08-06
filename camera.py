"""
camera.py
---------
Camera abstraction that supports:
  - Laptop webcam        source = "0"
  - USB camera           source = "1", "2" ...
  - IP camera (MJPEG)    source = "http://192.168.1.10:8080/video"
  - RTSP camera          source = "rtsp://user:pass@192.168.1.10:554/stream1"

Runs its own background thread so frame grabbing never blocks the
Flask request/response cycle, and always exposes the latest frame via
`.read()`.
"""

import cv2
import threading
import time

import config


class Camera:
    def __init__(self, source=None, name="Camera-1"):
        self.name = name
        self.source = self._normalize_source(source or config.DEFAULT_CAMERA_SOURCE)
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_frame_time = 0
        self.connected = False

    @staticmethod
    def _normalize_source(source):
        """Webcam/USB indices arrive as strings ('0') -> convert to int."""
        source = str(source).strip()
        if source.isdigit():
            return int(source)
        return source

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.source)
        # Encourage a reasonable resolution for local webcams (ignored by
        # network streams that don't support it).
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.connected = self.cap.isOpened()
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.connected = False
                time.sleep(1)
                self._reconnect()
                continue
            ok, frame = self.cap.read()
            if not ok:
                self.connected = False
                time.sleep(0.5)
                self._reconnect()
                continue
            self.connected = True
            with self.lock:
                self.frame = frame
                self.last_frame_time = time.time()
            # small sleep to cap CPU usage / target FPS
            time.sleep(max(0, 1.0 / config.TARGET_FPS - 0.005))

    def _reconnect(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.source)

    def read(self):
        """Return the most recent frame (a copy), or None if unavailable."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def is_stale(self, max_age=5):
        return (time.time() - self.last_frame_time) > max_age

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()

    def restart(self, new_source=None):
        self.stop()
        if new_source is not None:
            self.source = self._normalize_source(new_source)
        self.start()
