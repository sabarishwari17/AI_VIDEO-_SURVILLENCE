"""
app.py
------
Main Flask application for the AI Smart Video Surveillance System.

Responsibilities:
  - Serves the dashboard, login, settings, recordings, alerts pages.
  - Runs the background video processing pipeline (motion -> object ->
    face -> fire/smoke/weapon detection -> recording -> alerting).
  - Streams the annotated live camera feed as MJPEG (multipart/x-mixed-replace)
    so it can be shown directly in an <img> tag in the browser.
  - Exposes small JSON APIs used by the dashboard's JavaScript
    (stats, system info, recent events) via AJAX polling.

Run with:
    python app.py
"""

import os
import time
import threading
from datetime import datetime

import cv2
import psutil
from flask import (
    Flask, render_template, Response, request, redirect, url_for,
    flash, jsonify, session, send_from_directory, abort
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)

import config
import database
from camera import Camera
from motion_detection import MotionDetector
from object_detection import ObjectDetector
from face_detection import FaceDetector
from face_recognition_module import FaceRecognizer
import fire_detection
from weapon_detection import WeaponDetector
import alert as alert_module

# ---------------------------------------------------------------------
# Flask + Login setup
# ---------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.username = row["username"]
        self.role = row["role"]


@login_manager.user_loader
def load_user(user_id):
    row = database.get_user_by_id(user_id)
    return User(row) if row else None


# ---------------------------------------------------------------------
# Global surveillance engine state
# ---------------------------------------------------------------------
database.init_db()

camera = Camera(source=database.get_setting("camera_source"), name="Camera-1")
motion_detector = MotionDetector()
object_detector = ObjectDetector(confidence=float(database.get_setting("yolo_confidence", 0.45)))
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
weapon_detector = WeaponDetector()

engine_lock = threading.Lock()
engine_state = {
    "latest_annotated_frame": None,
    "fps": 0.0,
    "detection_enabled": database.get_setting("detection_enabled", "true") == "true",
    "recording": False,
    "current_recording_path": None,
    "last_detections": [],
    "last_faces": [],
    "camera_status": "connecting",
}

_video_writer = None
_video_writer_lock = threading.Lock()
_recording_meta = {"rec_id": None, "start_time": None, "path": None, "last_event_time": 0}


# ---------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------
def _start_recording(frame_shape, trigger_reason):
    global _video_writer, _recording_meta
    with _video_writer_lock:
        if _video_writer is not None:
            return  # already recording
        quality = database.get_setting("recording_quality", config.DEFAULT_RECORDING_QUALITY)
        w, h = config.RECORDING_QUALITY_PRESETS.get(quality, (640, 480))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.mp4"
        path = os.path.join(config.RECORDINGS_FOLDER, filename)
        fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC)
        _video_writer = cv2.VideoWriter(path, fourcc, config.TARGET_FPS, (w, h))
        rec_id = database.log_recording_start(filename, path, camera.name, trigger_reason)
        _recording_meta = {
            "rec_id": rec_id, "start_time": time.time(),
            "path": path, "last_event_time": time.time(),
            "target_size": (w, h),
        }
        engine_state["recording"] = True
        engine_state["current_recording_path"] = path


def _write_recording_frame(frame):
    global _video_writer
    with _video_writer_lock:
        if _video_writer is None:
            return
        w, h = _recording_meta["target_size"]
        resized = cv2.resize(frame, (w, h))
        _video_writer.write(resized)


def _stop_recording_if_idle():
    """Stop recording if no new trigger event happened in the last
    config.POST_EVENT_RECORD_SECONDS seconds."""
    global _video_writer, _recording_meta
    with _video_writer_lock:
        if _video_writer is None:
            return
        if time.time() - _recording_meta["last_event_time"] < config.POST_EVENT_RECORD_SECONDS:
            return
        duration = time.time() - _recording_meta["start_time"]
        _video_writer.release()
        _video_writer = None
        size_bytes = os.path.getsize(_recording_meta["path"]) if os.path.isfile(_recording_meta["path"]) else 0
        database.log_recording_end(_recording_meta["rec_id"], duration, size_bytes)
        engine_state["recording"] = False
        engine_state["current_recording_path"] = None


def _touch_recording_event():
    _recording_meta["last_event_time"] = time.time()


def _save_snapshot(frame, prefix="event"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(config.SNAPSHOTS_FOLDER, filename)
    cv2.imwrite(path, frame)
    return path


# ---------------------------------------------------------------------
# Background processing thread
# ---------------------------------------------------------------------
def processing_loop():
    camera.start()
    frame_count = 0
    fps_timer = time.time()

    while True:
        frame = camera.read()
        engine_state["camera_status"] = "online" if camera.connected else "offline"

        if frame is None:
            time.sleep(0.1)
            continue

        annotated = frame.copy()
        detections_summary = []
        faces_summary = []
        trigger_reasons = []

        if engine_state["detection_enabled"]:
            # ---- 1. Motion detection (cheap, always runs) ----
            motion_found, motion_boxes, _ = motion_detector.detect(frame)
            if motion_found:
                motion_detector.draw_boxes(annotated, motion_boxes)
                trigger_reasons.append("motion")

            # ---- 2. Object detection (YOLOv8) ----
            if object_detector.is_ready():
                detections = object_detector.detect(frame)
                object_detector.draw_detections(annotated, detections)
                for d in detections:
                    detections_summary.append(d)
                    if d["label"] in config.COCO_CLASSES_OF_INTEREST or d["label"] in config.CRITICAL_CLASSES:
                        trigger_reasons.append(d["label"])
                        database.log_event(camera.name, d["label"], d["confidence"], "object")
                    if d["label"] in config.CRITICAL_CLASSES:
                        img_path = _save_snapshot(frame, prefix=d["label"])
                        alert_module.send_alert(
                            camera.name, "object", d["label"], d["confidence"],
                            image_path=img_path, severity="high",
                        )

            # ---- 3. Weapon detection (knife / gun, custom model) ----
            weapons = weapon_detector.detect(frame)
            for w in weapons:
                trigger_reasons.append("weapon")
                x1, y1, x2, y2 = w["box"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f'WEAPON: {w["label"]}', (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                img_path = _save_snapshot(frame, prefix="weapon")
                database.log_event(camera.name, w["label"], w["confidence"], "weapon", image_path=img_path)
                alert_module.send_alert(camera.name, "weapon", w["label"], w["confidence"],
                                         image_path=img_path, severity="high")

            # ---- 4. Fire / smoke detection ----
            fire_found, fire_boxes, fire_conf = fire_detection.detect_fire(frame)
            if fire_found:
                fire_detection.draw_boxes(annotated, fire_boxes, "FIRE", (0, 69, 255))
                trigger_reasons.append("fire")
                img_path = _save_snapshot(frame, prefix="fire")
                database.log_event(camera.name, "fire", fire_conf, "fire", image_path=img_path)
                alert_module.send_alert(camera.name, "fire", "fire", fire_conf,
                                         image_path=img_path, severity="high")

            smoke_found, smoke_boxes, smoke_conf = fire_detection.detect_smoke(frame, motion_boxes)
            if smoke_found:
                fire_detection.draw_boxes(annotated, smoke_boxes, "SMOKE", (180, 180, 180))
                trigger_reasons.append("smoke")
                img_path = _save_snapshot(frame, prefix="smoke")
                database.log_event(camera.name, "smoke", smoke_conf, "smoke", image_path=img_path)
                alert_module.send_alert(camera.name, "smoke", "smoke", smoke_conf,
                                         image_path=img_path, severity="medium")

            # ---- 5. Face detection + recognition ----
            if database.get_setting("face_recognition_enabled", "true") == "true":
                faces = face_detector.detect(frame)
                labels = []
                for (x, y, w, h) in faces:
                    face_crop = frame[y:y + h, x:x + w]
                    name, distance = face_recognizer.recognize(face_crop)
                    labels.append(f"{name}")
                    faces_summary.append({"name": name, "distance": distance})

                    if name == "Unknown":
                        trigger_reasons.append("intruder")
                        img_path = _save_snapshot(frame, prefix="intruder")
                        event_id = database.log_event(
                            camera.name, "Unknown Person", None, "intruder", image_path=img_path
                        )
                        alert_module.send_alert(
                            camera.name, "intruder", "Unknown Person", None,
                            image_path=img_path, severity="high", event_id=event_id,
                        )
                face_detector.draw_boxes(annotated, faces, labels=labels)

        # ---- Recording logic ----
        if trigger_reasons:
            if engine_state["recording"] is False:
                _start_recording(frame.shape, ", ".join(sorted(set(trigger_reasons))))
            _touch_recording_event()
        if engine_state["recording"]:
            _write_recording_frame(frame)
        _stop_recording_if_idle()

        # ---- FPS calculation ----
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            engine_state["fps"] = round(frame_count / (time.time() - fps_timer), 1)
            frame_count = 0
            fps_timer = time.time()

        # ---- Overlay HUD (timestamp, camera name, fps) ----
        hud = f'{camera.name} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | FPS: {engine_state["fps"]}'
        cv2.putText(annotated, hud, (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        with engine_lock:
            engine_state["latest_annotated_frame"] = annotated
            engine_state["last_detections"] = detections_summary
            engine_state["last_faces"] = faces_summary


processing_thread = threading.Thread(target=processing_loop, daemon=True)
processing_thread.start()


# ---------------------------------------------------------------------
# MJPEG streaming generator
# ---------------------------------------------------------------------
def mjpeg_generator():
    while True:
        with engine_lock:
            frame = engine_state["latest_annotated_frame"]
        if frame is None:
            time.sleep(0.1)
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(1.0 / config.TARGET_FPS)


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = database.get_user_by_username(username)
        from werkzeug.security import check_password_hash
        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row))
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
@app.route("/")
@login_required
def dashboard():
    today_events = database.get_events_today()
    recent_alerts = database.get_recent_alerts(10)
    recent_recordings = database.get_recordings(6)
    storage_bytes = database.storage_usage_bytes()
    return render_template(
        "dashboard.html",
        today_events=today_events,
        recent_alerts=recent_alerts,
        recent_recordings=recent_recordings,
        storage_mb=round(storage_bytes / (1024 * 1024), 1),
        camera_status=engine_state["camera_status"],
        detection_enabled=engine_state["detection_enabled"],
    )


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/system_stats")
@login_required
def api_system_stats():
    return jsonify({
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "fps": engine_state["fps"],
        "camera_status": engine_state["camera_status"],
        "recording": engine_state["recording"],
        "storage_mb": round(database.storage_usage_bytes() / (1024 * 1024), 1),
    })


@app.route("/api/recent_events")
@login_required
def api_recent_events():
    events = database.get_recent_events(20)
    return jsonify([dict(e) for e in events])


@app.route("/api/toggle_detection", methods=["POST"])
@login_required
def api_toggle_detection():
    engine_state["detection_enabled"] = not engine_state["detection_enabled"]
    database.set_setting("detection_enabled", "true" if engine_state["detection_enabled"] else "false")
    return jsonify({"detection_enabled": engine_state["detection_enabled"]})


# ---------------------------------------------------------------------
# Recordings / playback / search
# ---------------------------------------------------------------------
@app.route("/recordings")
@login_required
def recordings_page():
    date = request.args.get("date")
    object_name = request.args.get("object")
    camera_name = request.args.get("camera")
    if date or object_name or camera_name:
        events = database.search_events(date=date, object_name=object_name, camera=camera_name)
    else:
        events = None
    recs = database.get_recordings(200)
    return render_template("recordings.html", recordings=recs, events=events)


@app.route("/recordings/file/<path:filename>")
@login_required
def serve_recording(filename):
    return send_from_directory(config.RECORDINGS_FOLDER, filename)


@app.route("/snapshots/file/<path:filename>")
@login_required
def serve_snapshot(filename):
    return send_from_directory(config.SNAPSHOTS_FOLDER, filename)


@app.route("/recordings/delete/<int:rec_id>", methods=["POST"])
@login_required
def delete_recording_route(rec_id):
    database.delete_recording(rec_id)
    flash("Recording deleted.", "success")
    return redirect(url_for("recordings_page"))


# ---------------------------------------------------------------------
# Alerts page
# ---------------------------------------------------------------------
@app.route("/alerts")
@login_required
def alerts_page():
    alerts = database.get_recent_alerts(200)
    return render_template("alerts.html", alerts=alerts)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(database.event_stats(period_days=30))


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    if request.method == "POST":
        form = request.form
        updates = {
            "email_enabled": "true" if form.get("email_enabled") else "false",
            "smtp_server": form.get("smtp_server", ""),
            "smtp_port": form.get("smtp_port", "587"),
            "smtp_username": form.get("smtp_username", ""),
            "smtp_password": form.get("smtp_password", ""),
            "alert_email_to": form.get("alert_email_to", ""),

            "telegram_enabled": "true" if form.get("telegram_enabled") else "false",
            "telegram_bot_token": form.get("telegram_bot_token", ""),
            "telegram_chat_id": form.get("telegram_chat_id", ""),

            "whatsapp_enabled": "true" if form.get("whatsapp_enabled") else "false",
            "twilio_account_sid": form.get("twilio_account_sid", ""),
            "twilio_auth_token": form.get("twilio_auth_token", ""),
            "twilio_from_number": form.get("twilio_from_number", ""),
            "twilio_to_number": form.get("twilio_to_number", ""),

            "camera_source": form.get("camera_source", "0"),
            "recording_quality": form.get("recording_quality", "medium"),
            "yolo_confidence": form.get("yolo_confidence", "0.45"),
            "motion_threshold": form.get("motion_threshold", "25"),
            "face_recognition_enabled": "true" if form.get("face_recognition_enabled") else "false",
        }
        database.update_settings(updates)

        # Apply live-updatable settings immediately
        object_detector.confidence = float(updates["yolo_confidence"])
        motion_detector.threshold = int(updates["motion_threshold"])
        if updates["camera_source"] != str(camera.source):
            camera.restart(new_source=updates["camera_source"])

        flash("Settings saved.", "success")
        return redirect(url_for("settings_page"))

    settings = database.get_all_settings()
    return render_template("settings.html", settings=settings)


@app.route("/settings/change_password", methods=["POST"])
@login_required
def change_password():
    new_password = request.form.get("new_password", "")
    confirm = request.form.get("confirm_password", "")
    if len(new_password) < 6:
        flash("Password must be at least 6 characters.", "danger")
    elif new_password != confirm:
        flash("Passwords do not match.", "danger")
    else:
        database.update_password(current_user.id, new_password)
        flash("Password updated.", "success")
    return redirect(url_for("settings_page"))


# ---------------------------------------------------------------------
# Face enrollment (add authorized users)
# ---------------------------------------------------------------------
@app.route("/faces", methods=["GET"])
@login_required
def faces_page():
    known = database.get_known_faces()
    return render_template("faces.html", known_faces=known)


@app.route("/faces/enroll", methods=["POST"])
@login_required
def enroll_face():
    name = request.form.get("name", "").strip()
    samples_needed = 15
    if not name:
        flash("Please provide a name.", "danger")
        return redirect(url_for("faces_page"))

    captured = []
    attempts = 0
    while len(captured) < samples_needed and attempts < samples_needed * 10:
        frame = camera.read()
        attempts += 1
        if frame is None:
            time.sleep(0.1)
            continue
        faces = face_detector.detect(frame)
        if faces:
            x, y, w, h = faces[0]
            captured.append(frame[y:y + h, x:x + w])
        time.sleep(0.15)

    if not captured:
        flash("Could not capture any face samples - make sure a face is visible to the camera.", "danger")
        return redirect(url_for("faces_page"))

    try:
        face_recognizer.enroll(name, captured)
        flash(f"Enrolled '{name}' with {len(captured)} face samples.", "success")
    except Exception as e:
        flash(f"Enrollment failed: {e}", "danger")

    return redirect(url_for("faces_page"))


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
