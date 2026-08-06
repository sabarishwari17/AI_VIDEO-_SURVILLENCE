"""
database.py
-----------
SQLite database layer for the AI Smart Video Surveillance System.

Tables
------
users        : login accounts (admin panel)
alerts       : high-priority notifications that were sent out (email/telegram/whatsapp)
events       : every detection event (motion / object / face / fire / weapon ...)
recordings   : metadata about saved .mp4 recordings
settings     : key/value store for all runtime-editable configuration

All functions in this module open a short-lived connection per call
(SQLite + Flask's multi-threaded dev server works best this way) and
use `check_same_thread=False` where a connection is cached for the
streaming thread.
"""

import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash

import config


def get_connection():
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create all tables (idempotent) and seed default settings/admin user."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL,
            event_type TEXT NOT NULL,     -- motion / object / face / fire / smoke / weapon / intruder
            video_path TEXT,
            image_path TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'medium',   -- low / medium / high
            channels TEXT,                    -- comma list: email,telegram,whatsapp
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            camera TEXT NOT NULL,
            trigger_reason TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds REAL,
            size_bytes INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            label_id INTEGER UNIQUE NOT NULL,
            sample_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()

    # Seed default admin user if no users exist yet
    cur.execute("SELECT COUNT(*) as c FROM users")
    if cur.fetchone()["c"] == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (
                config.DEFAULT_ADMIN_USERNAME,
                generate_password_hash(config.DEFAULT_ADMIN_PASSWORD),
                "admin",
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    # Seed default settings if not present
    defaults = {
        "email_enabled": "false",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": "587",
        "smtp_username": "",
        "smtp_password": "",
        "alert_email_to": "",
        "telegram_enabled": "false",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "whatsapp_enabled": "false",
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "twilio_to_number": "",
        "camera_source": config.DEFAULT_CAMERA_SOURCE,
        "recording_quality": config.DEFAULT_RECORDING_QUALITY,
        "yolo_confidence": str(config.DEFAULT_YOLO_CONFIDENCE),
        "motion_threshold": str(config.DEFAULT_MOTION_THRESHOLD),
        "detection_enabled": "true",
        "face_recognition_enabled": "true",
        "dark_mode": "true",
    }
    for k, v in defaults.items():
        cur.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------
def get_setting(key, default=None):
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def get_all_settings():
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def set_setting(key, value):
    conn = get_connection()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def update_settings(settings_dict):
    conn = get_connection()
    for k, v in settings_dict.items():
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (k, str(v)),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------
def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def create_user(username, password, role="admin"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, generate_password_hash(password), role, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def update_password(user_id, new_password):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(new_password), user_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------
def log_event(camera, object_name, confidence, event_type, video_path=None, image_path=None):
    now = datetime.now()
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO events (date, time, camera, object, confidence, event_type,
                                video_path, image_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            camera,
            object_name,
            confidence,
            event_type,
            video_path,
            image_path,
            now.isoformat(),
        ),
    )
    conn.commit()
    event_id = cur.lastrowid
    conn.close()

    # Also append to the CSV log (logs/events.csv) as required by spec
    _append_csv_log(now, camera, object_name, confidence, event_type, video_path, image_path)
    return event_id


def _append_csv_log(now, camera, object_name, confidence, event_type, video_path, image_path):
    import csv
    file_exists = os.path.isfile(config.EVENTS_CSV_PATH)
    with open(config.EVENTS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Camera", "Object", "Confidence",
                              "Event", "Video Path", "Image Path"])
        writer.writerow([
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), camera,
            object_name, confidence, event_type, video_path or "", image_path or "",
        ])


def get_recent_events(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_events_today():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events WHERE date = ? ORDER BY id DESC", (today,)
    ).fetchall()
    conn.close()
    return rows


def search_events(date=None, object_name=None, camera=None):
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if date:
        query += " AND date = ?"
        params.append(date)
    if object_name:
        query += " AND object LIKE ?"
        params.append(f"%{object_name}%")
    if camera:
        query += " AND camera LIKE ?"
        params.append(f"%{camera}%")
    query += " ORDER BY id DESC"
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def event_stats(period_days=7):
    """Return object-frequency + per-day counts for charts."""
    conn = get_connection()
    by_object = conn.execute(
        "SELECT object, COUNT(*) as cnt FROM events GROUP BY object ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    by_day = conn.execute(
        f"""SELECT date, COUNT(*) as cnt FROM events
            WHERE date >= date('now', '-{period_days} days')
            GROUP BY date ORDER BY date ASC"""
    ).fetchall()
    conn.close()
    return {
        "by_object": [{"object": r["object"], "count": r["cnt"]} for r in by_object],
        "by_day": [{"date": r["date"], "count": r["cnt"]} for r in by_day],
    }


# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------
def log_alert(title, message, severity="medium", channels=None, event_id=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO alerts (event_id, title, message, severity, channels, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (event_id, title, message, severity, ",".join(channels or []), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_alerts(limit=50):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------
def log_recording_start(filename, path, camera, trigger_reason):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO recordings (filename, path, camera, trigger_reason, start_time, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (filename, path, camera, trigger_reason, datetime.now().isoformat(), datetime.now().isoformat()),
    )
    conn.commit()
    rec_id = cur.lastrowid
    conn.close()
    return rec_id


def log_recording_end(rec_id, duration_seconds, size_bytes):
    conn = get_connection()
    conn.execute(
        """UPDATE recordings SET end_time = ?, duration_seconds = ?, size_bytes = ?
           WHERE id = ?""",
        (datetime.now().isoformat(), duration_seconds, size_bytes, rec_id),
    )
    conn.commit()
    conn.close()


def get_recordings(limit=100):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM recordings ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


def delete_recording(rec_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    if row and os.path.isfile(row["path"]):
        try:
            os.remove(row["path"])
        except OSError:
            pass
    conn.execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
    conn.commit()
    conn.close()


def storage_usage_bytes():
    total = 0
    for folder in (config.RECORDINGS_FOLDER, config.SNAPSHOTS_FOLDER):
        for root, _, files in os.walk(folder):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


# ---------------------------------------------------------------------
# Known faces (face recognition labels)
# ---------------------------------------------------------------------
def add_known_face(name, label_id, sample_count):
    conn = get_connection()
    conn.execute(
        "INSERT INTO known_faces (name, label_id, sample_count, created_at) VALUES (?, ?, ?, ?)",
        (name, label_id, sample_count, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_known_faces():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM known_faces ORDER BY name ASC").fetchall()
    conn.close()
    return rows


def next_label_id():
    conn = get_connection()
    row = conn.execute("SELECT MAX(label_id) as m FROM known_faces").fetchone()
    conn.close()
    return (row["m"] + 1) if row["m"] is not None else 1
