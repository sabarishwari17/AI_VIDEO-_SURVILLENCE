"""
alert.py
--------
Central alert orchestrator. Called by app.py whenever a
notification-worthy event happens (intruder / fire / smoke / weapon /
critical object). Responsible for:

  - Applying a cooldown per event-type+camera so we don't spam every
    single frame with duplicate emails/Telegram messages.
  - Fanning the alert out to whichever channels are enabled in
    Settings (email / Telegram / WhatsApp).
  - Persisting the alert in the database (see database.log_alert).
"""

import time
from datetime import datetime

import config
import database
import email_alert
import telegram_alert
import whatsapp_alert

_last_alert_time = {}   # key: (camera, event_type) -> timestamp


def _cooldown_ok(camera, event_type):
    key = (camera, event_type)
    now = time.time()
    last = _last_alert_time.get(key, 0)
    if now - last >= config.ALERT_COOLDOWN_SECONDS:
        _last_alert_time[key] = now
        return True
    return False


def send_alert(camera, event_type, object_name, confidence, image_path=None,
                video_path=None, severity="high", force=False, event_id=None):
    """
    event_type examples: "intruder", "fire", "smoke", "weapon", "object"
    Returns dict summary of what was sent (for logging/debugging).
    """
    if not force and not _cooldown_ok(camera, event_type):
        return {"sent": False, "reason": "cooldown"}

    now = datetime.now()
    title_map = {
        "intruder": "Intruder Detected",
        "fire": "Fire Detected",
        "smoke": "Smoke Detected",
        "weapon": "Weapon Detected",
        "object": f"{object_name.title()} Detected",
        "violence": "Possible Violence Detected",
    }
    title = title_map.get(event_type, "Security Alert")

    body_lines = [
        f"Alert: {title}",
        f"Date: {now.strftime('%Y-%m-%d')}",
        f"Time: {now.strftime('%H:%M:%S')}",
        f"Camera: {camera}",
        f"Object detected: {object_name}",
        f"Confidence: {confidence if confidence is not None else 'N/A'}",
    ]
    message_text = "\n".join(body_lines)

    channels_sent = []
    results = {}

    # Email
    ok, info = email_alert.send_email_alert(title, body_lines, image_path, video_path)
    results["email"] = info
    if ok:
        channels_sent.append("email")

    # Telegram
    ok, info = telegram_alert.send_telegram_message(message_text)
    results["telegram_text"] = info
    if ok:
        channels_sent.append("telegram")
    if image_path:
        telegram_alert.send_telegram_photo(image_path, caption=title)
    if video_path:
        telegram_alert.send_telegram_video(video_path, caption=title)

    # WhatsApp
    ok, info = whatsapp_alert.send_whatsapp_alert(message_text)
    results["whatsapp"] = info
    if ok:
        channels_sent.append("whatsapp")

    database.log_alert(
        title=title,
        message=message_text,
        severity=severity,
        channels=channels_sent,
        event_id=event_id,
    )

    return {"sent": True, "channels": channels_sent, "details": results}
