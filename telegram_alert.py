"""
telegram_alert.py
-------------------
Sends alerts (text / photo / video / location) via the Telegram Bot API.

Setup:
  1. Talk to @BotFather on Telegram -> /newbot -> get a bot token.
  2. Message your new bot once (or add it to a group).
  3. Get your chat_id: visit
       https://api.telegram.org/bot<TOKEN>/getUpdates
     after messaging the bot, and read "chat":{"id": ...}
  4. In the Settings page set telegram_bot_token and telegram_chat_id.
"""

import requests
import database

API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _settings():
    s = database.get_all_settings()
    return s.get("telegram_enabled") == "true", s.get("telegram_bot_token"), s.get("telegram_chat_id")


def send_telegram_message(text):
    enabled, token, chat_id = _settings()
    if not enabled or not token or not chat_id:
        return False, "Telegram alerts are disabled or not configured."
    try:
        url = API_BASE.format(token=token, method="sendMessage")
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        return resp.ok, resp.text
    except Exception as e:
        return False, f"Telegram send failed: {e}"


def send_telegram_photo(photo_path, caption=""):
    enabled, token, chat_id = _settings()
    if not enabled or not token or not chat_id:
        return False, "Telegram alerts are disabled or not configured."
    try:
        url = API_BASE.format(token=token, method="sendPhoto")
        with open(photo_path, "rb") as f:
            resp = requests.post(
                url, data={"chat_id": chat_id, "caption": caption},
                files={"photo": f}, timeout=20
            )
        return resp.ok, resp.text
    except Exception as e:
        return False, f"Telegram photo send failed: {e}"


def send_telegram_video(video_path, caption=""):
    enabled, token, chat_id = _settings()
    if not enabled or not token or not chat_id:
        return False, "Telegram alerts are disabled or not configured."
    try:
        url = API_BASE.format(token=token, method="sendVideo")
        with open(video_path, "rb") as f:
            resp = requests.post(
                url, data={"chat_id": chat_id, "caption": caption},
                files={"video": f}, timeout=60
            )
        return resp.ok, resp.text
    except Exception as e:
        return False, f"Telegram video send failed: {e}"


def send_telegram_location(latitude, longitude):
    enabled, token, chat_id = _settings()
    if not enabled or not token or not chat_id:
        return False, "Telegram alerts are disabled or not configured."
    try:
        url = API_BASE.format(token=token, method="sendLocation")
        resp = requests.post(
            url, data={"chat_id": chat_id, "latitude": latitude, "longitude": longitude},
            timeout=10
        )
        return resp.ok, resp.text
    except Exception as e:
        return False, f"Telegram location send failed: {e}"
