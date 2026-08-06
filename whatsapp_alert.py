"""
whatsapp_alert.py
-------------------
Sends WhatsApp alerts using Twilio's WhatsApp API (optional feature).

Setup:
  1. Create a Twilio account -> https://www.twilio.com/whatsapp
  2. Join the Twilio WhatsApp Sandbox (or set up a production sender).
  3. In Settings, configure:
       twilio_account_sid
       twilio_auth_token
       twilio_from_number   e.g. "whatsapp:+14155238886" (sandbox number)
       twilio_to_number     e.g. "whatsapp:+91XXXXXXXXXX"

Images must be reachable via a public URL for Twilio to attach them
(Twilio's servers fetch the media themselves) - if your surveillance
server isn't publicly reachable, WhatsApp alerts will still send the
text message, just without the image attached.
"""

import database

try:
    from twilio.rest import Client
    _TWILIO_AVAILABLE = True
except Exception:
    _TWILIO_AVAILABLE = False


def send_whatsapp_alert(message, image_url=None):
    settings = database.get_all_settings()

    if settings.get("whatsapp_enabled") != "true":
        return False, "WhatsApp alerts are disabled in Settings."

    if not _TWILIO_AVAILABLE:
        return False, "The 'twilio' package is not installed."

    sid = settings.get("twilio_account_sid")
    token = settings.get("twilio_auth_token")
    from_number = settings.get("twilio_from_number")
    to_number = settings.get("twilio_to_number")

    if not all([sid, token, from_number, to_number]):
        return False, "Twilio settings are incomplete."

    try:
        client = Client(sid, token)
        kwargs = {"from_": from_number, "to": to_number, "body": message}
        if image_url:
            kwargs["media_url"] = [image_url]
        msg = client.messages.create(**kwargs)
        return True, f"WhatsApp message sent (sid={msg.sid})."
    except Exception as e:
        return False, f"WhatsApp send failed: {e}"
