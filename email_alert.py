"""
email_alert.py
---------------
Sends email alerts via SMTP (defaults configured for Gmail SMTP).

Gmail setup:
  1. Enable 2-Step Verification on the Google account.
  2. Create an "App Password" (Google Account -> Security -> App
     Passwords) - use THIS as smtp_password, not your normal Gmail
     password.
  3. In the Settings page, set:
       smtp_server = smtp.gmail.com
       smtp_port   = 587
       smtp_username = your.email@gmail.com
       smtp_password = <the 16-character app password>
       alert_email_to = where alerts should be sent
"""

import smtplib
import ssl
from email.message import EmailMessage
import mimetypes

import database


def send_email_alert(subject, body_lines, image_path=None, video_path=None):
    """
    body_lines: list[str] rendered as separate lines in the email body.
    Returns (success: bool, message: str)
    """
    settings = database.get_all_settings()

    if settings.get("email_enabled", "false") != "true":
        return False, "Email alerts are disabled in Settings."

    smtp_server = settings.get("smtp_server")
    smtp_port = int(settings.get("smtp_port", "587"))
    smtp_username = settings.get("smtp_username")
    smtp_password = settings.get("smtp_password")
    to_addr = settings.get("alert_email_to")

    if not all([smtp_server, smtp_username, smtp_password, to_addr]):
        return False, "Email settings are incomplete."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_username
    msg["To"] = to_addr
    msg.set_content("\n".join(body_lines))

    for path in (image_path, video_path):
        if path:
            try:
                mime_type, _ = mimetypes.guess_type(path)
                maintype, subtype = (mime_type or "application/octet-stream").split("/")
                with open(path, "rb") as f:
                    msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                        filename=path.split("/")[-1])
            except FileNotFoundError:
                pass

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        return True, "Email sent."
    except Exception as e:
        return False, f"Email send failed: {e}"
