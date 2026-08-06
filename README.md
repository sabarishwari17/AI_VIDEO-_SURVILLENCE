# AI Smart Video Surveillance System

A Flask + OpenCV + YOLOv8 powered smart surveillance dashboard: live camera
streaming, motion/object/face detection, face recognition with intruder
alerts, automatic recording, and multi-channel notifications (Email,
Telegram, WhatsApp).

---

## 1. Features

| Category | What it does |
|---|---|
| Live streaming | Webcam / USB / IP camera / RTSP, shown live in the browser |
| Motion detection | OpenCV background subtraction, first-stage trigger |
| Object detection | YOLOv8 (person, car, bike, bus, truck, dog, cat, bag, phone, ...) |
| Face detection | Haar Cascade, draws boxes around every face |
| Face recognition | OpenCV LBPH recognizer; enroll authorized people from the dashboard |
| Intruder detection | Unknown face -> record + screenshot + email + Telegram alert |
| Fire / smoke detection | HSV colour-heuristic (works without extra models) |
| Weapon detection | Knife/gun - **requires a custom YOLO model** (see §6) |
| Auto recording | Starts on motion/object/face/fire/smoke/weapon trigger, saved as `.mp4` |
| Screenshots | Saved to `snapshots/` on every alert-worthy event |
| Alerts | Email (SMTP/Gmail), Telegram Bot API, WhatsApp (Twilio) |
| Dashboard | Live feed, today's detections, alerts, recordings, storage, FPS, CPU/RAM |
| Logs | `logs/events.csv` + SQLite `events` table |
| Database | SQLite: users, alerts, events, recordings, settings, known_faces |
| Settings page | Edit email/Telegram/WhatsApp/camera/recording/detection config live |
| Auth | Login/logout, session-based, admin only |
| Search & playback | Search recordings by date/object/camera, watch/download/delete |
| Statistics | Chart.js graphs: object frequency + detections over time |
| Dark/Light mode | Toggle in the sidebar, remembered via cookie |
| Responsive | Bootstrap 5 layout - desktop, tablet, mobile |

---

## 2. Project Structure

```
AI-Surveillance/
├── app.py                     # Main Flask app + background processing loop
├── camera.py                  # Webcam/USB/IP/RTSP camera abstraction
├── motion_detection.py        # OpenCV motion detector
├── object_detection.py        # YOLOv8 wrapper (Ultralytics)
├── face_detection.py          # Haar Cascade face detector
├── face_recognition_module.py # LBPH face recognizer (enroll + recognize)
├── fire_detection.py          # Fire (HSV) + smoke (heuristic) detection
├── weapon_detection.py        # Knife/gun detection (custom model required)
├── alert.py                   # Multi-channel alert orchestrator + cooldown
├── email_alert.py             # SMTP email sending
├── telegram_alert.py          # Telegram Bot API sending
├── whatsapp_alert.py          # Twilio WhatsApp sending
├── database.py                # SQLite schema + all DB helper functions
├── config.py                  # Static configuration & defaults
├── requirements.txt
├── templates/
│   ├── base.html               # Shared layout (sidebar, topbar, theme)
│   ├── login.html
│   ├── dashboard.html
│   ├── settings.html
│   ├── recordings.html
│   ├── alerts.html
│   ├── faces.html               # Enroll / manage authorized faces
│   └── 404.html
├── static/
│   ├── css/style.css
│   └── js/{dashboard.js, theme.js}
├── uploads/                   # Scratch space for uploads
├── recordings/                # Auto-saved .mp4 recordings
├── snapshots/                 # Auto-saved .jpg screenshots
├── logs/events.csv            # Plain-text event log
├── database/surveillance.db   # SQLite database (created on first run)
├── models/                    # yolov8n.pt (auto-downloaded), optional custom models
├── known_faces/                # Enrolled face samples, per person
└── README.md
```

### File-by-file explanation

- **app.py** - the Flask app. Boots the camera + detectors, runs a background
  thread (`processing_loop`) that continuously reads frames, runs motion →
  object → weapon → fire/smoke → face detection, triggers recording/alerts,
  and stores the latest annotated frame for MJPEG streaming. Also defines
  every HTTP route (dashboard, login, settings, recordings, alerts, faces,
  and small JSON APIs for AJAX polling).
- **camera.py** - a threaded camera reader that supports a webcam index,
  USB camera index, IP camera MJPEG URL, or RTSP URL, with auto-reconnect.
- **motion_detection.py** - `MotionDetector` using `cv2.createBackgroundSubtractorMOG2`;
  returns bounding boxes of moving regions.
- **object_detection.py** - `ObjectDetector` wraps Ultralytics YOLOv8 for
  general object detection and draws labeled boxes (critical classes in red).
- **face_detection.py** - `FaceDetector` using OpenCV's Haar Cascade (no
  extra downloads needed).
- **face_recognition_module.py** - `FaceRecognizer` using OpenCV's LBPH
  algorithm. Handles enrolling new authorized people and recognizing faces
  at runtime; unmatched faces are labeled "Unknown".
- **fire_detection.py** - HSV colour-threshold based fire detector, plus a
  motion+greyscale heuristic for smoke. Dependency-free; swap in a custom
  YOLO fire/smoke model for higher accuracy (see §6).
- **weapon_detection.py** - looks for `models/weapon_yolov8.pt`; if absent,
  weapon detection safely no-ops (see §6 to enable it for real).
- **alert.py** - the single place that decides *when* to actually fire an
  alert (per-camera/per-event-type cooldown) and fans it out to whichever
  of email/Telegram/WhatsApp are enabled, logging the result to the DB.
- **email_alert.py / telegram_alert.py / whatsapp_alert.py** - thin,
  independent wrappers around SMTP, the Telegram Bot API, and Twilio's
  WhatsApp API respectively. Each reads its config live from the `settings`
  DB table so changes in the Settings page take effect immediately.
- **database.py** - creates all SQLite tables on first run, seeds a default
  admin user + default settings, and exposes every query the app needs
  (events, alerts, recordings, settings, known faces, storage usage, stats).
- **config.py** - static, non-DB configuration: folder paths, Flask secret
  key, default admin credentials, camera/recording/detection defaults.

---

## 3. Installation

```bash
# 1. Clone / unzip the project, then enter the folder
cd AI-Surveillance

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **First run** will auto-download the YOLOv8n weights (`yolov8n.pt`, ~6MB)
> the first time object detection runs - make sure you have an internet
> connection for that one-time download.

---

## 4. Running the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

**Default login:** `admin` / `admin123` — change this immediately from the
Settings page (or set `SURVEILLANCE_ADMIN_USER` / `SURVEILLANCE_ADMIN_PASS`
environment variables before the very first run, before `database/surveillance.db`
is created).

---

## 5. How To Add New Faces (Authorized Users)

1. Log in and open the **Live Camera Feed** on the Dashboard so the person
   is visible to the camera.
2. Go to **Authorized Faces** in the sidebar.
3. Enter their name and click **Capture & Enroll** - the system grabs ~15
   face samples over a few seconds and (re)trains the LBPH recognizer.
4. From then on, that person's face will show their name in the live feed.
   Anyone else is labeled **Unknown** and triggers an intruder alert.

---

## 6. Custom Models (Fire / Smoke / Knife / Gun / Helmet)

The base YOLOv8n COCO model (auto-downloaded) only knows the 80 standard
COCO classes (person, car, dog, backpack, cell phone, etc.) — it does
**not** include fire, smoke, knife, gun, or helmet.

- **Fire/smoke** already works out of the box via a colour/motion
  heuristic in `fire_detection.py` (no model needed, but less accurate
  than a trained model in tricky lighting).
- **Knife/gun**: to get real weapon detection, train (or download) a
  YOLOv8 model on a weapon dataset (e.g. from Roboflow Universe):
  ```bash
  yolo train data=weapons.yaml model=yolov8n.pt epochs=100
  ```
  Copy the resulting `best.pt` to `models/weapon_yolov8.pt` and restart
  the app — `weapon_detection.py` will automatically start using it.
- **Helmet**: train/download a helmet-detection YOLO model the same way
  and load it similarly to the weapon model (duplicate `weapon_detection.py`
  as a starting point).

---

## 7. Configuring Email Alerts (Gmail)

1. Enable **2-Step Verification** on the Gmail account.
2. Create an **App Password**: Google Account → Security → App Passwords.
3. In the app's **Settings** page:
   - SMTP Server: `smtp.gmail.com`
   - SMTP Port: `587`
   - SMTP Username: your Gmail address
   - SMTP App Password: the 16-character app password (NOT your normal password)
   - Send Alerts To: the recipient email address
4. Toggle **Enable Email Alerts** and save.

---

## 8. Configuring Telegram Alerts

1. Message **@BotFather** on Telegram → `/newbot` → follow the prompts to
   get a bot token.
2. Send any message to your new bot (or add it to a group).
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   and copy the `"chat":{"id": ...}` value.
4. In **Settings**, paste the Bot Token and Chat ID, enable Telegram
   Alerts, and save.

---

## 9. Configuring Twilio WhatsApp Alerts (optional)

1. Create a free account at https://www.twilio.com/whatsapp and join the
   WhatsApp Sandbox (or configure a production WhatsApp sender).
2. In **Settings**, fill in:
   - Twilio Account SID / Auth Token (from the Twilio console)
   - From Number: e.g. `whatsapp:+14155238886` (sandbox number)
   - To Number: e.g. `whatsapp:+91XXXXXXXXXX` (must have joined the sandbox)
3. Enable WhatsApp Alerts and save.

> Image attachments require your server to be publicly reachable (Twilio
> fetches the media URL itself); text alerts work regardless.

---

## 10. Notes, Limitations & Honest Scope

This is a genuinely working, runnable system - not a mockup - but a few
things are worth knowing so you can extend it further:

- **Weapon detection** is disabled until you plug in a custom-trained
  model (§6) - the stock YOLO model has no knife/gun classes.
- **Fire/smoke detection** uses a colour/heuristic approach, which is
  dependency-free but less robust than a trained model, especially with
  unusual lighting or camera colour balance.
- **Face recognition** uses OpenCV's LBPH algorithm rather than a deep
  embedding model (like the `face_recognition`/dlib library), chosen
  specifically because it installs cleanly everywhere without a C++
  build toolchain. It works well for a handful of enrolled people in a
  home/office setting; for larger deployments, swap in a deep
  face-embedding model.
- **Violence/fighting detection** is listed as a bonus item in the spec.
  Reliable action recognition needs a temporal (video-clip) deep learning
  model (e.g. a 3D-CNN or transformer trained on a fight-detection
  dataset) which is a separate, heavier training/deployment project - it
  is not included here, but `alert.py`'s `event_type="violence"` path is
  already wired up so you can drop a classifier's output straight in.
- **Bonus items not implemented** (crowd counting, loitering, line
  crossing, license-plate recognition, attendance, cloud backup, Docker,
  JWT REST API, multi-camera, Raspberry Pi/GPU tuning): the codebase is
  structured so each of these is an additive module - e.g. a new
  `*_detection.py` file plus a call from `app.py`'s `processing_loop()` -
  rather than a rewrite. A few pointers:
  - **Multi-camera**: instantiate multiple `Camera` objects and run one
    `processing_loop`-style thread per camera.
  - **REST API / JWT**: add `Flask-JWT-Extended` and mirror the existing
    routes as `/api/v1/...` JSON endpoints.
  - **Docker**: a straightforward `Dockerfile` based on `python:3.12-slim`
    + `apt-get install -y libgl1 libglib2.0-0` (OpenCV's system deps) +
    `pip install -r requirements.txt` will run this app as-is.
  - **Cloud backup**: sync `recordings/` and `snapshots/` with `rclone`,
    `boto3` (S3), or the Google Drive API on a cron schedule.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| Camera feed is black/offline | Check `camera_source` in Settings; try `0`, `1`, or a full RTSP/HTTP URL |
| `cv2.face` not found | `pip install opencv-contrib-python` (not just `opencv-python`) |
| YOLO very slow | Use `yolov8n.pt` (already default, smallest/fastest) or enable GPU/CUDA |
| Emails not sending | Use a Gmail **App Password**, not your normal password |
| No recordings appearing | Check `detection_enabled` is ON and something is actually triggering (motion/object/face) |

---

Built with Python, Flask, OpenCV, and YOLOv8. Enjoy building on top of it!
