# SDV Driver Monitoring System — CLAUDE.md

## Project Purpose

A Software-Defined Vehicle (SDV) prototype that simulates an onboard driver management system.
Target platform: **Raspberry Pi 5, 64-bit Linux, headless, Eclipse S-CORE middleware**.

When a driver gets into the car, the system:

1. **Authenticates** the driver via live face recognition (SFace embeddings, cosine similarity)
2. **Loads** their personalised vehicle settings (seat position, climate temp, radio)
3. **Publishes** the loaded settings as VSS signals to the Eclipse S-CORE Kuksa Data Broker
4. **Monitors** for drowsiness continuously using Eye Aspect Ratio (EAR) via MediaPipe Face Mesh
5. **Publishes** a drowsiness level signal (0–10) each monitor frame

---

## State Machine

```
AUTH ──(face matched, dist < threshold)──► MONITOR
  │                                             │
  └──(timeout, no match)── re-enters AUTH ◄─────┘
                                           (future: on-seat sensor or command signal)
```

No keyboard or display interaction — fully headless. Enrollment is done separately
via the Flask web server (`enrollment_server.py`) before the monitor starts.

---

## File Structure

```
FaceDetection/
├── sdv_monitor.py          # Main pipeline — AUTH/MONITOR state machine
├── enrollment_server.py    # Flask web UI for headless face enrollment
├── inference.py            # YuNet (detect) + SFace (embed) + MediaPipe (EAR)
├── signals.py              # Eclipse S-CORE / Kuksa Data Broker publisher
├── sdv_db.py               # SQLite driver profiles + embedding store
├── config.json             # All tunable parameters
├── requirements_pi.txt     # Pi dependency list (~350 MB installed)
├── setup_pi.sh             # One-shot Pi setup: venv + packages + ONNX models
├── sdv.service             # systemd unit — auto-start monitor on boot
└── models/                 # Downloaded by setup_pi.sh
    ├── face_detection_yunet_2023mar.onnx    (~400 KB)
    └── face_recognition_sface_2021dec.onnx  (~37 MB)

Database: ~/.sdv/sdv_users.db
```

---

## Running on Pi

```bash
# One-time setup (installs packages + downloads models)
bash setup_pi.sh

# Enrol a driver (do this before starting the monitor)
# Open http://<pi-ip>:5000 on any phone/laptop on the same network
source venv/bin/activate && python enrollment_server.py

# Run the monitor (Ctrl+C to stop)
source venv/bin/activate && python sdv_monitor.py

# Install as auto-start systemd service
sudo cp sdv.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sdv && sudo systemctl start sdv

# View live logs
journalctl -u sdv -f
```

---

## config.json Parameters

```json
{
  "face_recognition": {
    "match_threshold": 0.40,          // cosine dist — lower = stricter
    "auth_timeout_seconds": 10,       // retry AUTH after this many seconds with no match
    "auth_check_every_n_frames": 5    // skip frames to save CPU in AUTH mode
  },
  "models": {
    "det_model": "models/face_detection_yunet_2023mar.onnx",
    "rec_model": "models/face_recognition_sface_2021dec.onnx"
  },
  "drowsiness": {
    "ear_threshold": 0.25,            // EAR below this = eye considered closed
    "consecutive_frames_alert": 20    // closed-eye frames before alert fires
  },
  "database": {
    "path": "~/.sdv/sdv_users.db"
  },
  "enrollment": {
    "num_sample_frames": 10,          // face samples captured per enrolment
    "capture_interval_frames": 8
  },
  "camera": {
    "source": 0                       // 0 = /dev/video0 (USB or Pi CSI via v4l2)
  },
  "score": {
    "host": "127.0.0.1",             // Kuksa Data Broker address
    "port": 55555
  }
}
```

---

## Eclipse S-CORE Integration

The system uses the **Kuksa Data Broker** (part of Eclipse S-CORE) as the vehicle signal bus.
Python client: `kuksa-client` (gRPC). All signals follow VSS paths.

**Signals published:**

| VSS Path | Type | Description |
|---|---|---|
| `Vehicle.Driver.IsAuthenticated` | bool | True once driver is matched |
| `Vehicle.Driver.Identifier` | string | Driver name ("" while scanning) |
| `Vehicle.Driver.DrowsinessLevel` | uint8 | 0 (alert) → 10 (very drowsy) |
| `Vehicle.Driver.AttentionLvl` | string | "ATTENTIVE" or "ALERT" |
| `Vehicle.Cabin.Seat.Row1.DriverSide.Position` | uint8 | Driver's saved seat position |
| `Vehicle.Cabin.HVAC.AmbientAirTemperature` | float | Driver's saved climate temp |

If `kuksa-client` is not installed or the broker is unreachable, `signals.py` logs
signals to stdout and continues — the monitor never crashes due to a missing broker.

---

## ML Stack (Pi-native, no TensorFlow)

| Task | Library | Model | Latency Pi 5 |
|---|---|---|---|
| Face detection | OpenCV `cv2.FaceDetectorYN` | YuNet ONNX | ~20ms |
| Face embedding | OpenCV `cv2.FaceRecognizerSF` | SFace ONNX | ~40ms |
| Landmark EAR | MediaPipe Face Mesh | built-in | ~20ms |
| Total (auth frame) | — | — | ~80ms |
| Total (monitor frame) | — | — | ~25ms |

Both YuNet and SFace are bundled with OpenCV 4.8+ — no TFLite or tflite-runtime needed.

---

## Enrollment Web UI

`enrollment_server.py` runs a Flask server at `http://<pi-ip>:5000`.

- Open on phone/laptop browser — no display or keyboard on the Pi needed
- Enter driver name → click Enroll
- Live progress bar as samples are captured
- Driver list with delete buttons
- The enrollment server and the monitor **cannot share the camera simultaneously** —
  enrol drivers first, then start the monitor

---

## Development Status

- [x] Pi-native ML stack (YuNet + SFace, no TensorFlow)
- [x] Headless monitor (`sdv_monitor.py`, no `cv2.imshow`)
- [x] Headless enrollment web UI (`enrollment_server.py`)
- [x] Eclipse S-CORE / Kuksa signal publishing (`signals.py`)
- [x] Clean SQLite store (`sdv_db.py`, standard locking)
- [x] systemd service unit (`sdv.service`)
- [x] One-shot Pi setup script (`setup_pi.sh`)
- [ ] Pi Camera Module (CSI) support — currently uses `/dev/video0` via v4l2
- [ ] GPIO alert (buzzer/LED on drowsiness)
- [ ] Subscribe to S-CORE command signals (e.g., re-auth on door open)
- [ ] Android companion app (separate Kotlin project, ML Kit)
