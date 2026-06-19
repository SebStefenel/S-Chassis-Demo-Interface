# SDV Driver Monitoring System — CLAUDE.md

## Project Purpose

A Software-Defined Vehicle (SDV) prototype that simulates an onboard driver management system. When a driver gets into the car, the system:

1. **Authenticates** the driver via live facial recognition (Facenet embeddings, cosine similarity)
2. **Loads** their personalised vehicle settings (seat position, climate, radio station)
3. **Monitors** for drowsiness continuously using Eye Aspect Ratio (EAR) via dlib's 68-point facial landmark detector
4. **Alerts** the driver (visual overlay + stdout) if drowsiness thresholds are breached

The core engineering challenge: two separate open-source repos (deepface for face ID, dlib-based for drowsiness) both wanted exclusive camera access. The solution is a single-camera state machine that shares one `cv2.VideoCapture` instance.

---

## State Machine

```
AUTH ──(face matched)──► MONITOR
  │                         │
  └──(timeout)──► UNRECOGNIZED   [A] re-auth
                     │
              [E] ENROLLING ──(complete)──► MONITOR
              [G] ──────────────────────► MONITOR (guest)
              [R] ──────────────────────► AUTH
```

Keys: `Q` quit | `A` re-authenticate | `E` enroll | `G` guest | `R` retry

---

## File Structure

```
FaceDetection/
├── sdv_monitor.py          # Main entry point — state machine pipeline
├── enroll_driver.py        # CLI tool to register new drivers
├── sdv_db.py               # SQLite user/embedding store
├── config.json             # All tunable parameters
├── run_monitor.sh          # Launch script (activates venv)
├── run_enroll.sh           # Enroll script (activates venv)
├── requirements_sdv.txt    # Unified dependency list
├── deepface/               # Local deepface repo (face recognition engine)
├── Drowsiness_Detection/   # Source repo — dlib EAR drowsiness code
│   └── models/
│       └── shape_predictor_68_face_landmarks.dat   # dlib landmark model
└── venv/                   # Python 3.11 virtual environment
```

Database stored at: `~/Library/Application Support/SDV/sdv_users.db`

---

## Running (macOS)

```bash
cd ~/Documents/Infosys/FaceDetection

# Enroll a driver
./run_enroll.sh -u YourName

# Run the monitor
./run_monitor.sh

# List enrolled drivers
./run_enroll.sh --list

# Delete a profile
./run_enroll.sh --delete YourName
```

**macOS requirement:** Terminal must have Full Disk Access enabled in System Settings → Privacy & Security → Full Disk Access. F-Secure (if installed) must also have the Python binary or project directory whitelisted.

---

## config.json Parameters

```json
{
  "face_recognition": {
    "model_name": "Facenet",            // recognition model
    "detector_backend": "retinaface",   // face detector (see Pi notes)
    "distance_metric": "cosine",
    "match_threshold": 0.40,            // lower = stricter matching
    "auth_timeout_seconds": 10,
    "auth_check_every_n_frames": 5
  },
  "drowsiness": {
    "ear_threshold": 0.25,              // EAR below this = eye closed
    "consecutive_frames_alert": 20,     // frames closed before alert fires
    "landmark_model_path": "Drowsiness_Detection/models/shape_predictor_68_face_landmarks.dat"
  },
  "enrollment": {
    "num_sample_frames": 10,            // face samples captured per enroll
    "capture_interval_frames": 8
  },
  "camera": {
    "source": 0,                        // 0 = default webcam
    "display_width": 640
  }
}
```

---

## Key Implementation Notes

- **SQLite quirk (macOS):** The db uses `isolation_level=None` (autocommit) and `PRAGMA journal_mode=MEMORY` to avoid `fcntl` file-locking failures caused by macOS security restrictions on Homebrew Python.
- **No `silent` param:** The local deepface version does not support the `silent` kwarg in `DeepFace.represent()` — do not add it back.
- **Deepface path:** Both `sdv_monitor.py` and `enroll_driver.py` do `sys.path.insert(0, "deepface")` — deepface is NOT pip-installed, it's used from the local folder.
- **venv installed via uv + staging:** macOS sandbox blocks Python's `os.rename()` in site-packages, so packages were installed to `/tmp/sdv_staging` then `cp -r`'d into the venv.

---

## Raspberry Pi — Required Changes

### Target: Raspberry Pi 4/5, 64-bit Linux (Raspberry Pi OS or Ubuntu)

The current Mac stack will NOT run on Pi without significant changes. Here is the full migration plan:

### 1. Replace TensorFlow with TensorFlow Lite

Full TF 2.x is 2.6 GB installed and far too slow on Pi's ARM CPU.

```bash
# Instead of tensorflow:
pip install tflite-runtime  # ~5 MB, ARM-optimised
```

The Facenet model needs to be converted:
```python
# On Mac/PC, convert once:
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(facenet_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantise to INT8
tflite_model = converter.convert()
open("facenet_int8.tflite", "wb").write(tflite_model)
```

Then replace `DeepFace.represent()` with a direct TFLite inference wrapper. deepface itself won't be used on Pi — it depends on full TF.

### 2. Replace RetinaFace with MediaPipe or YuNet

RetinaFace (current detector) is too slow on Pi (~500ms/frame). Alternatives:

| Detector | Pi 4 latency | Notes |
|---|---|---|
| `mediapipe` | ~30ms | Best choice — Google's ARM-optimised graph |
| `yunet` (OpenCV) | ~20ms | Built into OpenCV 4.8+, no extra install |
| `opencv` Haar | ~15ms | Least accurate but fastest |

Recommended for Pi: **YuNet** — zero extra deps (built into OpenCV), ~20ms, good accuracy.

```python
# config.json on Pi:
"detector_backend": "yunet"   # or "opencv" as fallback
```

### 3. Replace dlib with MediaPipe Face Mesh

dlib's 68-point landmark detector works on Pi but is slow (~150ms). MediaPipe Face Mesh gives 468 landmarks at ~20ms.

```bash
pip install mediapipe
```

The EAR calculation stays identical — just swap which landmark indices you read. MediaPipe's equivalent eye indices are documented in its Face Mesh topology map.

### 4. Camera: Use Pi Camera Module (CSI)

The Pi Camera (CSI ribbon cable) is faster and lower-latency than USB webcams.

```python
# config.json
"camera": { "source": 0 }   // still works for Pi Camera via v4l2

# Or use picamera2 for better control:
pip install picamera2
```

If using `picamera2`, wrap it in an OpenCV-compatible adapter.

### 5. Display / Headless Mode

Pi may run headless (no monitor) or with a small HDMI/DSI display.

Add to config:
```json
"display": {
  "enabled": true,          // false = headless, log to stdout only
  "width": 480              // smaller for 3.5" DSI displays
}
```

Guard all `cv2.imshow()` / `cv2.waitKey()` calls with the display flag.

### 6. GPIO Alerts (Optional but Realistic)

Replace the visual alert with physical hardware:

```python
import RPi.GPIO as GPIO
BUZZER_PIN = 17
LED_PIN = 27

def trigger_alert():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)   # sound buzzer
    GPIO.output(LED_PIN, GPIO.HIGH)      # flash LED
```

### 7. Pi-Specific requirements_pi.txt

```
# Core
numpy>=1.24.0
opencv-python>=4.8.0        # includes YuNet detector
mediapipe>=0.10.0
tflite-runtime>=2.14.0
scipy>=1.7.0
imutils>=0.5.4
picamera2>=0.3.0            # optional, for CSI camera
RPi.GPIO>=0.7.0             # optional, for buzzer/LED alerts

# Removed vs Mac version:
# tensorflow        (too large)
# dlib              (replaced by mediapipe)
# deepface          (replaced by direct TFLite)
# retina-face       (replaced by yunet/mediapipe)
```

### 8. Face Recognition Model Alternative

If Facenet TFLite is complex to convert, use **MobileFaceNet** instead:
- Pre-converted TFLite available publicly
- 99% of Facenet accuracy at 10% of the compute cost
- ~1MB model file vs 87MB Facenet

### Performance Budget (Pi 4, ARM Cortex-A72 @ 1.8GHz)

| Operation | Current (Mac) | Target (Pi) | How |
|---|---|---|---|
| Face detection | ~50ms (retinaface) | ~20ms | YuNet |
| Face embedding | ~200ms (Facenet/TF) | ~80ms | MobileFaceNet/TFLite |
| EAR calculation | ~5ms (dlib) | ~20ms | MediaPipe |
| Total auth frame | ~260ms | ~120ms | ~8 FPS auth |
| Total monitor frame | ~10ms | ~25ms | ~40 FPS monitor |

### Android on Raspberry Pi

Running Android (LineageOS) on Pi 4 is possible but adds significant complexity:
- The Python stack does not run on Android without Termux or similar
- The realistic path is a native Android app using:
  - **ML Kit Face Detection** (Google, free, on-device)
  - **TFLite Android API** for custom models
  - **CameraX** for camera access
- This is a full rewrite in Kotlin/Java, not a port of the Python code

**Recommendation:** Target Linux on Pi first, then consider an Android companion app later.

---

## Development Status

- [x] Mac development environment working
- [x] Face enrollment (`enroll_driver.py`)
- [x] Auth + drowsiness monitor state machine (`sdv_monitor.py`)
- [x] SQLite driver profile storage (`sdv_db.py`)
- [ ] Raspberry Pi port (TFLite + MediaPipe + YuNet)
- [ ] Headless/display toggle
- [ ] GPIO alert integration
- [ ] Android app
