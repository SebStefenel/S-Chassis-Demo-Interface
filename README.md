# SDV In-Vehicle AI Assistant Demo

A Software-Defined Vehicle (SDV) prototype that integrates driver authentication, a personalized AI chatbot, and real-time drowsiness monitoring — designed to run on a Raspberry Pi 5 with Eclipse S-CORE middleware.

---

## Overview

The system runs a three-phase state machine:

```
AUTH  →  CHAT  →  MONITOR  → (loop)
```

| Phase | Description |
|-------|-------------|
| **AUTH** | Identifies the driver via face recognition. Publishes authentication state to the vehicle signal bus. |
| **CHAT** | Opens a Streamlit chatbot personalised to the authenticated driver. Maintains persistent per-driver memory across sessions. |
| **MONITOR** | Continuously measures Eye Aspect Ratio (EAR) to detect drowsiness. Publishes alert level and drowsiness score to the vehicle signal bus. |

---

## Architecture

```
main.py (orchestrator)
├── sdv-driver-monitor/
│   ├── inference.py       ← YuNet (detection) + SFace (recognition) + MediaPipe (EAR)
│   ├── sdv_monitor.py     ← AUTH / MONITOR state machine
│   ├── signals.py         ← Kuksa Data Broker publisher (VSS signals)
│   ├── sdv_db.py          ← SQLite driver profiles & embeddings
│   ├── enroll_driver.py   ← CLI enrollment (with display)
│   └── enrollment_server.py ← Flask headless enrollment UI
└── llm-chatbot/
    └── sdv_chatbot.py     ← Streamlit chatbot (Llama 3.1 via Groq)
voice.py                   ← Shared offline STT (Vosk)
```

### VSS Signals Published

| Signal | Type | Values |
|--------|------|--------|
| `Vehicle.Driver.IsAuthenticated` | bool | |
| `Vehicle.Driver.Identifier` | string | driver name |
| `Vehicle.Driver.DrowsinessLevel` | uint8 | 0–10 |
| `Vehicle.Driver.AttentionLvl` | string | `ALERT` / `ATTENTIVE` |
| `Vehicle.Cabin.Seat.Row1.DriverSide.Position` | uint8 | |
| `Vehicle.Cabin.HVAC.AmbientAirTemperature` | float | |

---

## Requirements

- Raspberry Pi 5 (aarch64 Linux) — also runs on x86_64 for development
- USB camera or Pi CSI camera (v4l2)
- Microphone
- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)
- Eclipse S-CORE / Kuksa Data Broker running locally (optional — signals are skipped gracefully if unreachable)

---

## Setup

### Raspberry Pi (production)

```bash
cd sdv-driver-monitor
chmod +x setup_pi.sh
./setup_pi.sh          # creates venv, installs deps, downloads models
```

To start automatically on boot:

```bash
sudo cp sdv.service /etc/systemd/system/
sudo systemctl enable sdv
sudo systemctl start sdv
```

### Development / local

```bash
pip install -r requirements.txt

# configure Groq key
cp llm-chatbot/.env.example llm-chatbot/.env
# edit llm-chatbot/.env and add: GROQ_API_KEY=your_key_here
```

### Enroll a driver

With a display attached:

```bash
python sdv-driver-monitor/enroll_driver.py
```

Headless (browser-based):

```bash
python sdv-driver-monitor/enrollment_server.py
# then open http://<pi-ip>:5000 on any device
```

### Run

```bash
python main.py
```

---

## Configuration

Edit `sdv-driver-monitor/config.json`:

```json
{
  "face_recognition": {
    "detection_confidence": 0.75,
    "recognition_threshold": 0.40
  },
  "drowsiness": {
    "ear_threshold": 0.25,
    "consecutive_frames": 20
  },
  "camera_source": 0,
  "broker_address": "127.0.0.1:55555"
}
```

---

## Performance (Raspberry Pi 5)

| Component | Latency |
|-----------|---------|
| YuNet face detection | ~20 ms |
| SFace face recognition | ~40 ms |
| MediaPipe EAR | ~20 ms |
| **Total AUTH frame** | **~80 ms** |
| **Total MONITOR frame** | **~25 ms** |

---

## Open Source Models & Credits

This project would not be possible without the following open source models and frameworks.

### Face Detection — YuNet

- **Model:** `face_detection_yunet_2023mar.onnx`
- **Authors:** Shiqi Yu et al. (Shenzhen Technology University)
- **Source:** [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **License:** MIT
- **Paper:** [YuNet: A Tiny Millisecond-level Face Detector](https://doi.org/10.1007/s11633-023-1423-y)

### Face Recognition — SFace

- **Model:** `face_recognition_sface_2021dec.onnx`
- **Authors:** Zhong Yaoyao et al. (Institute of Automation, CAS)
- **Source:** [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- **License:** MIT
- **Paper:** [SFace: Privacy-friendly and Accurate Face Recognition using Synthetic Data](https://arxiv.org/abs/2206.03298)

### Facial Landmarks & Drowsiness — MediaPipe FaceLandmarker

- **Model:** `face_landmarker.task` (468-point face mesh)
- **Authors:** Google LLC
- **Source:** [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **License:** Apache 2.0
- Used to compute the Eye Aspect Ratio (EAR) for drowsiness detection.

### Large Language Model — Llama 3.1 8B

- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (served via [Groq](https://groq.com/))
- **Authors:** Meta AI
- **Source:** [Llama 3.1 on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- **License:** [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- Powers the in-vehicle chatbot, driver memory extraction, and history condensation.

### Speech-to-Text — Vosk / Kaldi

- **Model:** `vosk-model-small-en-us-0.15` (~50 MB, runs fully offline)
- **Vosk authors:** Alpha Cephei
- **Source:** [Vosk](https://alphacephei.com/vosk/) / [GitHub](https://github.com/alphacep/vosk-api)
- **License:** Apache 2.0
- Built on [Kaldi](https://kaldi-asr.org/) (Apache 2.0), developed by Daniel Povey et al.
- Provides offline, cloud-free voice input with no data leaving the vehicle.

### Computer Vision — OpenCV

- **Source:** [opencv.org](https://opencv.org/) / [GitHub](https://github.com/opencv/opencv)
- **License:** Apache 2.0
- Used for camera capture, the YuNet detector wrapper (`FaceDetectorYN`), and the SFace recognizer wrapper (`FaceRecognizerSF`).

---

## Framework Credits

| Framework | Purpose | License |
|-----------|---------|---------|
| [Streamlit](https://streamlit.io/) | Chatbot web UI | Apache 2.0 |
| [Flask](https://flask.palletsprojects.com/) | Headless enrollment server | BSD 3-Clause |
| [Eclipse Kuksa Data Broker](https://github.com/eclipse-kuksa/kuksa-databroker) | VSS signal bus (S-CORE) | Apache 2.0 |
| [kuksa-client](https://github.com/eclipse-kuksa/kuksa-python-sdk) | gRPC client for Data Broker | Apache 2.0 |
| [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) | Microphone capture | MIT |
| [NumPy](https://numpy.org/) | Numerical computing | BSD 3-Clause |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable loading | BSD 3-Clause |

---

## License

See individual component licenses above. Application code in this repository is provided as a demonstration prototype.