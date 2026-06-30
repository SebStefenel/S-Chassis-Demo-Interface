#!/usr/bin/env bash
# setup_pi.sh — one-shot Pi setup: venv, packages, ONNX models.
# Run once after cloning: bash setup_pi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SDV Pi Setup ==="

# 1. Create virtualenv
echo "[1/4] Creating Python virtualenv…"
python3 -m venv venv
# shellcheck source=/dev/null
source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing Python packages…"
pip install --upgrade pip --quiet
pip install -r requirements_pi.txt

# 3. Download ONNX models (OpenCV model zoo, hosted on GitHub)
echo "[3/4] Downloading ONNX models…"
mkdir -p models

DET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
REC_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
LM_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if [ ! -f models/face_detection_yunet_2023mar.onnx ]; then
    echo "  Downloading YuNet face detector (~228 KB)…"
    wget -q --show-progress -O models/face_detection_yunet_2023mar.onnx "$DET_URL"
else
    echo "  YuNet model already present, skipping."
fi

if [ ! -f models/face_recognition_sface_2021dec.onnx ]; then
    echo "  Downloading SFace recogniser (~37 MB)…"
    wget -q --show-progress -O models/face_recognition_sface_2021dec.onnx "$REC_URL"
else
    echo "  SFace model already present, skipping."
fi

if [ ! -f models/face_landmarker.task ]; then
    echo "  Downloading MediaPipe FaceLandmarker (~3.6 MB)…"
    wget -q --show-progress -O models/face_landmarker.task "$LM_URL"
else
    echo "  FaceLandmarker model already present, skipping."
fi

# 4. Create database directory
echo "[4/4] Creating database directory…"
mkdir -p ~/.sdv

echo ""
echo "=== Setup complete ==="
echo ""
echo "Enrol a driver (do this before starting the monitor):"
echo "  source venv/bin/activate && python enrollment_server.py"
echo "  Then open http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo '<pi-ip>'):5000"
echo ""
echo "Run the monitor (Ctrl+C to stop):"
echo "  source venv/bin/activate && python sdv_monitor.py"
echo ""
echo "Install as a systemd service:"
echo "  sudo cp sdv.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable sdv"
echo "  sudo systemctl start sdv"
