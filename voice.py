#!/usr/bin/env python3
"""
Shared voice / STT utilities — Vosk offline speech-to-text.

Used by:
  main.py       — keyword detection during cv2 AUTH popup and name enrollment
  sdv_chatbot.py — transcription of audio clips recorded via st.audio_input

The Vosk small-EN model (~50 MB) is downloaded automatically on first use
to ~/.sdv/vosk-model-small-en-us/
"""

import io
import json
import logging
import os
import time
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("sdv.voice")

_MODEL_DIR = Path(os.path.expanduser("~/.sdv/vosk-model-small-en-us"))
_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

_model = None   # lazily loaded


# ═════════════════════════════════════════════════════════════════════════════
#  Model management
# ═════════════════════════════════════════════════════════════════════════════

def _download_model() -> None:
    tmp = _MODEL_DIR.parent / "_vosk_download.zip"
    _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Vosk model (~50 MB) to %s …", tmp)
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    with zipfile.ZipFile(tmp, "r") as zf:
        top_dir = zf.namelist()[0].split("/")[0]
        zf.extractall(_MODEL_DIR.parent)
    extracted = _MODEL_DIR.parent / top_dir
    extracted.rename(_MODEL_DIR)
    tmp.unlink(missing_ok=True)
    log.info("Vosk model ready at %s", _MODEL_DIR)


def _ensure_model():
    global _model
    if _model is not None:
        return _model
    try:
        import vosk
    except ImportError:
        raise RuntimeError("vosk not installed — run: pip install vosk")
    if not _MODEL_DIR.exists():
        _download_model()
    vosk.SetLogLevel(-1)
    _model = vosk.Model(str(_MODEL_DIR))
    log.info("Vosk STT model loaded.")
    return _model


# ═════════════════════════════════════════════════════════════════════════════
#  Transcription from recorded bytes (for Streamlit st.audio_input)
# ═════════════════════════════════════════════════════════════════════════════

def transcribe_wav(wav_bytes: bytes) -> str:
    """
    Transcribe WAV audio bytes and return the transcript string.
    wav_bytes can come from st.audio_input (returns UploadedFile — call .read() first).
    Returns empty string if nothing was heard.
    """
    import vosk
    model = _ensure_model()

    try:
        wf = wave.open(io.BytesIO(wav_bytes), "rb")
    except Exception as exc:
        log.warning("Could not read WAV bytes: %s", exc)
        return ""

    rec  = vosk.KaldiRecognizer(model, wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            text += json.loads(rec.Result()).get("text", "")
    text += json.loads(rec.FinalResult()).get("text", "")
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
#  Live microphone capture (for main.py — keyword detection and enrollment)
# ═════════════════════════════════════════════════════════════════════════════

def listen_once(timeout: float = 6.0, sample_rate: int = 16000) -> str:
    """
    Record from the microphone until the first complete utterance or timeout.
    Blocking — call from a background thread if the camera loop must keep running.
    Returns the transcript string (may be empty).
    """
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError("pyaudio not installed — run: pip install pyaudio")
    import vosk

    model = _ensure_model()
    rec   = vosk.KaldiRecognizer(model, sample_rate)

    pa = pyaudio.PyAudio()
    # Try default device; fall back to device_index=0 on Pi if needed
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=4000,
        )
    except OSError:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=0,
            frames_per_buffer=4000,
        )

    stream.start_stream()
    text     = ""
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                partial = json.loads(rec.Result()).get("text", "")
                if partial:
                    text = partial
                    break   # got a complete utterance — stop early
        text += json.loads(rec.FinalResult()).get("text", "")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return text.strip()
