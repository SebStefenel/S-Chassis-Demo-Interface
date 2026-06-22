#!/usr/bin/env python3
"""
SDV Driver Monitor — headless pipeline for Raspberry Pi + Eclipse S-CORE.

State machine:
  AUTH    → runs SFace embedding every N frames, compares against enrolled drivers.
            On match: transitions to MONITOR and publishes authenticated signal.
            On timeout with no match: retries (logs unrecognized, publishes signal).
  MONITOR → MediaPipe EAR drowsiness detection, publishes DrowsinessLevel signal.

Signals published to Eclipse S-CORE (Kuksa Data Broker, gRPC port 55555):
  Vehicle.Driver.IsAuthenticated
  Vehicle.Driver.Identifier
  Vehicle.Driver.DrowsinessLevel
  Vehicle.Driver.AttentionLvl
  Vehicle.Cabin.Seat.Row1.DriverSide.Position
  Vehicle.Cabin.HVAC.AmbientAirTemperature

Usage:
  python sdv_monitor.py
  python sdv_monitor.py --config /etc/sdv/config.json
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

import sdv_db
from inference import FaceEngine
from signals import SCOREPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdv.monitor")

_HERE = os.path.dirname(os.path.abspath(__file__))


class State(Enum):
    AUTH = auto()
    MONITOR = auto()


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_db(config: dict) -> str:
    raw = config["database"]["path"]
    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _best_match(
    query: np.ndarray,
    enrolled: List[Tuple[int, str, np.ndarray]],
    threshold: float,
) -> Optional[Tuple[int, str, float]]:
    if not enrolled:
        return None
    best_dist = float("inf")
    best_uid, best_name = None, None
    for uid, uname, emb in enrolled:
        d = FaceEngine.cosine_dist(query, emb)
        if d < best_dist:
            best_dist, best_uid, best_name = d, uid, uname
    return (best_uid, best_name, best_dist) if best_dist < threshold else None


class SDVMonitor:

    def __init__(self, config: dict) -> None:
        self.cfg = config
        fr = config["face_recognition"]
        drw = config["drowsiness"]

        self.db_path = _resolve_db(config)
        self.match_threshold = fr["match_threshold"]
        self.auth_timeout = fr["auth_timeout_seconds"]
        self.auth_interval = fr["auth_check_every_n_frames"]
        self.ear_thresh = drw["ear_threshold"]
        self.ear_frames = drw["consecutive_frames_alert"]

        models = config["models"]
        self.engine = FaceEngine(
            os.path.join(_HERE, models["det_model"]),
            os.path.join(_HERE, models["rec_model"]),
        )

        score_cfg = config.get("score", {})
        self.pub = SCOREPublisher(
            host=score_cfg.get("host", "127.0.0.1"),
            port=score_cfg.get("port", 55555),
        )

        self.state = State.AUTH
        self._enrolled: List[Tuple[int, str, np.ndarray]] = []
        self._auth_start = 0.0
        self._auth_frame = 0
        self._ear_flag = 0
        self._is_drowsy = False
        self._driver_name: Optional[str] = None
        self._running = True

    # ── state transitions ─────────────────────────────────────────────────────

    def _enter_auth(self) -> None:
        self.state = State.AUTH
        self._auth_start = time.time()
        self._auth_frame = 0
        self._enrolled = sdv_db.get_all_embeddings(self.db_path)
        self._driver_name = None
        unique = len(set(n for _, n, _ in self._enrolled))
        log.info("AUTH — %d driver(s) in database. Scanning…", unique)
        self.pub.scanning()

    def _enter_monitor(self, uid: int, name: str) -> None:
        settings = sdv_db.get_user_settings(self.db_path, uid)
        self.state = State.MONITOR
        self._driver_name = name
        self._ear_flag = 0
        self._is_drowsy = False
        log.info("MONITOR — driver authenticated: %s", name)
        self.pub.authenticated(name, settings)

    # ── per-frame ticks ───────────────────────────────────────────────────────

    def _tick_auth(self, frame: np.ndarray) -> None:
        self._auth_frame += 1

        if time.time() - self._auth_start > self.auth_timeout:
            log.info("Auth timeout — no match found. Retrying.")
            self.pub.unrecognized()
            self._enter_auth()
            return

        if self._auth_frame % self.auth_interval != 0:
            return

        emb = self.engine.embed(frame)
        if emb is None:
            return

        match = _best_match(emb, self._enrolled, self.match_threshold)
        if match:
            uid, name, dist = match
            log.info("Face matched: %s (cosine dist=%.4f)", name, dist)
            self._enter_monitor(uid, name)
        else:
            log.debug("Face detected but not matched (best dist > %.2f)", self.match_threshold)

    def _tick_monitor(self, frame: np.ndarray) -> None:
        ear = self.engine.ear(frame)
        if ear is None:
            return

        if ear < self.ear_thresh:
            self._ear_flag += 1
        else:
            self._ear_flag = 0

        was_drowsy = self._is_drowsy
        self._is_drowsy = self._ear_flag >= self.ear_frames
        self.pub.drowsiness(ear, self._is_drowsy)

        if self._is_drowsy and not was_drowsy:
            log.warning("*** DROWSINESS ALERT — driver: %s  EAR=%.3f ***", self._driver_name, ear)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        sdv_db.init_db(self.db_path)
        self._enter_auth()

        cam_src = self.cfg["camera"]["source"]
        cap = cv2.VideoCapture(cam_src)
        if not cap.isOpened():
            log.error("Cannot open camera source: %s", cam_src)
            sys.exit(1)
        log.info("Camera opened (source=%s).", cam_src)

        def _handle_stop(sig, frame):
            log.info("Signal %d received — shutting down.", sig)
            self._running = False

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — retrying in 100ms.")
                time.sleep(0.1)
                continue

            if self.state is State.AUTH:
                self._tick_auth(frame)
            else:
                self._tick_monitor(frame)

            time.sleep(0.02)  # ~50 FPS ceiling; keeps CPU usage sane on Pi

        cap.release()
        self.pub.close()
        log.info("Monitor stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SDV headless driver monitor")
    parser.add_argument(
        "--config", default=os.path.join(_HERE, "config.json"),
        help="Path to config.json (default: ./config.json)",
    )
    args = parser.parse_args()
    SDVMonitor(_load_config(args.config)).run()


if __name__ == "__main__":
    main()
