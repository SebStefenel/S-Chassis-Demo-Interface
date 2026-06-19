#!/usr/bin/env python3
"""
SDV Monitor — unified driver-monitoring pipeline
-------------------------------------------------
State machine with a single cv2.VideoCapture instance:

  AUTH        → scans the live feed with Facenet (via deepface) every N frames.
                Transitions to MONITOR on a match, or to UNRECOGNIZED on timeout.

  UNRECOGNIZED → shows an options overlay.  Keys: [E] enroll  [G] guest  [R] retry

  ENROLLING   → captures face embeddings inline without releasing the camera.
                Transitions to MONITOR when enough samples are collected.

  MONITOR     → runs the dlib 68-point EAR drowsiness detector on every frame.
                Keys: [A] re-authenticate  [Q] quit

Usage
-----
  python sdv_monitor.py
"""

import json
import os
import sys
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "deepface"))

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as scipy_dist

from deepface import DeepFace

import sdv_db


# ── state enum ────────────────────────────────────────────────────────────────

class State(Enum):
    AUTH = auto()
    UNRECOGNIZED = auto()
    ENROLLING = auto()
    MONITOR = auto()


# ── pure functions ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(os.path.join(_HERE, "config.json")) as f:
        return json.load(f)


def _resolve_db_path(config: dict) -> str:
    raw = config["database"]["path"]
    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1.0 - np.dot(a, b) / (denom + 1e-10))


def _ear(eye: np.ndarray) -> float:
    A = scipy_dist.euclidean(eye[1], eye[5])
    B = scipy_dist.euclidean(eye[2], eye[4])
    C = scipy_dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def _extract_embedding(
    frame: np.ndarray, model_name: str, detector_backend: str
) -> "Optional[np.ndarray]":
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
        )
        return np.array(results[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def _best_match(
    query: np.ndarray,
    enrolled: List[Tuple[int, str, np.ndarray]],
    threshold: float,
) -> "Optional[Tuple[int, str, float]]":
    """Return (user_id, username, distance) of best match or None."""
    if not enrolled:
        return None
    best_dist = float("inf")
    best_uid: Optional[int] = None
    best_name: Optional[str] = None
    for uid, uname, emb in enrolled:
        d = _cosine_dist(query, emb)
        if d < best_dist:
            best_dist, best_uid, best_name = d, uid, uname
    if best_dist < threshold:
        return (best_uid, best_name, best_dist)
    return None


# ── overlay drawing helpers ───────────────────────────────────────────────────

def _bar(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (15, 15, 15), -1)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def _lines(
    frame: np.ndarray,
    texts: List[str],
    y0: int = 58,
    color: Tuple[int, int, int] = (210, 210, 210),
) -> None:
    for i, t in enumerate(texts):
        cv2.putText(frame, t, (10, y0 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)


# ── background auth worker ────────────────────────────────────────────────────

class _AuthWorker:
    """Runs deepface.represent in a daemon thread so the video loop stays smooth."""

    def __init__(self) -> None:
        self._result: Optional[np.ndarray] = None
        self._busy = False
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, frame: np.ndarray, model_name: str, detector_backend: str) -> None:
        if self._busy:
            return
        self._busy = True
        t = threading.Thread(
            target=self._run,
            args=(frame.copy(), model_name, detector_backend),
            daemon=True,
        )
        t.start()

    def _run(self, frame: np.ndarray, model_name: str, detector_backend: str) -> None:
        emb = _extract_embedding(frame, model_name, detector_backend)
        with self._lock:
            self._result = emb
            self._busy = False

    def pop_result(self) -> "Optional[np.ndarray]":
        with self._lock:
            r = self._result
            self._result = None
        return r


# ── main pipeline class ───────────────────────────────────────────────────────

class SDVMonitor:

    def __init__(self, config: dict) -> None:
        self.config = config
        self.db_path = _resolve_db_path(config)
        self.fr = config["face_recognition"]
        self.drw = config["drowsiness"]
        self.cam_cfg = config["camera"]
        self.enroll_cfg = config["enrollment"]

        # pipeline state
        self.state = State.AUTH
        self.driver_name: Optional[str] = None
        self.driver_id: Optional[int] = None
        self.driver_settings: dict = {}

        # auth sub-state
        self._auth_start = time.time()
        self._auth_frame = 0
        self._auth_worker = _AuthWorker()
        self._enrolled: List[Tuple[int, str, np.ndarray]] = []

        # drowsiness sub-state
        self._ear_flag = 0
        self._last_ear = 0.0
        self._is_drowsy = False

        # inline-enrollment sub-state
        self._enroll_user: Optional[str] = None
        self._enroll_uid: Optional[int] = None
        self._enroll_count = 0
        self._enroll_frame = 0

        # ── load dlib models (shared across states) ───────────────────────────
        landmark_path = os.path.join(_HERE, self.drw["landmark_model_path"])
        if not os.path.exists(landmark_path):
            raise FileNotFoundError(
                f"dlib landmark model not found:\n  {landmark_path}\n"
                "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "Then: bunzip2 shape_predictor_68_face_landmarks.dat.bz2 and place under "
                "Drowsiness_Detection/models/"
            )
        self._dlib_det = dlib.get_frontal_face_detector()
        self._dlib_pred = dlib.shape_predictor(landmark_path)
        self._l_start, self._l_end = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self._r_start, self._r_end = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    # ── state transitions ─────────────────────────────────────────────────────

    def _to_auth(self) -> None:
        print("[SDV] ── AUTH ──────────────────────────────")
        self.state = State.AUTH
        self.driver_name = None
        self.driver_id = None
        self.driver_settings = {}
        self._auth_start = time.time()
        self._auth_frame = 0
        self._ear_flag = 0
        self._is_drowsy = False
        self._enrolled = sdv_db.get_all_embeddings(self.db_path)
        print(f"[SDV] {len(set(n for _, n, _ in self._enrolled))} driver(s) loaded.")

    def _to_monitor(self, name: str, uid: Optional[int], settings: dict) -> None:
        print(f"[SDV] ── MONITOR ── Driver: {name} ────────")
        self.state = State.MONITOR
        self.driver_name = name
        self.driver_id = uid
        self.driver_settings = settings
        self._ear_flag = 0
        self._is_drowsy = False
        if uid is not None:
            print(f"[SDV] Vehicle settings loaded: {settings}")

    def _begin_enroll(self, username: str) -> None:
        print(f"[SDV] ── ENROLLING ── {username} ──────────")
        existing = sdv_db.get_user_by_name(self.db_path, username)
        if existing:
            sdv_db.delete_user(self.db_path, existing["id"])
        uid = sdv_db.add_user(self.db_path, username,
                               self.config.get("vehicle_settings_defaults", {}))
        self._enroll_user = username
        self._enroll_uid = uid
        self._enroll_count = 0
        self._enroll_frame = 0
        self.state = State.ENROLLING

    # ── per-frame processing ──────────────────────────────────────────────────

    def _tick_auth(self, frame: np.ndarray) -> None:
        self._auth_frame += 1
        elapsed = time.time() - self._auth_start
        timeout = self.fr["auth_timeout_seconds"]

        if elapsed > timeout:
            print("[SDV] Auth timeout → UNRECOGNIZED")
            self.state = State.UNRECOGNIZED
            return

        # submit a frame to the background worker every N frames
        interval = self.fr["auth_check_every_n_frames"]
        if self._auth_frame % interval == 0 and not self._auth_worker.busy:
            self._auth_worker.submit(frame, self.fr["model_name"], self.fr["detector_backend"])

        # check if worker has a result
        emb = self._auth_worker.pop_result()
        if emb is None:
            return

        match = _best_match(emb, self._enrolled, self.fr["match_threshold"])
        if match:
            uid, uname, dist = match
            print(f"[SDV] Recognized: {uname} (cosine dist={dist:.4f})")
            settings = sdv_db.get_user_settings(self.db_path, uid)
            self._to_monitor(uname, uid, settings)

    def _tick_monitor(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._dlib_det(gray, 0)
        if not faces:
            # no face visible — don't increment drowsiness flag but don't reset it either
            return

        shape = face_utils.shape_to_np(self._dlib_pred(gray, faces[0]))
        left_eye = shape[self._l_start:self._l_end]
        right_eye = shape[self._r_start:self._r_end]
        self._last_ear = (_ear(left_eye) + _ear(right_eye)) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 220, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 220, 0), 1)

        if self._last_ear < self.drw["ear_threshold"]:
            self._ear_flag += 1
        else:
            self._ear_flag = 0

        was_drowsy = self._is_drowsy
        self._is_drowsy = self._ear_flag >= self.drw["consecutive_frames_alert"]
        if self._is_drowsy and not was_drowsy:
            print("[SDV] *** DROWSINESS DETECTED — ALERT TRIGGERED ***")

    def _tick_enroll(self, frame: np.ndarray) -> None:
        self._enroll_frame += 1
        interval = self.enroll_cfg["capture_interval_frames"]
        target = self.enroll_cfg["num_sample_frames"]

        if self._enroll_frame % interval != 0:
            return

        emb = _extract_embedding(frame, self.fr["model_name"], self.fr["detector_backend"])
        if emb is not None:
            sdv_db.add_embedding(self.db_path, self._enroll_uid, emb)
            self._enroll_count += 1
            print(f"  Enrollment sample {self._enroll_count}/{target}")

        if self._enroll_count >= target:
            print(f"[SDV] Enrollment complete for '{self._enroll_user}'.")
            settings = sdv_db.get_user_settings(self.db_path, self._enroll_uid)
            self._enrolled = sdv_db.get_all_embeddings(self.db_path)
            self._to_monitor(self._enroll_user, self._enroll_uid, settings)

    # ── overlay drawing ───────────────────────────────────────────────────────

    def _draw_auth(self, frame: np.ndarray) -> None:
        remaining = max(0.0, self.fr["auth_timeout_seconds"] -
                        (time.time() - self._auth_start))
        status = "scanning…" if not self._auth_worker.busy else "analyzing…"
        _bar(frame, f"[AUTH]  {status}  {remaining:.1f}s remaining", (0, 200, 255))
        n_drivers = len(set(n for _, n, _ in self._enrolled))
        _lines(frame, [
            "Driver: Unknown / Searching",
            f"Enrolled drivers in DB: {n_drivers}",
            "Press [Q] to quit",
        ])

    def _draw_unrecognized(self, frame: np.ndarray) -> None:
        _bar(frame, "[UNRECOGNIZED]  No matching driver profile", (0, 80, 255))
        _lines(frame, [
            "Face not matched against any enrolled profile.",
            "",
            "  [E]  Enroll as new driver",
            "  [G]  Continue as Guest (no profile loaded)",
            "  [R]  Retry authentication",
            "  [Q]  Quit",
        ], color=(200, 200, 255))

    def _draw_enrolling(self, frame: np.ndarray) -> None:
        target = self.enroll_cfg["num_sample_frames"]
        _bar(frame,
             f"[ENROLLING]  {self._enroll_user}  |  "
             f"{self._enroll_count}/{target} samples",
             (0, 255, 200))
        pct = int(self._enroll_count / target * frame.shape[1])
        cv2.rectangle(frame, (0, 44), (pct, 52), (0, 200, 120), -1)
        _lines(frame, [
            "Face the camera and stay still.",
            "Capturing face embeddings…",
        ])

    def _draw_monitor(self, frame: np.ndarray) -> None:
        if self._is_drowsy:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            _bar(frame, "*** DROWSINESS ALERT!  PULL OVER NOW! ***", (0, 0, 255))
            alert_color = (0, 0, 255)
        else:
            _bar(frame, f"[MONITORING]  Driver: {self.driver_name}", (0, 210, 0))
            alert_color = (0, 220, 0)

        ear_color = (0, 0, 255) if self._is_drowsy else (0, 220, 0)
        status_str = "DROWSY" if self._is_drowsy else "Alert"
        _lines(frame, [
            f"EAR: {self._last_ear:.3f}   |   Status: {status_str}",
            f"Consecutive low-EAR frames: "
            f"{self._ear_flag} / {self.drw['consecutive_frames_alert']}",
            "",
            "  [A] Re-authenticate   [Q] Quit",
        ], color=ear_color)

        if self.driver_settings:
            h = frame.shape[0]
            seat = self.driver_settings.get("seat_position", "—")
            temp = self.driver_settings.get("climate_temp_c", "—")
            radio = self.driver_settings.get("preferred_radio_station", "—")
            cv2.putText(
                frame,
                f"Seat: {seat}  |  Climate: {temp}°C  |  Radio: {radio}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1,
            )

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        sdv_db.init_db(self.db_path)
        self._to_auth()

        cam_source = self.cam_cfg["source"]
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {cam_source}")

        display_width = self.cam_cfg["display_width"]
        print(f"[SDV] Camera opened (source={cam_source}).  Starting pipeline.")
        print("[SDV] Keys: [Q] quit | [A] re-auth (monitor) | [E/G/R] (unrecognized)")

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[SDV] Camera read failed — exiting.")
                break

            h, w = raw_frame.shape[:2]
            scale = display_width / w
            display = cv2.resize(raw_frame, (display_width, int(h * scale)))

            # ── per-state tick (inference on full-res frame, draw on display) ──
            if self.state == State.AUTH:
                self._tick_auth(raw_frame)
                self._draw_auth(display)

            elif self.state == State.UNRECOGNIZED:
                self._draw_unrecognized(display)

            elif self.state == State.ENROLLING:
                self._tick_enroll(raw_frame)
                self._draw_enrolling(display)

            elif self.state == State.MONITOR:
                self._tick_monitor(display)   # dlib runs at display resolution — sufficient
                self._draw_monitor(display)

            cv2.imshow("SDV Monitor", display)
            key = cv2.waitKey(1) & 0xFF

            # ── global key: quit ──────────────────────────────────────────────
            if key == ord("q"):
                print("[SDV] Quit.")
                break

            # ── MONITOR keys ──────────────────────────────────────────────────
            elif self.state == State.MONITOR and key == ord("a"):
                self._to_auth()

            # ── UNRECOGNIZED keys ─────────────────────────────────────────────
            elif self.state == State.UNRECOGNIZED:
                if key == ord("g"):
                    self._to_monitor("Guest", None, {})

                elif key == ord("r"):
                    self._to_auth()

                elif key == ord("e"):
                    # Release camera briefly to read username from the terminal.
                    # This is the most portable approach — OpenCV has no text-input widget.
                    cap.release()
                    cv2.destroyAllWindows()

                    username = input("[SDV] Enter new driver name: ").strip()
                    if not username:
                        username = "Driver"

                    cap = cv2.VideoCapture(cam_source)
                    if not cap.isOpened():
                        print("[SDV] ERROR: Could not re-open camera after enrollment prompt.")
                        break

                    self._begin_enroll(username)

        cap.release()
        cv2.destroyAllWindows()
        print("[SDV] Pipeline stopped.")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config = _load_config()
    monitor = SDVMonitor(config)
    monitor.run()


if __name__ == "__main__":
    main()
