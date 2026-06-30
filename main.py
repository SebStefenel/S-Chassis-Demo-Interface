#!/usr/bin/env python3
"""
S-Chassis Demo Interface — Main Orchestrator
============================================
3-phase state machine:

  AUTH    → Camera on. Face recognition runs until a driver is matched.
  CHAT    → Camera released. Streamlit chatbot launches in browser,
             pre-loaded with that driver's history and greeting them by name.
             Waits until the driver clicks "End Session".
  MONITOR → Camera back on. Drowsiness detection runs until quit or signal.
             Then loops back to AUTH for the next driver.

Run (with display attached):
    python main.py

Run (headless Pi — open browser at http://<pi-ip>:8501 during CHAT):
    python main.py --no-display
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.abspath(__file__))
_MONITOR_DIR = os.path.join(_ROOT, "sdv-driver-monitor")
_CHATBOT_DIR = os.path.join(_ROOT, "llm-chatbot")

sys.path.insert(0, _MONITOR_DIR)

import sdv_db
from inference import FaceEngine
from signals import SCOREPublisher

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdv.main")

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_WINDOW = "S-Chassis Demo"


# ═════════════════════════════════════════════════════════════════════════════
#  Config loader
# ═════════════════════════════════════════════════════════════════════════════

def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_db(config: dict) -> str:
    raw  = config["database"]["path"]
    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(_MONITOR_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class DemoOrchestrator:

    def __init__(self, config: dict, display: bool) -> None:
        self.cfg             = config
        self.display_enabled = display
        self.display_width   = config.get("display", {}).get("width", 640)

        fr = config["face_recognition"]
        self.match_threshold = fr["match_threshold"]
        self.auth_timeout    = fr["auth_timeout_seconds"]
        self.auth_interval   = fr["auth_check_every_n_frames"]

        drw = config["drowsiness"]
        self.ear_thresh = drw["ear_threshold"]
        self.ear_frames = drw["consecutive_frames_alert"]

        self.db_path = _resolve_db(config)

        models = config["models"]
        self.engine = FaceEngine(
            os.path.join(_MONITOR_DIR, models["det_model"]),
            os.path.join(_MONITOR_DIR, models["rec_model"]),
            os.path.join(_MONITOR_DIR, models.get("landmark_model", "")),
        )

        score_cfg = config.get("score", {})
        self.pub = SCOREPublisher(
            host=score_cfg.get("host", "127.0.0.1"),
            port=score_cfg.get("port", 55555),
        )

        self._running = True

    # ── AUTH ──────────────────────────────────────────────────────────────────

    def _run_auth(self, cap) -> Optional[Tuple[int, str]]:
        """
        Face recognition loop.
        Returns (uid, driver_name) on a successful match, or None if the user
        pressed Q / a shutdown signal was received.
        """
        enrolled = sdv_db.get_all_embeddings(self.db_path)
        unique   = len(set(n for _, n, _ in enrolled))
        log.info("AUTH — %d driver(s) enrolled. Scanning…", unique)
        self.pub.scanning()

        deadline = time.time() + self.auth_timeout
        frame_n  = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_n += 1

            if self.display_enabled:
                cv2.imshow(_WINDOW, self._draw_auth(frame, deadline))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return None
            else:
                time.sleep(0.02)  # ~50 fps ceiling; keeps Pi CPU usage sane

            # Only run face embedding every N frames to save CPU on Pi
            if frame_n % self.auth_interval == 0:
                emb = self.engine.embed(frame)
                if emb is not None:
                    match = self._best_match(emb, enrolled)
                    if match:
                        uid, name, dist = match
                        log.info("Face matched: %s (cosine dist=%.4f)", name, dist)
                        return uid, name

            if time.time() > deadline:
                log.info("Auth timeout — no match.")
                self.pub.unrecognized()

                action = self._run_unrecognized_prompt(cap)
                if action == "quit":
                    return None
                elif action == "enroll":
                    result = self._run_inline_enrollment(cap)
                    if result:
                        return result   # (uid, name) — skip straight to CHAT
                # 'retry' or cancelled enrollment → reset and scan again
                deadline = time.time() + self.auth_timeout
                frame_n  = 0
                enrolled = sdv_db.get_all_embeddings(self.db_path)

        return None

    def _best_match(
        self, query: np.ndarray, enrolled: list
    ) -> Optional[Tuple[int, str, float]]:
        if not enrolled:
            return None
        best_dist = float("inf")
        best_uid = best_name = None
        for uid, name, emb in enrolled:
            d = FaceEngine.cosine_dist(query, emb)
            if d < best_dist:
                best_dist, best_uid, best_name = d, uid, name
        return (best_uid, best_name, best_dist) if best_dist < self.match_threshold else None

    # ── Unrecognized-driver prompt ────────────────────────────────────────────

    def _run_unrecognized_prompt(self, cap) -> str:
        """
        Show a popup asking whether to enroll or retry.
        Returns 'enroll', 'retry', or 'quit'.
        On headless Pi the same question is printed to the terminal (SSH session).
        """
        if not self.display_enabled:
            print("\n[S-Chassis] Driver not recognised.")
            print("  [N]  Create new profile")
            print("  [R]  Try again")
            print("  [Q]  Quit")
            while True:
                choice = input("Choose: ").strip().lower()
                if choice == "n":
                    return "enroll"
                elif choice == "r":
                    return "retry"
                elif choice == "q":
                    return "quit"

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            cv2.imshow(_WINDOW, self._draw_unrecognized_popup(frame))
            key = cv2.waitKey(100) & 0xFF
            if key == ord("n"):
                return "enroll"
            elif key == ord("r"):
                return "retry"
            elif key in (ord("q"), 27):
                return "quit"
        return "quit"

    def _draw_unrecognized_popup(self, frame: np.ndarray) -> np.ndarray:
        disp = self._resize(frame)
        dh, dw = disp.shape[:2]

        # Dim the background
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (dw, dh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)

        # Panel
        pw, ph = min(500, dw - 40), 210
        x1 = (dw - pw) // 2
        y1 = (dh - ph) // 2
        cv2.rectangle(disp, (x1, y1), (x1 + pw, y1 + ph), (28, 28, 28), -1)
        cv2.rectangle(disp, (x1, y1), (x1 + pw, y1 + ph), (90, 90, 90), 2)

        cx = dw // 2
        cv2.putText(disp, "Driver not recognised",
                    (cx - 185, y1 + 42), _FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "[N]  Create new profile",
                    (x1 + 30, y1 + 95), _FONT, 0.70, (80, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(disp, "[R]  Try again",
                    (x1 + 30, y1 + 140), _FONT, 0.70, (60, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "[Q]  Quit",
                    (x1 + 30, y1 + 185), _FONT, 0.60, (130, 130, 130), 1, cv2.LINE_AA)
        return disp

    # ── Inline enrollment ─────────────────────────────────────────────────────

    def _run_inline_enrollment(self, cap) -> Optional[Tuple[int, str]]:
        """
        Capture face embeddings for a new driver directly from the live camera.
        Name is entered via the terminal (one-time setup; SSH works fine on Pi).
        Returns (uid, name) on success, None on cancel.
        """
        # Briefly show "check the terminal" so the user isn't confused
        if self.display_enabled:
            ret, frame = cap.read()
            if ret:
                disp = self._resize(frame)
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (18, 18, 18), -1)
                cv2.putText(disp, "Type your name in the terminal, then press Enter…",
                            (10, 37), _FONT, 0.65, (255, 200, 60), 2, cv2.LINE_AA)
                cv2.imshow(_WINDOW, disp)
                cv2.waitKey(200)

        name = input("\nEnter a name for the new profile: ").strip()
        if not name:
            log.info("Enrollment cancelled — no name entered.")
            return None

        existing = sdv_db.get_user_by_name(self.db_path, name)
        if existing:
            log.info("Profile '%s' already exists — adding more samples.", name)
            uid = existing["id"]
        else:
            defaults = self.cfg.get("vehicle_settings_defaults", {
                "seat_position": 3, "mirror_angle": 0,
                "climate_temp_c": 22, "preferred_radio_station": "Unknown",
            })
            uid = sdv_db.add_user(self.db_path, name, defaults)
            log.info("Created new profile: %s (uid=%d)", name, uid)

        enr_cfg   = self.cfg.get("enrollment", {})
        num_frames = enr_cfg.get("num_sample_frames", 10)
        interval   = enr_cfg.get("capture_interval_frames", 8)
        captured   = 0
        frame_n    = 0
        log.info("Enrolling %s — capturing %d face samples…", name, num_frames)

        while captured < num_frames and self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_n += 1

            if self.display_enabled:
                cv2.imshow(_WINDOW, self._draw_enrollment(frame, name, captured, num_frames))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    log.info("Enrollment cancelled mid-capture.")
                    return None
            else:
                time.sleep(0.02)

            if frame_n % interval == 0:
                emb = self.engine.embed(frame)
                if emb is not None:
                    sdv_db.add_embedding(self.db_path, uid, emb)
                    captured += 1
                    log.info("  Captured %d/%d", captured, num_frames)
                else:
                    log.debug("  No face detected — keep facing the camera.")

        if captured < num_frames:
            log.warning("Enrollment incomplete (%d/%d) — profile saved with partial data.",
                        captured, num_frames)
            return None

        log.info("Enrollment complete for %s.", name)
        return uid, name

    def _draw_enrollment(self, frame: np.ndarray, name: str,
                          captured: int, total: int) -> np.ndarray:
        disp = self._resize(frame)
        dh, dw = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (dw, 52), (18, 18, 18), -1)
        cv2.putText(disp, f"ENROLLING: {name}   {captured}/{total}",
                    (10, 37), _FONT, 0.80, (255, 200, 60), 2, cv2.LINE_AA)
        bar_w = int((captured / total) * (dw - 20))
        cv2.rectangle(disp, (10, 58), (dw - 10, 76), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(disp, (10, 58), (10 + bar_w, 76), (80, 220, 80), -1)
        cv2.putText(disp, "Look directly at the camera",
                    (10, 100), _FONT, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, "Q — cancel", (8, dh - 8),
                    _FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        return disp

    # ── CHAT ──────────────────────────────────────────────────────────────────

    def _run_chat(self, uid: int, name: str) -> None:
        """
        Launch the Streamlit chatbot for this driver and block until it exits.

        The driver's name is passed via the SDV_DRIVER_NAME environment variable.
        The chatbot loads that driver's personal history file and greets them by name.
        When the driver clicks "End Session", the chatbot calls os._exit(0) which
        terminates the subprocess and unblocks proc.wait() here.

        On a headless Pi the user opens http://<pi-ip>:8501 in any browser on the
        same network.  On a Pi with a display the browser opens automatically if
        server.headless is false — adjust the flag below as needed.
        """
        settings = sdv_db.get_user_settings(self.db_path, uid)
        self.pub.authenticated(name, settings)

        if self.display_enabled:
            cv2.destroyAllWindows()

        log.info("Launching chatbot for driver: %s", name)
        log.info("Chat UI available at  http://0.0.0.0:8501  (or <pi-ip>:8501)")

        env = os.environ.copy()
        env["SDV_DRIVER_NAME"] = name

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                os.path.join(_CHATBOT_DIR, "sdv_chatbot.py"),
                "--server.headless",          "true",
                "--server.address",           "0.0.0.0",
                "--server.port",              "8501",
                "--browser.gatherUsageStats", "false",
            ],
            env=env,
            cwd=_CHATBOT_DIR,   # so .env and relative paths resolve correctly
        )

        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()

        log.info("Chat session ended for driver: %s. Starting drowsiness monitor.", name)

    # ── MONITOR ───────────────────────────────────────────────────────────────

    def _run_monitor(self, cap, name: str) -> None:
        """
        Drowsiness detection loop.
        Returns on quit key (Q / Esc) or when _running is set to False by a signal.
        After returning, the main loop restarts AUTH for the next session.
        """
        log.info("MONITOR — watching driver: %s", name)

        ear_flag  = 0
        is_drowsy = False
        last_ear: Optional[float] = None

        if self.display_enabled:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW, self.display_width, int(self.display_width * 0.75))

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            ear = self.engine.ear(frame)
            if ear is not None:
                last_ear = ear
                ear_flag = ear_flag + 1 if ear < self.ear_thresh else 0
                was_drowsy = is_drowsy
                is_drowsy  = ear_flag >= self.ear_frames
                self.pub.drowsiness(ear, is_drowsy)
                if is_drowsy and not was_drowsy:
                    log.warning("DROWSINESS ALERT — %s  EAR=%.3f", name, ear)

            if self.display_enabled:
                cv2.imshow(_WINDOW, self._draw_monitor(frame, name, last_ear, is_drowsy))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.02)

    # ── Display helpers ───────────────────────────────────────────────────────

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w  = frame.shape[:2]
        scale = self.display_width / w
        return cv2.resize(frame, (self.display_width, int(h * scale)))

    def _draw_auth(self, frame: np.ndarray, deadline: float) -> np.ndarray:
        disp       = self._resize(frame)
        dh, dw     = disp.shape[:2]
        remaining  = max(0.0, deadline - time.time())
        cv2.rectangle(disp, (0, 0), (dw, 52), (18, 18, 18), -1)
        cv2.putText(disp, f"AUTH  Scanning for driver…  ({remaining:.0f}s)",
                    (10, 37), _FONT, 0.75, (60, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "Q — quit", (8, dh - 8),
                    _FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        return disp

    def _draw_monitor(self, frame: np.ndarray, name: str,
                      ear: Optional[float], is_drowsy: bool) -> np.ndarray:
        disp   = self._resize(frame)
        dh, dw = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (dw, 52), (18, 18, 18), -1)
        cv2.putText(disp, f"MONITOR  |  {name}",
                    (10, 37), _FONT, 0.80, (80, 220, 80), 2, cv2.LINE_AA)
        if ear is not None:
            col = (80, 220, 80) if ear >= self.ear_thresh else (40, 60, 240)
            cv2.putText(disp, f"EAR {ear:.3f}", (dw - 145, 37),
                        _FONT, 0.75, col, 2, cv2.LINE_AA)
        if is_drowsy:
            cv2.rectangle(disp, (0, 0), (dw - 1, dh - 1), (30, 30, 220), 10)
            bh = 48
            cv2.rectangle(disp, (0, dh - bh), (dw, dh), (20, 20, 180), -1)
            cv2.putText(disp, "!  DROWSINESS ALERT  !",
                        (dw // 2 - 195, dh - 13),
                        _FONT, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, "Q — quit", (8, dh - bh - 8),
                        _FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        else:
            cv2.putText(disp, "Q — quit", (8, dh - 8),
                        _FONT, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        return disp

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        sdv_db.init_db(self.db_path)

        cam_src = self.cfg["camera"]["source"]

        def _handle_stop(sig, _frame):
            log.info("Signal %d received — shutting down.", sig)
            self._running = False

        signal.signal(signal.SIGINT,  _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        cap = cv2.VideoCapture(cam_src)
        if not cap.isOpened():
            log.error("Cannot open camera source: %s", cam_src)
            sys.exit(1)
        log.info("Camera opened (source=%s). Display=%s.", cam_src, self.display_enabled)

        if self.display_enabled:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW, self.display_width, int(self.display_width * 0.75))

        try:
            while self._running:

                # ── Phase 1: AUTH ──────────────────────────────────────────
                result = self._run_auth(cap)
                if result is None:
                    break   # Q pressed or signal received

                uid, name = result

                # ── Phase 2: CHAT ──────────────────────────────────────────
                # Release camera before launching chatbot so the Pi camera
                # bus is free (only one consumer at a time on most Pi configs).
                cap.release()
                if self.display_enabled:
                    cv2.destroyAllWindows()

                self._run_chat(uid, name)

                # ── Phase 3: MONITOR ───────────────────────────────────────
                cap = cv2.VideoCapture(cam_src)
                if not cap.isOpened():
                    log.error("Cannot reopen camera for MONITOR phase.")
                    break

                if self.display_enabled:
                    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(_WINDOW, self.display_width,
                                     int(self.display_width * 0.75))

                self._run_monitor(cap, name)
                # Loop back to AUTH for the next driver / next session

        finally:
            cap.release()
            if self.display_enabled:
                cv2.destroyAllWindows()
            self.pub.close()
            log.info("S-Chassis Demo stopped cleanly.")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="S-Chassis Demo Interface")
    parser.add_argument(
        "--config",
        default=os.path.join(_MONITOR_DIR, "config.json"),
        help="Path to config.json (default: sdv-driver-monitor/config.json)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Headless mode — disables cv2 windows. "
             "Ideal for Pi without a monitor attached.",
    )
    args = parser.parse_args()

    config  = _load_config(args.config)
    display = config.get("display", {}).get("enabled", True)
    if args.no_display:
        display = False

    DemoOrchestrator(config, display).run()


if __name__ == "__main__":
    main()
