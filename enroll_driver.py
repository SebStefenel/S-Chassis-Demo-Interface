#!/usr/bin/env python3
"""
SDV Driver Enrollment Utility
-------------------------------
Registers a new driver by capturing N face samples from the webcam,
extracting Facenet embeddings via deepface, and saving them to the local
SQLite user database.

Usage
-----
  python enroll_driver.py                      # interactive prompt for name
  python enroll_driver.py -u Alice             # enroll 'Alice'
  python enroll_driver.py --list               # list enrolled drivers
  python enroll_driver.py --delete Alice       # remove Alice's profile
"""

import argparse
import json
import os
import sys

# Resolve paths relative to this file so the script works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "deepface"))

import cv2
import numpy as np

import sdv_db

# deepface import is deferred to after sys.path is set
from deepface import DeepFace  # noqa: E402


# ── config ────────────────────────────────────────────────────────────────────

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


# ── embedding extraction ──────────────────────────────────────────────────────

def _extract_embedding(
    frame: np.ndarray, model_name: str, detector_backend: str
) -> "np.ndarray | None":
    """Return the first face embedding from a BGR frame, or None on failure."""
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
        )
        return np.array(results[0]["embedding"], dtype=np.float32)
    except Exception as e:
        _last_err = str(e)
        if "Face could not be detected" not in _last_err and "No face" not in _last_err:
            print(f"  [DEBUG] DeepFace error: {_last_err}")
        return None


# ── enrollment flow ───────────────────────────────────────────────────────────

def enroll(username: str, config: dict) -> bool:
    """
    Open the camera, capture num_sample_frames embeddings for *username*,
    and persist them in the SQLite database.  Returns True on success.
    """
    db_path = _resolve_db_path(config)
    fr = config["face_recognition"]
    enroll_cfg = config["enrollment"]
    cam_source = config["camera"]["source"]
    display_width = config["camera"]["display_width"]
    num_frames = enroll_cfg["num_sample_frames"]
    capture_interval = enroll_cfg["capture_interval_frames"]

    # ── handle existing profile ───────────────────────────────────────────────
    existing = sdv_db.get_user_by_name(db_path, username)
    if existing:
        answer = input(
            f"[ENROLL] '{username}' already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            print("[ENROLL] Aborted.")
            return False
        sdv_db.delete_user(db_path, existing["id"])
        print(f"[ENROLL] Removed old profile for '{username}'.")

    default_settings = config.get("vehicle_settings_defaults", {})
    user_id = sdv_db.add_user(db_path, username, default_settings)
    print(f"[ENROLL] Created profile for '{username}' (id={user_id}).")

    # ── open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print("[ENROLL] ERROR: Cannot open camera.")
        sdv_db.delete_user(db_path, user_id)
        return False

    samples_captured = 0
    frame_counter = 0
    no_face_streak = 0
    print(
        f"[ENROLL] Look at the camera. "
        f"Capturing {num_frames} samples (every {capture_interval} frames)…"
    )

    while samples_captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("[ENROLL] Camera read failed.")
            break

        # ── display ───────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (display_width, int(h * display_width / w)))

        bar_color = (0, 180, 0) if no_face_streak < 5 else (0, 100, 255)
        cv2.rectangle(display, (0, 0), (display.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"ENROLLING: {username}  |  {samples_captured}/{num_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, bar_color, 2,
        )
        hint = "Face the camera and stay still."
        if no_face_streak >= 5:
            hint = "No face detected — adjust lighting / position."
        cv2.putText(display, hint, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("SDV Enrollment", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[ENROLL] Interrupted by user.")
            break

        # ── sample capture ────────────────────────────────────────────────────
        frame_counter += 1
        if frame_counter % capture_interval != 0:
            continue

        embedding = _extract_embedding(frame, fr["model_name"], fr["detector_backend"])
        if embedding is not None:
            sdv_db.add_embedding(db_path, user_id, embedding)
            samples_captured += 1
            no_face_streak = 0
            print(f"  Sample {samples_captured}/{num_frames} ✓")
        else:
            no_face_streak += 1
            print(f"  No face detected (attempt {no_face_streak}).")

    cap.release()
    cv2.destroyAllWindows()

    if samples_captured < 1:
        print("[ENROLL] No samples captured — removing incomplete profile.")
        sdv_db.delete_user(db_path, user_id)
        return False

    if samples_captured < num_frames:
        print(
            f"[ENROLL] Partial enrollment: {samples_captured}/{num_frames} samples. "
            "Recognition accuracy may be reduced."
        )
    else:
        print(f"[ENROLL] Enrollment complete for '{username}'.")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SDV Driver Enrollment Tool")
    parser.add_argument("--username", "-u", type=str, help="Driver name to enroll")
    parser.add_argument("--list", "-l", action="store_true", help="List enrolled drivers")
    parser.add_argument("--delete", "-d", type=str, metavar="NAME",
                        help="Delete a driver profile by username")
    args = parser.parse_args()

    config = _load_config()
    db_path = _resolve_db_path(config)
    sdv_db.init_db(db_path)

    if args.list:
        users = sdv_db.list_users(db_path)
        if not users:
            print("No enrolled drivers.")
        else:
            print(f"\n{'ID':<6} {'Username':<24} {'Enrolled At'}")
            print("─" * 55)
            for u in users:
                count = sdv_db.embedding_count(db_path, u["id"])
                print(f"{u['id']:<6} {u['username']:<24} {u['created_at']}  ({count} embeddings)")
        return

    if args.delete:
        user = sdv_db.get_user_by_name(db_path, args.delete)
        if not user:
            print(f"User '{args.delete}' not found.")
        else:
            sdv_db.delete_user(db_path, user["id"])
            print(f"Deleted profile for '{args.delete}'.")
        return

    username = args.username or input("Enter driver name to enroll: ").strip()
    if not username:
        print("No username provided. Aborting.")
        return

    enroll(username, config)


if __name__ == "__main__":
    main()
