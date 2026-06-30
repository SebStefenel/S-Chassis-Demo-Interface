#!/usr/bin/env python3
"""
SDV Driver Enrollment — CLI tool with live camera window.

Use this when a display is available (connected HDMI or SSH with X11 forwarding).
For headless Pi enrollment, use enrollment_server.py instead.

Usage:
  python enroll_driver.py -u Alice
  python enroll_driver.py --list
  python enroll_driver.py --delete Alice
"""

import argparse
import json
import os
import sys
import time

import cv2

import sdv_db
from inference import FaceEngine

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _load_config() -> dict:
    with open(os.path.join(_HERE, "config.json")) as f:
        return json.load(f)


def _resolve_db(config: dict) -> str:
    raw = config["database"]["path"]
    path = os.path.expanduser(raw)
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def enroll(username: str, config: dict) -> bool:
    db_path = _resolve_db(config)
    enroll_cfg = config["enrollment"]
    disp_cfg = config.get("display", {})
    num_frames = enroll_cfg["num_sample_frames"]
    interval = enroll_cfg["capture_interval_frames"]
    cam_src = config["camera"]["source"]
    disp_w = disp_cfg.get("width", 640)

    engine = FaceEngine(
        os.path.join(_HERE, config["models"]["det_model"]),
        os.path.join(_HERE, config["models"]["rec_model"]),
        os.path.join(_HERE, config["models"].get("landmark_model", "")),
    )

    existing = sdv_db.get_user_by_name(db_path, username)
    if existing:
        ans = input(f"'{username}' already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return False
        sdv_db.delete_user(db_path, existing["id"])
        print(f"Removed old profile for '{username}'.")

    defaults = config.get("vehicle_settings_defaults", {})
    uid = sdv_db.add_user(db_path, username, defaults)
    print(f"Created profile for '{username}' (id={uid}).")

    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sdv_db.delete_user(db_path, uid)
        return False

    print(f"Look at the camera. Capturing {num_frames} samples (every {interval} frames)…")
    print("Press Q to abort.")

    captured = 0
    frame_n = 0
    no_face = 0

    while captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        # ── overlay ───────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        scale = disp_w / w
        disp = cv2.resize(frame, (disp_w, int(h * scale)))
        dh, dw = disp.shape[:2]

        cv2.rectangle(disp, (0, 0), (dw, 52), (18, 18, 18), -1)
        bar_color = (80, 220, 80) if no_face < 5 else (60, 100, 220)
        cv2.putText(disp, f"ENROLLING: {username}  {captured}/{num_frames}",
                    (10, 37), _FONT, 0.82, bar_color, 2, cv2.LINE_AA)

        hint = "Face the camera and hold still."
        if no_face >= 5:
            hint = "No face — check lighting or move closer."
        cv2.putText(disp, hint, (10, dh - 10), _FONT, 0.46, (140, 140, 140), 1, cv2.LINE_AA)

        # Progress bar
        bar_x = int(dw * captured / num_frames)
        cv2.rectangle(disp, (0, 52), (bar_x, 58), (80, 220, 80), -1)

        cv2.imshow("SDV Enrollment", disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("Aborted by user.")
            break

        # ── capture ───────────────────────────────────────────────────────────
        if frame_n % interval != 0:
            continue

        emb = engine.embed(frame)
        if emb is not None:
            sdv_db.add_embedding(db_path, uid, emb)
            captured += 1
            no_face = 0
            print(f"  Sample {captured}/{num_frames}")
        else:
            no_face += 1

    cap.release()
    cv2.destroyAllWindows()

    if captured < 1:
        print("No samples captured — profile removed.")
        sdv_db.delete_user(db_path, uid)
        return False

    if captured < num_frames:
        print(f"Partial enrollment: {captured}/{num_frames} samples. Accuracy may vary.")
    else:
        print(f"Enrollment complete for '{username}'.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="SDV Driver Enrollment (display mode)")
    parser.add_argument("--username", "-u", help="Driver name to enroll")
    parser.add_argument("--list", "-l", action="store_true", help="List enrolled drivers")
    parser.add_argument("--delete", "-d", metavar="NAME", help="Delete a driver profile")
    args = parser.parse_args()

    config = _load_config()
    db_path = _resolve_db(config)
    sdv_db.init_db(db_path)

    if args.list:
        users = sdv_db.list_users(db_path)
        if not users:
            print("No enrolled drivers.")
        else:
            print(f"\n{'ID':<6} {'Username':<24} {'Enrolled At'}")
            print("─" * 60)
            for u in users:
                count = sdv_db.embedding_count(db_path, u["id"])
                print(f"{u['id']:<6} {u['username']:<24} {u['created_at']}  ({count} samples)")
        return

    if args.delete:
        user = sdv_db.get_user_by_name(db_path, args.delete)
        if not user:
            print(f"'{args.delete}' not found.")
        else:
            sdv_db.delete_user(db_path, user["id"])
            print(f"Deleted '{args.delete}'.")
        return

    username = args.username or input("Driver name: ").strip()
    if not username:
        print("No name provided.")
        return

    enroll(username, config)


if __name__ == "__main__":
    main()
