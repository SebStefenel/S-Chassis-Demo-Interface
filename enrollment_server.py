#!/usr/bin/env python3
"""
SDV Headless Enrollment Server.

Exposes a minimal web UI at http://<pi-ip>:5000 so a driver can be enrolled
from any phone or laptop on the same network — no keyboard or display required.

API:
  GET  /           — enrollment page (HTML)
  POST /enroll     — start enrollment  body: {"name": "Alice"}
  GET  /status     — poll enrollment progress  response: JSON state
  GET  /list       — list all enrolled drivers
  DELETE /delete/<name> — remove a driver profile

Usage:
  python enrollment_server.py
"""

import json
import logging
import os
import threading
import time

import cv2
from flask import Flask, jsonify, render_template_string, request

import sdv_db
from inference import FaceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sdv.enroll")

_HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

_state: dict = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "Ready",
    "done": False,
    "error": False,
}
_lock = threading.Lock()

# ── minimal dark-theme web UI ─────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SDV Enrollment</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
    .card { max-width: 480px; margin: 48px auto; padding: 24px; background: #1a1a1a;
            border-radius: 12px; box-shadow: 0 4px 24px #0006; }
    h1 { margin: 0 0 24px; color: #0cf; font-size: 1.5rem; }
    label { display: block; margin-bottom: 6px; font-size: .9rem; color: #aaa; }
    input[type=text] { width: 100%; padding: 12px; font-size: 1rem; border: 1px solid #333;
                       border-radius: 8px; background: #222; color: #eee; outline: none; }
    input[type=text]:focus { border-color: #0cf; }
    button { width: 100%; padding: 14px; margin-top: 12px; font-size: 1rem; font-weight: 700;
             border: none; border-radius: 8px; cursor: pointer; transition: opacity .2s; }
    #enroll-btn { background: #0cf; color: #000; }
    #enroll-btn:disabled { opacity: .4; cursor: default; }
    .progress-wrap { margin-top: 20px; }
    .bar-bg { background: #333; border-radius: 6px; height: 10px; overflow: hidden; }
    .bar-fill { background: #0cf; height: 100%; border-radius: 6px; width: 0;
                transition: width .4s ease; }
    #msg { margin-top: 10px; min-height: 24px; font-size: .9rem; }
    .ok   { color: #4f4; }
    .err  { color: #f66; }
    hr { border: none; border-top: 1px solid #333; margin: 28px 0; }
    h2 { font-size: 1rem; color: #aaa; margin: 0 0 12px; }
    #driver-list { list-style: none; padding: 0; margin: 0; }
    #driver-list li { display: flex; justify-content: space-between; align-items: center;
                      padding: 8px 12px; background: #222; border-radius: 6px; margin-bottom: 8px; }
    #driver-list li span { font-size: .85rem; color: #888; }
    #driver-list li button { width: auto; padding: 6px 14px; font-size: .8rem; font-weight: 600;
                             background: #f44; color: #fff; margin-top: 0; border-radius: 6px; }
  </style>
</head>
<body>
<div class="card">
  <h1>SDV Driver Enrollment</h1>
  <label for="name">Driver name</label>
  <input type="text" id="name" placeholder="e.g. Sebastian" autocomplete="off">
  <button id="enroll-btn" onclick="startEnroll()">Enroll</button>
  <div class="progress-wrap">
    <div class="bar-bg"><div class="bar-fill" id="bar"></div></div>
    <div id="msg"></div>
  </div>

  <hr>
  <h2>Enrolled drivers</h2>
  <ul id="driver-list"><li><span>Loading…</span></li></ul>
</div>

<script>
  let poll = null;

  function setMsg(text, cls) {
    const el = document.getElementById('msg');
    el.textContent = text;
    el.className = cls || '';
  }

  function setBar(pct) {
    document.getElementById('bar').style.width = Math.min(pct, 100) + '%';
  }

  function startEnroll() {
    const name = document.getElementById('name').value.trim();
    if (!name) { alert('Enter a driver name first.'); return; }
    document.getElementById('enroll-btn').disabled = true;
    setMsg('Starting…'); setBar(0);

    fetch('/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        setMsg(data.error, 'err');
        document.getElementById('enroll-btn').disabled = false;
        return;
      }
      poll = setInterval(checkStatus, 600);
    })
    .catch(e => { setMsg('Network error: ' + e, 'err'); document.getElementById('enroll-btn').disabled = false; });
  }

  function checkStatus() {
    fetch('/status').then(r => r.json()).then(data => {
      setMsg(data.message, data.error ? 'err' : data.done ? 'ok' : '');
      const pct = data.total > 0 ? (data.progress / data.total * 100) : 0;
      setBar(pct);
      if (data.done || data.error) {
        clearInterval(poll);
        document.getElementById('enroll-btn').disabled = false;
        loadDrivers();
      }
    });
  }

  function loadDrivers() {
    fetch('/list').then(r => r.json()).then(drivers => {
      const ul = document.getElementById('driver-list');
      if (!drivers.length) {
        ul.innerHTML = '<li><span>No drivers enrolled yet.</span></li>';
        return;
      }
      ul.innerHTML = drivers.map(d =>
        `<li><div><strong>${d.name}</strong><br><span>${d.samples} samples &middot; ${d.enrolled_at}</span></div>
         <button onclick="deleteDriver('${d.name}')">Delete</button></li>`
      ).join('');
    });
  }

  function deleteDriver(name) {
    if (!confirm('Delete profile for ' + name + '?')) return;
    fetch('/delete/' + encodeURIComponent(name), { method: 'DELETE' })
      .then(() => loadDrivers());
  }

  loadDrivers();
</script>
</body>
</html>"""


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── enrollment worker (background thread) ────────────────────────────────────

def _enroll_worker(name: str, config: dict) -> None:
    with _lock:
        _state.update(running=True, progress=0, done=False, error=False,
                      message=f"Starting enrollment for '{name}'…")

    db_path = _resolve_db(config)
    enroll_cfg = config["enrollment"]
    models = config["models"]
    num_frames = enroll_cfg["num_sample_frames"]
    interval = enroll_cfg["capture_interval_frames"]
    cam_src = config["camera"]["source"]

    with _lock:
        _state["total"] = num_frames

    engine = FaceEngine(
        os.path.join(_HERE, models["det_model"]),
        os.path.join(_HERE, models["rec_model"]),
    )

    existing = sdv_db.get_user_by_name(db_path, name)
    if existing:
        sdv_db.delete_user(db_path, existing["id"])
        log.info("Removed existing profile for '%s'.", name)

    defaults = config.get("vehicle_settings_defaults", {})
    uid = sdv_db.add_user(db_path, name, defaults)
    log.info("Created profile for '%s' (id=%d).", name, uid)

    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        with _lock:
            _state.update(running=False, error=True, done=True, message="ERROR: Cannot open camera.")
        return

    captured = 0
    frame_n = 0
    no_face = 0

    while captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % interval != 0:
            time.sleep(0.01)
            continue

        emb = engine.embed(frame)
        if emb is not None:
            sdv_db.add_embedding(db_path, uid, emb)
            captured += 1
            no_face = 0
            log.info("Sample %d/%d captured.", captured, num_frames)
            with _lock:
                _state.update(progress=captured, message=f"Sample {captured}/{num_frames} captured.")
        else:
            no_face += 1
            with _lock:
                _state["message"] = f"No face detected — look directly at the camera. ({no_face})"
            if no_face > 80:
                break

    cap.release()

    if captured < 1:
        sdv_db.delete_user(db_path, uid)
        with _lock:
            _state.update(running=False, error=True, done=True,
                          message="No samples captured — profile removed. Check camera and lighting.")
        return

    if captured < num_frames:
        msg = f"Partial enrolment: {captured}/{num_frames} samples. Accuracy may vary."
    else:
        msg = f"Done! '{name}' enrolled with {captured} samples."

    log.info(msg)
    with _lock:
        _state.update(running=False, done=True, error=False, message=msg)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/enroll", methods=["POST"])
def start_enroll():
    with _lock:
        if _state["running"]:
            return jsonify({"error": "Enrollment already in progress."}), 409
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Driver name is required."}), 400
    config = _load_config()
    threading.Thread(target=_enroll_worker, args=(name, config), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with _lock:
        return jsonify(_state.copy())


@app.route("/list")
def list_drivers():
    config = _load_config()
    db_path = _resolve_db(config)
    sdv_db.init_db(db_path)
    users = sdv_db.list_users(db_path)
    result = []
    for u in users:
        count = sdv_db.embedding_count(db_path, u["id"])
        result.append({
            "id": u["id"],
            "name": u["username"],
            "enrolled_at": u["created_at"],
            "samples": count,
        })
    return jsonify(result)


@app.route("/delete/<name>", methods=["DELETE"])
def delete_driver(name: str):
    config = _load_config()
    db_path = _resolve_db(config)
    user = sdv_db.get_user_by_name(db_path, name)
    if not user:
        return jsonify({"error": f"'{name}' not found."}), 404
    sdv_db.delete_user(db_path, user["id"])
    return jsonify({"ok": True, "deleted": name})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = _load_config()
    db_path = _resolve_db(config)
    sdv_db.init_db(db_path)

    import socket
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "0.0.0.0"

    log.info("Enrollment server starting at http://%s:5000", ip)
    log.info("Open the URL above on any device connected to the same network.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
