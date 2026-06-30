"""
Face inference engine for Raspberry Pi.

Uses OpenCV's built-in YuNet detector and SFace recogniser (both ONNX,
no extra install beyond opencv-python >=4.8).  MediaPipe FaceLandmarker
(Tasks API, 0.10+) is used for the 468-landmark EAR calculation.

Model files (download via setup_pi.sh):
  models/face_detection_yunet_2023mar.onnx   ~228 KB
  models/face_recognition_sface_2021dec.onnx  ~37 MB
  models/face_landmarker.task                 ~3.6 MB
"""

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# MediaPipe 468-point eye indices for EAR (same formula as dlib, different coords)
_L_EYE = [33, 160, 158, 133, 153, 144]
_R_EYE = [362, 385, 387, 263, 373, 380]


class FaceEngine:
    """Wraps face detection (YuNet) + recognition (SFace) + landmark EAR (MediaPipe)."""

    def __init__(self, det_model: str, rec_model: str, landmark_model: str = "") -> None:
        self._detector = cv2.FaceDetectorYN.create(
            det_model, "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=1,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(rec_model, "")

        self._landmarker = None
        self._mp_ok = False
        if landmark_model:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks import python as mp_python

                options = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=landmark_model),
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                self._mp = mp
                self._mp_ok = True
                log.info("MediaPipe FaceLandmarker loaded.")
            except Exception as exc:
                log.warning("MediaPipe FaceLandmarker failed to load (%s) — EAR disabled", exc)

    # ── face embedding ────────────────────────────────────────────────────────

    def embed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return 128-dim SFace embedding for the best face in frame, or None."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        best = faces[int(np.argmax(faces[:, 14]))]
        aligned = self._recognizer.alignCrop(frame, best)
        feat = self._recognizer.feature(aligned)
        return feat.flatten().astype(np.float32)

    # ── Eye Aspect Ratio ──────────────────────────────────────────────────────

    def ear(self, frame: np.ndarray) -> Optional[float]:
        """Return mean EAR for both eyes via MediaPipe Tasks API, or None."""
        if not self._mp_ok:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        h, w = frame.shape[:2]

        def pt(idx: int) -> np.ndarray:
            p = lm[idx]
            return np.array([p.x * w, p.y * h])

        def _eye_ear(indices):
            p = [pt(i) for i in indices]
            A = np.linalg.norm(p[1] - p[5])
            B = np.linalg.norm(p[2] - p[4])
            C = np.linalg.norm(p[0] - p[3])
            return (A + B) / (2.0 * C + 1e-6)

        return (_eye_ear(_L_EYE) + _eye_ear(_R_EYE)) / 2.0

    # ── distance util ─────────────────────────────────────────────────────────

    @staticmethod
    def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
