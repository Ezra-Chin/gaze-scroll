"""MediaPipe Tasks FaceLandmarker backend (478 landmarks incl. irises)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .base import FaceResult, LandmarkBackend


class MediaPipeBackend(LandmarkBackend):
    def __init__(self, model_path: Path):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python scripts/download_models.py"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> FaceResult | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        lm = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        return FaceResult(landmarks=lm, image_size=(w, h))

    def close(self) -> None:
        self._landmarker.close()
