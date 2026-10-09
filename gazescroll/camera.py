"""Webcam capture."""
from __future__ import annotations

import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On macOS, grant your terminal "
                "Camera access in System Settings > Privacy & Security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[np.ndarray | None, float]:
        """Return (frame_bgr, capture_timestamp). Frame is None on failure."""
        ok, frame = self.cap.read()
        t = time.perf_counter()
        return (frame if ok else None), t

    def release(self) -> None:
        self.cap.release()
