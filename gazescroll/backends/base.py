"""Swappable landmark backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceResult:
    landmarks: np.ndarray                 # (478, 3) normalized x, y in [0,1]; z relative
    image_size: tuple[int, int]           # (width, height)
    head_pose: tuple[float, float, float] | None = None  # (pitch, yaw, roll) deg, if the backend provides it


class LandmarkBackend(ABC):
    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> FaceResult | None:
        """Run landmark inference on one BGR frame. None if no face."""

    def close(self) -> None:
        pass


def create_backend(name: str, cfg: dict) -> LandmarkBackend:
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend
        from ..config import resolve
        return MediaPipeBackend(resolve(cfg["model_path"]))
    if name == "hailo":
        from .hailo_backend import HailoBackend
        return HailoBackend(cfg)
    raise ValueError(f"Unknown backend: {name!r}")
