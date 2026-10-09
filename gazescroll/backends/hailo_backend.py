"""Hailo-10H (ASUS UGen300) backend — Stage II stub.

Plan for Stage II:
1. Model: export the face detector + face/iris landmark model to ONNX, then
   compile to a HEF with the Hailo Dataflow Compiler (parse -> optimize with a
   calibration set of face crops -> compile). Alternatively, start from a
   matching face-landmark model in the Hailo Model Zoo.
2. Runtime: load the HEF with HailoRT (`hailo_platform` Python API), create a
   VDevice, configure the network group, and run inference through
   InferVStreams / the async infer API.
3. Pre/post-processing (on host): face detection -> crop/resize to the model's
   input size -> run landmarks -> map back to full-frame normalized coords so
   `process()` returns the same 478-point FaceResult as the MediaPipe backend.
   Nothing downstream (gaze, gestures, actions) needs to change.
4. Benchmark with `python -m gazescroll.main --benchmark` and compare FPS,
   latency and CPU load against the laptop MediaPipe baseline.
"""
from __future__ import annotations

from .base import FaceResult, LandmarkBackend


class HailoBackend(LandmarkBackend):
    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "Hailo backend is planned for Stage II (UGen300 / Hailo-10H). "
            "Set `backend: mediapipe` in config.yaml."
        )

    def process(self, frame_bgr) -> FaceResult | None:  # pragma: no cover
        raise NotImplementedError
