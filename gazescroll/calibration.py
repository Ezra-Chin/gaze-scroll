"""Per-user calibration, centered on the user's resting gaze.

The center of the screen is the baseline (where people look by default).
Scores below center are UP (looking toward the camera), above are DOWN.
The top/bottom dots only set how far from center each threshold sits.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .gaze import ZoneClassifier, eye_aspect_ratio, gaze_score, head_pitch, vertical_iris_offset

WINDOW = "GazeScroll calibration"
SETTLE_S = 0.7


@dataclass
class Calibration:
    center: float
    top: float
    bottom: float
    center_std: float
    up_thresh: float
    down_thresh: float
    screen_size: tuple[int, int]
    timestamp: float
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_scores(cls, center: float, top: float, bottom: float, center_std: float,
                    screen_size: tuple[int, int], cal_cfg: dict) -> "Calibration":
        """Thresholds = center -/+ a fraction of the way to the top/bottom dot,
        but never inside the center's own noise band."""
        noise = cal_cfg["noise_k"] * center_std
        fallback = cal_cfg["fallback_margin"]
        warnings = []

        up_span = center - top
        if up_span <= noise:
            warnings.append("Looking UP barely changed the reading; using a default margin.")
            up_margin = max(fallback, noise)
        else:
            up_margin = max(cal_cfg["up_fraction"] * up_span, noise)

        down_span = bottom - center
        if down_span <= noise:
            warnings.append("Looking DOWN barely changed the reading; using a default margin.")
            down_margin = max(fallback, noise)
        else:
            down_margin = max(cal_cfg["down_fraction"] * down_span, noise)

        return cls(
            center=center, top=top, bottom=bottom, center_std=center_std,
            up_thresh=center - up_margin, down_thresh=center + down_margin,
            screen_size=tuple(screen_size), timestamp=time.time(), warnings=tuple(warnings),
        )

    def recentered(self, score: float) -> "Calibration":
        """Shift everything so `score` becomes the new center (user moved/settled)."""
        d = score - self.center
        return replace(self, center=score, top=self.top + d, bottom=self.bottom + d,
                       up_thresh=self.up_thresh + d, down_thresh=self.down_thresh + d,
                       timestamp=time.time())

    def classifier(self, hysteresis_ms: float) -> ZoneClassifier:
        # Overshoot reaches 1.0 when gaze is at the calibration dot.
        return ZoneClassifier(
            self.up_thresh, self.down_thresh, hysteresis_ms,
            up_scale=self.up_thresh - self.top,
            down_scale=self.bottom - self.down_thresh,
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration | None":
        """None if missing or from an older format (forces recalibration)."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            d["screen_size"] = tuple(d["screen_size"])
            d["warnings"] = tuple(d.get("warnings", ()))
            return cls(**d)
        except (ValueError, KeyError, TypeError):
            return None


def _draw(screen_w: int, screen_h: int, pos: tuple[int, int], label: str,
          progress: float, status: str) -> np.ndarray:
    img = np.full((screen_h, screen_w, 3), 30, np.uint8)
    cv2.circle(img, pos, 22, (255, 255, 255), -1)
    cv2.circle(img, pos, 8, (0, 0, 220), -1)
    if progress > 0:
        cv2.ellipse(img, pos, (34, 34), -90, 0, int(360 * progress), (0, 200, 0), 4)
    tx, ty = 40, screen_h // 2 + (-140 if pos[1] > screen_h // 2 else 140)
    if pos[1] == screen_h // 2:
        ty = screen_h // 2 + 100
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, status, (tx, ty + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (160, 160, 160), 1, cv2.LINE_AA)
    return img


def score_of(cfg: dict):
    """Measure function for the scroll calibration: the 1-D vertical gaze score."""
    def measure(face):
        lm = face.landmarks
        w, h = face.image_size
        return gaze_score(vertical_iris_offset(lm), head_pitch(lm, w, h), cfg)
    return measure


def open_fullscreen() -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def close_fullscreen() -> None:
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)


def sample_target(camera, backend, cfg, screen_size, pos, label, duration, measure,
                  settle_s: float = SETTLE_S) -> list | None:
    """Show a dot and collect `measure(face)` values while the user looks at it.
    Eyes-closed frames are skipped. None if the user aborts with Esc/Q."""
    sw, sh = screen_size
    ear_thresh = cfg["eyes"]["ear_closed_thresh"]
    samples: list = []
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        frame, _ = camera.read()
        status = "No face detected - stay in view of the camera"
        if frame is not None:
            face = backend.process(frame)
            if face is not None:
                if eye_aspect_ratio(face.landmarks) < ear_thresh:
                    status = "Eyes closed"
                else:
                    status = "Keep your head still, move only your eyes"
                    if elapsed >= settle_s:
                        value = measure(face)
                        if value is not None:
                            samples.append(value)
        progress = max(0.0, (elapsed - settle_s) / duration)
        cv2.imshow(WINDOW, _draw(sw, sh, pos, label, min(progress, 1.0), status))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return None
        if elapsed >= settle_s + duration and len(samples) >= 10:
            return samples


def run_calibration(camera, backend, cfg: dict, screen_size: tuple[int, int]) -> Calibration | None:
    """Interactive calibration: center (baseline), then top, then bottom.
    Returns None if the user aborts with Esc/Q."""
    sw, sh = screen_size
    cal = cfg["calibration"]
    steps = [
        ("Look at the CENTER dot, as you normally would", (sw // 2, sh // 2), cal["center_s"]),
        ("Look UP at the dot (toward the camera)", (sw // 2, 30), cal["edge_s"]),
        ("Look DOWN at the dot", (sw // 2, sh - 30), cal["edge_s"]),
    ]

    measure = score_of(cfg)
    open_fullscreen()
    results = []
    try:
        for label, pos, duration in steps:
            samples = sample_target(camera, backend, cfg, screen_size, pos, label, duration, measure)
            if samples is None:
                return None
            results.append(samples)
    finally:
        close_fullscreen()

    center_s, top_s, bottom_s = (np.asarray(r) for r in results)
    center = float(np.median(center_s))
    # Robust std via MAD so a stray blink frame doesn't inflate the noise band.
    center_std = float(1.4826 * np.median(np.abs(center_s - center)))
    return Calibration.from_scores(center, float(np.median(top_s)), float(np.median(bottom_s)),
                                   center_std, screen_size, cal)
