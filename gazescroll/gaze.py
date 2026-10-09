"""Gaze estimation: iris ratio, head pitch, smoothing, zone classification.

All functions take the (478, 3) normalized landmark array from a FaceResult
and are pure, so they can be unit-tested with synthetic data.
"""
from __future__ import annotations

from collections import deque
from enum import Enum

import cv2
import numpy as np

# MediaPipe face mesh indices (subject's right eye appears on the image left).
R_CORNERS = (33, 133)
R_UPPER, R_LOWER, R_IRIS = 159, 145, 468
L_CORNERS = (362, 263)
L_UPPER, L_LOWER, L_IRIS = 386, 374, 473

# solvePnP points: nose tip, chin, outer eye corners, mouth corners.
PNP_IDX = (1, 152, 33, 263, 61, 291)
# Generic 3D face (mm) in camera convention: x right, y down, z away from
# the camera. A frontal face therefore solves to ~identity rotation.
PNP_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, 330.0, 65.0),
        (-225.0, -170.0, 135.0),
        (225.0, -170.0, 135.0),
        (-150.0, 150.0, 125.0),
        (150.0, 150.0, 125.0),
    ],
    dtype=np.float64,
)


class Zone(str, Enum):
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"


def _eye_ratio(lm: np.ndarray, upper: int, lower: int, iris: int) -> float:
    span = lm[lower, 1] - lm[upper, 1]
    if abs(span) < 1e-6:
        return 0.5
    return float((lm[iris, 1] - lm[upper, 1]) / span)


def vertical_iris_ratio(lm: np.ndarray) -> float:
    """0 = iris at upper lid, 1 = iris at lower lid. Averaged over both eyes."""
    r = _eye_ratio(lm, R_UPPER, R_LOWER, R_IRIS)
    l = _eye_ratio(lm, L_UPPER, L_LOWER, L_IRIS)
    return (r + l) / 2.0


def vertical_iris_offset(lm: np.ndarray) -> float:
    """Iris height relative to the eye-corner line, divided by eye width.

    Positive = looking down, negative = looking up. Eye corners stay put when the
    eyes move, unlike the lids (which follow the iris), so this works much better
    than the lid ratio, especially for looking up. Averaged over both eyes.
    """
    def offset(c1, c2, iris):
        width = np.linalg.norm(lm[c1, :2] - lm[c2, :2])
        if width < 1e-6:
            return 0.0
        mid_y = (lm[c1, 1] + lm[c2, 1]) / 2
        return float((lm[iris, 1] - mid_y) / width)

    return (offset(*R_CORNERS, R_IRIS) + offset(*L_CORNERS, L_IRIS)) / 2.0


def horizontal_iris_offset(lm: np.ndarray) -> float:
    """Iris x relative to the eye-corner midpoint, divided by eye width.

    In (unmirrored) image coordinates: positive = iris toward the image right,
    which is the user's left. Averaged over both eyes.
    """
    def offset(c1, c2, iris):
        width = np.linalg.norm(lm[c1, :2] - lm[c2, :2])
        if width < 1e-6:
            return 0.0
        mid_x = (lm[c1, 0] + lm[c2, 0]) / 2
        return float((lm[iris, 0] - mid_x) / width)

    return (offset(*R_CORNERS, R_IRIS) + offset(*L_CORNERS, L_IRIS)) / 2.0


def eye_aspect_ratio(lm: np.ndarray) -> float:
    """Lid opening / eye width, averaged over both eyes. ~0.25-0.35 open, <0.15 closed."""
    def ear(c1, c2, up, lo):
        width = np.linalg.norm(lm[c1, :2] - lm[c2, :2])
        if width < 1e-6:
            return 0.0
        return float(np.linalg.norm(lm[up, :2] - lm[lo, :2]) / width)

    return (ear(*R_CORNERS, R_UPPER, R_LOWER) + ear(*L_CORNERS, L_UPPER, L_LOWER)) / 2.0


def _wrap(a: float) -> float:
    """Keep an angle near zero for a roughly frontal face (guard against 180° flips)."""
    if a > 90:
        return a - 180
    if a < -90:
        return a + 180
    return a


def head_pose(lm: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    """(pitch, yaw, roll) in degrees. Pitch positive = head tilted down; yaw
    positive = nose toward the image left (user turning to their right).
    None if PnP fails."""
    pts2d = np.array([(lm[i, 0] * width, lm[i, 1] * height) for i in PNP_IDX], dtype=np.float64)
    f = float(width)
    cam = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(PNP_MODEL, pts2d, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    return _wrap(float(angles[0])), _wrap(float(angles[1])), _wrap(float(angles[2]))


def head_pitch(lm: np.ndarray, width: int, height: int) -> float | None:
    """Head pitch in degrees, positive = head tilted down. None if PnP fails."""
    pose = head_pose(lm, width, height)
    return None if pose is None else pose[0]


def gaze_score(iris_offset: float, pitch_deg: float | None, cfg: dict) -> float:
    """Combined vertical gaze score; larger = looking further down."""
    g = cfg["gaze"]
    pitch_term = 0.0 if pitch_deg is None else pitch_deg / g["pitch_norm_deg"]
    return g["iris_weight"] * iris_offset + g["pitch_weight"] * pitch_term


class Smoother:
    """Median filter (spike rejection) followed by an EMA."""

    def __init__(self, window: int = 5, alpha: float = 0.3):
        self.buf: deque[float] = deque(maxlen=max(1, window))
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.buf.append(x)
        med = float(np.median(self.buf))
        self.value = med if self.value is None else self.alpha * med + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.buf.clear()
        self.value = None


class ZoneClassifier:
    """Maps a smoothed score to UP/NEUTRAL/DOWN with time hysteresis.

    `up_thresh` < `down_thresh` come from calibration: scores below
    `up_thresh` are UP, above `down_thresh` are DOWN, in between is NEUTRAL.
    """

    def __init__(self, up_thresh: float, down_thresh: float, hysteresis_ms: float = 200,
                 up_scale: float = 1.0, down_scale: float = 1.0):
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        # Overshoot is divided by these so 1.0 ~= gaze at the calibration dot.
        self.up_scale = max(up_scale, 1e-6)
        self.down_scale = max(down_scale, 1e-6)
        self.hold_s = hysteresis_ms / 1000.0
        self.zone = Zone.NEUTRAL
        self._candidate = Zone.NEUTRAL
        self._candidate_since: float | None = None

    def raw_zone(self, score: float) -> tuple[Zone, float]:
        """Zone without hysteresis, plus overshoot (score distance past threshold)."""
        if score >= self.down_thresh:
            return Zone.DOWN, (score - self.down_thresh) / self.down_scale
        if score <= self.up_thresh:
            return Zone.UP, (self.up_thresh - score) / self.up_scale
        return Zone.NEUTRAL, 0.0

    def update(self, score: float, t: float) -> tuple[Zone, float]:
        raw, overshoot = self.raw_zone(score)
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = t
        if raw != self.zone and self._candidate_since is not None and t - self._candidate_since >= self.hold_s:
            self.zone = raw
        return self.zone, (overshoot if raw == self.zone else 0.0)

    def reset(self) -> None:
        self.zone = Zone.NEUTRAL
        self._candidate = Zone.NEUTRAL
        self._candidate_since = None
