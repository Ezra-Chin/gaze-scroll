"""2-D gaze pointer: map eye features to a screen position.

A 9-dot calibration collects eye features while the user looks at known
screen points; a ridge-regularized quadratic regression then maps features
to (x, y). A One-Euro filter keeps the dot steady during fixations while
still following fast eye jumps.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .calibration import close_fullscreen, open_fullscreen, sample_target
from .gaze import head_pose, horizontal_iris_offset, vertical_iris_offset

GRID = (0.1, 0.5, 0.9)
SETTLE_S = 0.8
SAMPLE_S = 1.5


def features(lm: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """[1, hx, vy, hx^2, vy^2, hx*vy, yaw/20, pitch/20], or None if head pose fails."""
    pose = head_pose(lm, width, height)
    if pose is None:
        return None
    pitch, yaw, _ = pose
    hx, vy = horizontal_iris_offset(lm), vertical_iris_offset(lm)
    return np.array([1.0, hx, vy, hx * hx, vy * vy, hx * vy, yaw / 20.0, pitch / 20.0])


class GazeMapper:
    def __init__(self, wx: np.ndarray, wy: np.ndarray, screen_size: tuple[int, int],
                 mean_error_px: float = 0.0):
        self.wx = np.asarray(wx, dtype=np.float64)
        self.wy = np.asarray(wy, dtype=np.float64)
        self.screen_size = tuple(screen_size)
        self.mean_error_px = mean_error_px

    @classmethod
    def fit(cls, X: np.ndarray, targets: np.ndarray, screen_size: tuple[int, int],
            ridge: float = 1e-3) -> "GazeMapper":
        """X: (n, k) features; targets: (n, 2) screen pixels. Features are
        standardized internally so the ridge penalty treats them evenly."""
        X = np.asarray(X, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        mu = X[:, 1:].mean(axis=0)
        sd = X[:, 1:].std(axis=0) + 1e-9
        Z = np.hstack([X[:, :1], (X[:, 1:] - mu) / sd])
        k = Z.shape[1]
        reg = np.sqrt(ridge * len(Z)) * np.eye(k)
        reg[0, 0] = 0.0  # don't shrink the intercept
        A = np.vstack([Z, reg])
        w = []
        for col in range(2):
            b = np.concatenate([targets[:, col], np.zeros(k)])
            wz = np.linalg.lstsq(A, b, rcond=None)[0]
            # Fold the standardization back into plain weights on raw features.
            wr = np.empty(k)
            wr[1:] = wz[1:] / sd
            wr[0] = wz[0] - np.dot(wz[1:], mu / sd)
            w.append(wr)
        mapper = cls(w[0], w[1], screen_size)
        pred = np.array([mapper.predict(x, clamp=False) for x in X])
        mapper.mean_error_px = float(np.mean(np.linalg.norm(pred - targets, axis=1)))
        return mapper

    def predict(self, feat: np.ndarray, clamp: bool = True) -> tuple[float, float]:
        x, y = float(feat @ self.wx), float(feat @ self.wy)
        if clamp:
            w, h = self.screen_size
            x, y = min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)
        return x, y

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "wx": self.wx.tolist(), "wy": self.wy.tolist(),
            "screen_size": list(self.screen_size), "mean_error_px": self.mean_error_px,
            "timestamp": time.time(),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GazeMapper | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return cls(d["wx"], d["wy"], tuple(d["screen_size"]), d.get("mean_error_px", 0.0))
        except (ValueError, KeyError, TypeError):
            return None


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) for one scalar signal."""

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.003, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: float | None = None
        self.dx = 0.0
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.t = x, t
            return x
        dt = t - self.t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self) -> None:
        self.x, self.t, self.dx = None, None, 0.0


class PointFilter:
    def __init__(self, min_cutoff: float, beta: float):
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


class OverlayProcess:
    """Owns the `pointer_overlay` subprocess and streams gaze points to it."""

    def __init__(self, radius: float):
        self.radius = radius
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "gazescroll.pointer_overlay"],
            stdin=subprocess.PIPE, text=True, bufsize=1,
        )

    def send(self, x: float, y: float, visible: bool) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(f"{x:.1f} {y:.1f} {int(visible)} {self.radius}\n")
        except (BrokenPipeError, OSError):
            pass

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_pointer_calibration(camera, backend, cfg: dict,
                            screen_size: tuple[int, int]) -> GazeMapper | None:
    """3x3 dot grid in random order; fits a GazeMapper. None if aborted."""
    sw, sh = screen_size
    points = [(int(fx * sw), int(fy * sh)) for fy in GRID for fx in GRID]
    random.shuffle(points)

    def measure(face):
        w, h = face.image_size
        return features(face.landmarks, w, h)

    X, T = [], []
    open_fullscreen()
    try:
        for i, pos in enumerate(points, 1):
            samples = sample_target(camera, backend, cfg, screen_size, pos,
                                    f"Look at the dot ({i}/{len(points)})", SAMPLE_S, measure,
                                    settle_s=SETTLE_S)
            if samples is None:
                return None
            # Per-dot median rejects saccades/outliers; also keep a few raw
            # samples so the fit sees some of the natural spread.
            arr = np.asarray(samples)
            X.append(np.median(arr, axis=0))
            T.append(pos)
            for row in arr[:: max(1, len(arr) // 5)]:
                X.append(row)
                T.append(pos)
    finally:
        close_fullscreen()

    return GazeMapper.fit(np.asarray(X), np.asarray(T, dtype=np.float64), screen_size,
                          ridge=cfg["pointer"]["ridge"])
