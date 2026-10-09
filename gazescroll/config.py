"""Config loading: YAML on top of built-in defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "camera": {"index": 0, "width": 1280, "height": 720},
    "backend": "mediapipe",
    "model_path": "models/face_landmarker.task",
    "calibration_path": "calibration.json",
    "gaze": {
        "iris_weight": 1.0,
        "pitch_weight": 0.05,
        "pitch_norm_deg": 20.0,
        "median_window": 5,
        "ema_alpha": 0.3,
        "hysteresis_ms": 200,
    },
    "calibration": {
        "center_s": 3.0,
        "edge_s": 2.0,
        "up_fraction": 0.35,
        "down_fraction": 0.5,
        "noise_k": 2.5,
        "fallback_margin": 0.02,
    },
    "scroll": {"tick_ms": 50, "min": 1, "max": 8},
    "eyes": {"ear_closed_thresh": 0.18},
    "gestures": {
        "blink_click": False,
        "blink_click_ms": 600,
        "dwell_click": False,
        "dwell_s": 1.5,
        "dwell_radius": 0.03,
    },
    "pointer": {
        "enabled": False,
        "map_path": "gaze_map.json",
        "ridge": 1e-3,
        "min_cutoff": 0.5,
        "beta": 0.003,
        "radius": 22,
    },
    "overlay": True,
    "failsafe": True,
    "hotkey": "<ctrl>+<alt>+p",
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    user = {}
    if path.exists():
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, user)


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
