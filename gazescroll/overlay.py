"""Debug overlay drawing."""
from __future__ import annotations

import cv2
import numpy as np

from .gaze import (L_CORNERS, L_IRIS, L_LOWER, L_UPPER, PNP_IDX, R_CORNERS, R_IRIS,
                   R_LOWER, R_UPPER, Zone)

WINDOW = "GazeScroll"
EYE_POINTS = (*R_CORNERS, R_UPPER, R_LOWER, *L_CORNERS, L_UPPER, L_LOWER)
IRIS_RINGS = (range(469, 473), range(474, 478))
ZONE_COLORS = {Zone.UP: (255, 160, 0), Zone.NEUTRAL: (120, 120, 120), Zone.DOWN: (0, 140, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw on a mirrored copy of the frame. `state` keys are produced by main.py."""
    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    lm = state.get("landmarks")
    if lm is not None:
        def px(i):
            return int((1.0 - lm[i, 0]) * w), int(lm[i, 1] * h)
        for i in EYE_POINTS:
            cv2.circle(img, px(i), 2, (0, 255, 0), -1)
        for i in PNP_IDX:
            cv2.circle(img, px(i), 2, (255, 0, 255), -1)
        for c, ring in zip((R_IRIS, L_IRIS), IRIS_RINGS):
            r = int(np.mean([np.hypot(*np.subtract(px(j), px(c))) for j in ring]))
            cv2.circle(img, px(c), max(r, 1), (0, 255, 255), 1)
            cv2.circle(img, px(c), 2, (0, 0, 255), -1)

    zone: Zone = state.get("zone", Zone.NEUTRAL)
    status = state.get("status", "")
    banner = status if status else zone.value
    color = (60, 60, 200) if status else ZONE_COLORS[zone]
    cv2.rectangle(img, (0, 0), (w, 44), color, -1)
    _text(img, banner, (12, 31), 0.9, thick=2)

    def fmt(v, spec=".3f"):
        return "-" if v is None else format(v, spec)

    lines = [
        f"iris {fmt(state.get('iris'), '+.4f')}  pitch {fmt(state.get('pitch'), '.1f')} deg  EAR {fmt(state.get('ear'))}",
        f"score {fmt(state.get('score'), '+.4f')}  up<{fmt(state.get('up_thresh'), '+.4f')}"
        f"  center {fmt(state.get('center'), '+.4f')}  down>{fmt(state.get('down_thresh'), '+.4f')}",
        f"FPS {fmt(state.get('fps'), '.1f')}  frame {fmt(state.get('frame_ms'), '.1f')} ms"
        f"  infer {fmt(state.get('infer_ms'), '.1f')} ms",
        "P pause  R re-center  C calibrate  G gaze dot  K dot calib  O overlay  Q quit",
    ]
    for k, s in enumerate(lines):
        _text(img, s, (12, h - 16 - 24 * (len(lines) - 1 - k)))

    # Vertical gauge: smoothed score against thresholds.
    score, up, down = state.get("score"), state.get("up_thresh"), state.get("down_thresh")
    if None not in (score, up, down):
        lo, hi = sorted((up, down))
        pad = (hi - lo) or 0.1
        lo, hi = lo - pad, hi + pad
        x0, y0, y1 = w - 30, 60, h - 120
        def gy(v):
            v = (v - lo) / (hi - lo)
            return int(y0 + np.clip(v, 0, 1) * (y1 - y0))
        cv2.rectangle(img, (x0 - 6, y0), (x0 + 6, y1), (200, 200, 200), 1)
        for t in (up, down):
            cv2.line(img, (x0 - 12, gy(t)), (x0 + 12, gy(t)), (255, 255, 255), 1)
        if state.get("center") is not None:
            cv2.line(img, (x0 - 8, gy(state["center"])), (x0 + 8, gy(state["center"])), (0, 200, 0), 2)
        cv2.circle(img, (x0, gy(score)), 7, ZONE_COLORS[zone], -1)
    return img


def draw_minimal(state: dict) -> np.ndarray:
    """Tiny status window shown when the overlay is toggled off (keeps key input)."""
    zone: Zone = state.get("zone", Zone.NEUTRAL)
    status = state.get("status", "")
    img = np.zeros((60, 320, 3), np.uint8)
    img[:] = (60, 60, 200) if status else ZONE_COLORS[zone]
    _text(img, status or zone.value, (10, 38), 0.8, thick=2)
    return img
