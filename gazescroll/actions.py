"""OS input: scroll and click via pyautogui.

macOS: the terminal running GazeScroll needs Accessibility permission
(System Settings > Privacy & Security > Accessibility), otherwise events are
silently dropped. Scroll events go to the window under the mouse pointer.
"""
from __future__ import annotations

import pyautogui

from .gaze import Zone


def configure(failsafe: bool) -> None:
    pyautogui.FAILSAFE = failsafe
    pyautogui.PAUSE = 0


class Scroller:
    """Sends scroll ticks at a steady rate; speed proportional to overshoot."""

    def __init__(self, tick_ms: float = 50, min_units: int = 1, max_units: int = 8, send=None):
        self.tick_s = tick_ms / 1000.0
        self.min_units = min_units
        self.max_units = max_units
        self._send = send or pyautogui.scroll
        self._last: float | None = None

    def units_for(self, overshoot: float) -> int:
        o = min(max(overshoot, 0.0), 1.0)
        return int(round(self.min_units + (self.max_units - self.min_units) * o))

    def step(self, zone: Zone, overshoot: float, t: float) -> float | None:
        """Maybe send a scroll. Returns t if an event was sent, else None."""
        if zone == Zone.NEUTRAL:
            self._last = None
            return None
        if self._last is not None and t - self._last < self.tick_s:
            return None
        n = self.units_for(overshoot)
        if n <= 0:
            return None
        self._send(n if zone == Zone.UP else -n)
        self._last = t
        return t


def click() -> None:
    pyautogui.click()
