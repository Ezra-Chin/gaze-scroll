"""Transparent, click-through, always-on-top gaze dot (macOS, PyObjC).

Runs as its own process (OpenCV's highgui owns the Cocoa run loop in the main
app). Reads lines "x y visible [radius]" from stdin, with screen coordinates in
points and a top-left origin (same as pyautogui). Exits when stdin closes.

    python -m gazescroll.pointer_overlay
"""
from __future__ import annotations

import sys
import threading

import AppKit
import objc
from Foundation import NSObject, NSTimer

_state = {"x": 0.0, "y": 0.0, "visible": False, "radius": 22.0, "alive": True}
_lock = threading.Lock()


def _reader() -> None:
    for line in sys.stdin:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, vis = float(parts[0]), float(parts[1]), parts[2] == "1"
            r = float(parts[3]) if len(parts) > 3 else None
        except ValueError:
            continue
        with _lock:
            _state.update(x=x, y=y, visible=vis)
            if r is not None:
                _state["radius"] = r
    with _lock:
        _state["alive"] = False


class DotView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(DotView, self).initWithFrame_(frame)
        if self is not None:
            self.snapshot = dict(_state)
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        AppKit.NSColor.clearColor().set()
        AppKit.NSRectFill(self.bounds())
        s = self.snapshot
        if not s["visible"]:
            return
        h = self.bounds().size.height
        r = s["radius"]
        cx, cy = s["x"], h - s["y"]  # flip to Cocoa's bottom-left origin
        ring = AppKit.NSBezierPath.bezierPathWithOvalInRect_(((cx - r, cy - r), (2 * r, 2 * r)))
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.55, 0.0, 0.25).set()
        ring.fill()
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.55, 0.0, 0.9).set()
        ring.setLineWidth_(2.5)
        ring.stroke()
        d = 4.0
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(((cx - d, cy - d), (2 * d, 2 * d))).fill()


class Ticker(NSObject):
    def initWithView_(self, view):
        self = objc.super(Ticker, self).init()
        if self is not None:
            self.view = view
            self.last = None
        return self

    def tick_(self, timer):
        with _lock:
            snap = dict(_state)
        if not snap["alive"]:
            AppKit.NSApp().terminate_(None)
            return
        key = (round(snap["x"]), round(snap["y"]), snap["visible"], snap["radius"])
        if key != self.last:
            self.last = key
            self.view.snapshot = snap
            self.view.setNeedsDisplay_(True)


def main() -> int:
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon

    frame = AppKit.NSScreen.mainScreen().frame()
    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
    win.setOpaque_(False)
    win.setBackgroundColor_(AppKit.NSColor.clearColor())
    win.setHasShadow_(False)
    win.setIgnoresMouseEvents_(True)
    win.setLevel_(AppKit.NSStatusWindowLevel)
    win.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary)
    view = DotView.alloc().initWithFrame_(((0, 0), (frame.size.width, frame.size.height)))
    win.setContentView_(view)
    win.orderFrontRegardless()

    threading.Thread(target=_reader, daemon=True).start()
    ticker = Ticker.alloc().initWithView_(view)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / 60, ticker, "tick:", None, True)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
