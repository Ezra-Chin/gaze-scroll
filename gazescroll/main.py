"""GazeScroll entry point: python -m gazescroll.main"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import pyautogui

from . import actions, overlay
from .backends import create_backend
from .calibration import Calibration, run_calibration
from .camera import Camera
from .config import load_config, resolve
from .gaze import Smoother, Zone, eye_aspect_ratio, gaze_score, head_pitch, vertical_iris_offset


def calibrate(camera, backend, cfg) -> Calibration | None:
    calib = run_calibration(camera, backend, cfg, pyautogui.size())
    if calib is None:
        return None
    for w in calib.warnings:
        print(f"Warning: {w} Press C to recalibrate.")
    calib.save(resolve(cfg["calibration_path"]))
    print(f"Calibration saved: top={calib.top:.4f} center={calib.center:.4f} "
          f"(noise {calib.center_std:.4f}) bottom={calib.bottom:.4f} -> "
          f"UP < {calib.up_thresh:.4f}, DOWN > {calib.down_thresh:.4f}")
    return calib


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hands-free scrolling with gaze tracking.")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--calibrate", action="store_true", help="force calibration on start")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    actions.configure(cfg["failsafe"])
    cam_cfg = cfg["camera"]
    camera = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"])
    backend = create_backend(cfg["backend"], cfg)

    calib = None if args.calibrate else Calibration.load(resolve(cfg["calibration_path"]))
    if calib is None:
        calib = calibrate(camera, backend, cfg)
        if calib is None:
            print("Calibration aborted.")
            return 1

    g = cfg["gaze"]
    classifier = calib.classifier(g["hysteresis_ms"])
    smoother = Smoother(g["median_window"], g["ema_alpha"])
    sc = cfg["scroll"]
    scroller = actions.Scroller(sc["tick_ms"], sc["min"], sc["max"])
    ear_thresh = cfg["eyes"]["ear_closed_thresh"]

    show_overlay = cfg["overlay"]
    paused = False
    fps = None
    last_t = time.perf_counter()

    cv2.namedWindow(overlay.WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, t_cap = camera.read()
            if frame is None:
                continue
            t0 = time.perf_counter()
            face = backend.process(frame)
            infer_ms = (time.perf_counter() - t0) * 1000

            state = {"zone": Zone.NEUTRAL, "status": "", "infer_ms": infer_ms,
                     "up_thresh": calib.up_thresh, "down_thresh": calib.down_thresh,
                     "center": calib.center}
            zone, overshoot = Zone.NEUTRAL, 0.0
            now = time.perf_counter()
            if face is None:
                state["status"] = "NO FACE"
                smoother.reset()
                classifier.reset()
            else:
                lm = face.landmarks
                w, h = face.image_size
                iris, ear, pitch = vertical_iris_offset(lm), eye_aspect_ratio(lm), head_pitch(lm, w, h)
                state.update(landmarks=lm, iris=iris, ear=ear, pitch=pitch)
                if ear < ear_thresh:
                    # Iris position is meaningless mid-blink; hold state, don't scroll.
                    state["status"] = "EYES CLOSED"
                    state["score"] = smoother.value
                else:
                    score = smoother.update(gaze_score(iris, pitch, cfg))
                    zone, overshoot = classifier.update(score, now)
                    state["score"] = score
                    state["zone"] = zone

            if paused:
                state["status"] = "PAUSED"
            active = not state["status"]
            scroller.step(zone if active else Zone.NEUTRAL, overshoot, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 1 / dt if fps is None else 0.9 * fps + 0.1 / dt
            state["fps"] = fps
            state["frame_ms"] = (time.perf_counter() - t_cap) * 1000

            img = overlay.draw(frame, state) if show_overlay else overlay.draw_minimal(state)
            cv2.imshow(overlay.WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
            elif key == ord("o"):
                show_overlay = not show_overlay
            elif key == ord("r") and smoother.value is not None:
                # Re-center: current gaze becomes the new neutral baseline.
                calib = calib.recentered(smoother.value)
                calib.save(resolve(cfg["calibration_path"]))
                classifier = calib.classifier(g["hysteresis_ms"])
                print(f"Re-centered at {calib.center:.4f}")
            elif key == ord("c"):
                new = calibrate(camera, backend, cfg)
                if new is not None:
                    calib = new
                    classifier = calib.classifier(g["hysteresis_ms"])
                smoother.reset()
    finally:
        camera.release()
        backend.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
