"""FretCoach live demo (webcam or video file).

Run:
  .venv/bin/python webcam_demo.py                 # webcam 0, steel-string guitar
  .venv/bin/python webcam_demo.py --guitar classical --chord C
  .venv/bin/python webcam_demo.py --video clip.mp4 --headless --frames 120   # offline test, prints stats

Keys (window must be focused):
  c        calibrate: freeze the frame, then click 4 points in this order
             1) 6th string at the nut   2) 1st string at the nut
             3) 6th string at fret 5    4) 1st string at fret 5
  1..8     choose chord (C G D A E Am Em Dm)
  m        mirror view on/off
  q / Esc  quit

Design notes (see the technical report):
  * contact point = TIP + k * (TIP - DIP); raw MediaPipe tips sit ~half a string toward the palm.
  * the contact point is averaged over the last `window` frames in fretboard (mm) coordinates.
  * dwell: a finger is only judged after it has stayed still for `dwell` seconds (avoids Midas touch
    while the player is still placing fingers).
  * errors are shown with a cross (not only red) and one text hint at a time.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from fretcoach import CHORDS, FINGERS, Fretboard, contact_point, fret_distance

HERE = Path(__file__).parent
MODEL = HERE / "models" / "hand_landmarker.task"
CALIB = HERE / "calib.json"
CHORD_KEYS = {ord(str(i + 1)): name for i, name in enumerate(CHORDS)}
GUITARS = {"steel": dict(scale_mm=645.0, span_nut_mm=36.5), "classical": dict(scale_mm=650.0, span_nut_mm=43.0)}
GREEN, RED, GREY, WHITE = (80, 200, 80), (60, 60, 230), (170, 170, 170), (255, 255, 255)


def make_video_landmarker():
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


def calibrate(frame) -> dict | None:
    pts, names = [], ["6th @ nut", "1st @ nut", "6th @ fret 5", "1st @ fret 5"]
    win = "calibrate"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 4:
        img = frame.copy()
        for p in pts:
            cv2.circle(img, p, 5, (0, 255, 255), -1)
        cv2.putText(img, f"click: {names[len(pts)]}   (Esc = cancel)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    corners = dict(zip(["nut_6", "nut_1", "far_6", "far_1"], pts))
    CALIB.write_text(json.dumps(corners))
    return corners


class FingerState:
    """Per-finger smoothing + dwell timer."""

    def __init__(self, window: int):
        self.buf = deque(maxlen=window)
        self.still_since = None

    def update(self, world_pt, now, move_tol_mm=3.0):
        if self.buf and np.linalg.norm(world_pt - np.mean(self.buf, axis=0)) > move_tol_mm:
            self.still_since = now  # moved: restart the dwell timer
        elif self.still_since is None:
            self.still_since = now
        self.buf.append(world_pt)
        return np.mean(self.buf, axis=0)

    def reset(self):
        self.buf.clear()
        self.still_since = None


def target_world(board: Fretboard, string: int, fret: int):
    """Where the finger should land: just behind the fret wire (70% of the way into the cell)."""
    lo, hi = fret_distance(fret - 1, board.scale_mm), fret_distance(fret, board.scale_mm)
    u = float(lo + 0.7 * (hi - lo))
    return np.array([u, board.string_v(string, u)], dtype=np.float32)


def run(a):
    corners = json.loads(CALIB.read_text()) if CALIB.exists() and not a.video else None
    if a.corners:
        c = [float(x) for x in a.corners.split(",")]
        corners = {"nut_6": c[0:2], "nut_1": c[2:4], "far_6": c[4:6], "far_1": c[6:8]}
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera/video. On macOS, allow camera access for your terminal app.")
    lm = make_video_landmarker()
    chord, mirror = a.chord, False
    states = {f: FingerState(a.window) for f in FINGERS}
    board = Fretboard(corners, **GUITARS[a.guitar]) if corners else None
    t0, n, lat, hits = time.perf_counter(), 0, [], {"frames": 0, "hand": 0, "all_ok": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        ts_ms = int((now - t0) * 1000) if not a.video else int(n * 1000 / (cap.get(cv2.CAP_PROP_FPS) or 30))
        h, w = frame.shape[:2]
        tic = time.perf_counter()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)
        n += 1
        hits["frames"] += 1
        msg = ""
        if board is not None and not a.headless:
            board.draw(frame)  # grid first, finger markers on top
        if board is None:
            msg = "press C to calibrate the fretboard"
        elif not res.hand_landmarks:
            msg = "no hand - keep palm and wrist in view"
            for s in states.values():
                s.reset()
        else:
            hits["hand"] += 1
            lms = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)
            all_ok, first_err = True, None
            for finger, (want_s, want_f) in CHORDS[chord].items():
                st = states[finger]
                world = board.to_world([contact_point(lms, finger, a.k)])[0]
                smooth = st.update(world, now)
                loc = board.locate(board.to_image([smooth])[0])
                correct = (loc["string"], loc["fret"]) == (want_s, want_f)
                settled = st.still_since is not None and now - st.still_since >= a.dwell
                all_ok &= correct
                tgt = board.to_image([target_world(board, want_s, want_f)])[0]
                p = board.to_image([smooth])[0]
                if not a.headless:
                    cv2.circle(frame, tuple(map(int, tgt)), 9, WHITE, 1, cv2.LINE_AA)  # ghost target
                    if correct:
                        cv2.circle(frame, tuple(map(int, p)), 8, GREEN, -1, cv2.LINE_AA)
                    elif settled:
                        cv2.drawMarker(frame, tuple(map(int, p)), RED, cv2.MARKER_TILTED_CROSS, 18, 3, cv2.LINE_AA)
                        cv2.line(frame, tuple(map(int, p)), tuple(map(int, tgt)), RED, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, tuple(map(int, p)), 6, GREY, -1, cv2.LINE_AA)
                if not correct and settled and first_err is None:
                    if loc["fret"] != want_f:
                        first_err = f"{finger}: move to fret {want_f}"
                    else:
                        side = "up (toward 6th)" if want_s > loc["string"] else "down (toward 1st)"
                        first_err = f"{finger}: one string {side}"
            hits["all_ok"] += all_ok
            msg = "nice! chord correct" if all_ok else (first_err or "hold still...")
        lat.append((time.perf_counter() - tic) * 1000)

        if a.headless:
            if a.frames and n >= a.frames:
                break
            continue
        fps = n / (time.perf_counter() - t0)
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"[{chord}] {msg}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:4.1f} fps  {np.mean(lat[-30:]):4.0f} ms", (w - 210, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("FretCoach", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("c"):
            c = calibrate(frame)
            if c:
                board = Fretboard(c, **GUITARS[a.guitar])
        if key in CHORD_KEYS:
            chord = CHORD_KEYS[key]
            for s in states.values():
                s.reset()

    cap.release()
    cv2.destroyAllWindows()
    lat = np.array(lat[5:] or lat)  # skip warm-up frames
    stats = {"frames": hits["frames"], "hand_detected": hits["hand"], "chord_all_correct": hits["all_ok"],
             "latency_ms_mean": round(float(lat.mean()), 1), "latency_ms_p95": round(float(np.percentile(lat, 95)), 1)}
    print(json.dumps(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--guitar", choices=GUITARS, default="steel")
    ap.add_argument("--chord", choices=list(CHORDS), default="Am")
    ap.add_argument("--k", type=float, default=0.4, help="contact-point correction (see report E1)")
    ap.add_argument("--window", type=int, default=10, help="frames to average")
    ap.add_argument("--dwell", type=float, default=0.6, help="seconds a finger must be still before it is judged")
    ap.add_argument("--corners", help="skip clicking: nut6x,nut6y,nut1x,nut1y,far6x,far6y,far1x,far1y")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    run(ap.parse_args())
