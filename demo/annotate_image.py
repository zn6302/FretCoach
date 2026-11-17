"""Run the full FretCoach pipeline on a still photo and save an annotated result.

Usage:
  python annotate_image.py IMAGE --corners nut6x,nut6y,nut1x,nut1y,far6x,far6y,far1x,far1y \
      --far-fret 5 --chord Am --k 0.6 --out result.png
"""
import argparse
import json

import cv2
import mediapipe as mp
import numpy as np

from fretcoach import FINGERS, Fretboard, check_chord, contact_point
from probe_hands import make_landmarker

STATUS_COLOR = {"ok": (80, 200, 80), "wrong_fret": (60, 60, 230), "wrong_string": (60, 60, 230)}


def detect_landmarks(bgr, min_conf=0.3):
    h, w = bgr.shape[:2]
    res = make_landmarker(num_hands=1, min_conf=min_conf).detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)


def draw_result(img, board, lms, results, k):
    board.draw(img, max_fret=board.far_fret)
    for f, (tip, dip) in FINGERS.items():
        cv2.line(img, tuple(map(int, lms[dip])), tuple(map(int, lms[tip])), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, lms[tip])), 5, (255, 255, 255), 1, cv2.LINE_AA)  # raw tip: hollow
    for r in results:
        p = tuple(map(int, r["point"]))
        col = STATUS_COLOR[r["status"]]
        if r["status"] == "ok":
            cv2.circle(img, p, 8, col, -1, cv2.LINE_AA)
        else:  # different shape, not only colour (colour-blind safe)
            cv2.drawMarker(img, p, col, cv2.MARKER_TILTED_CROSS, 18, 3, cv2.LINE_AA)
        label = f"{r['finger']} s{r['got'][0]}f{r['got'][1]}"
        cv2.putText(img, label, (p[0] + 10, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (p[0] + 10, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(img, f"k = {k}", (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--corners", required=True)
    ap.add_argument("--far-fret", type=int, default=5)
    ap.add_argument("--scale-mm", type=float, default=650.0)
    ap.add_argument("--span-mm", type=float, default=43.0)
    ap.add_argument("--chord", required=True)
    ap.add_argument("--k", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--min-conf", type=float, default=0.3)
    a = ap.parse_args()

    c = [float(x) for x in a.corners.split(",")]
    board = Fretboard({"nut_6": c[0:2], "nut_1": c[2:4], "far_6": c[4:6], "far_1": c[6:8]},
                      far_fret=a.far_fret, scale_mm=a.scale_mm, span_nut_mm=a.span_mm)
    bgr = cv2.imread(a.image)
    lms = detect_landmarks(bgr, a.min_conf)
    if lms is None:
        print(json.dumps({"image": a.image, "hand": False}))
        return
    report = []
    for k in a.k:
        res = check_chord(board, lms, a.chord, k)
        img = bgr.copy()
        draw_result(img, board, lms, res, k)
        cv2.imwrite(f"{a.out}_k{k}.png", img)
        for r in res:
            report.append({"k": k, "finger": r["finger"], "want": r["want"], "got": r["got"], "status": r["status"],
                           "string_offset_mm": round(r["string_offset_mm"], 1),
                           "margin_mm": round(r["string_margin_mm"], 1),
                           "mm_per_px": round(board.mm_per_px(r["point"]), 3)})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
