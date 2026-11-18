"""E8: markerless fretboard tracking — can we follow a moving guitar without any stickers?

Calibrate once on the reference frame (four clicks). In every later frame, find SIFT features inside the
fretboard area (hand excluded), match them to the reference frame, and estimate the homography with RANSAC.
The new calibration is the reference calibration composed with that homography.

Same movements and image degradation as E6/E7, and the same error measure: calibration error in mm at the
three fingertips. Uses the original photo (no markers). The whole photo moves rigidly, so this is an
optimistic test: real frames add motion blur, lighting changes and string vibration.
"""
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from experiments_movement import (CHORD, CORNERS, HALF, K, PIVOT_MM, OUT, ROOT, degrade, p_correct)
from fretcoach import Fretboard, check_chord
from probe_hands import make_landmarker

SIFT = cv2.SIFT_create(nfeatures=3000)
MATCHER = cv2.BFMatcher(cv2.NORM_L2)


def fretboard_mask(board, shape, hand_pts, grow_px=0, hand_px=30):
    """Fretboard polygon (nut to fret 5, plus a margin) minus the hand."""
    poly = board.to_image(np.float32([[-4, -6], [175, -8], [175, 51], [-4, 49]])).astype(np.int32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if grow_px:
        mask = cv2.dilate(mask, np.ones((2 * grow_px + 1,) * 2, np.uint8))
    hull = cv2.convexHull(hand_pts.astype(np.int32))
    hand = np.zeros_like(mask)
    cv2.fillConvexPoly(hand, hull, 255)
    hand = cv2.dilate(hand, np.ones((2 * hand_px + 1,) * 2, np.uint8))
    return cv2.bitwise_and(mask, cv2.bitwise_not(hand))


def main():
    rng = np.random.default_rng(23)
    ref = cv2.imread(str(ROOT / "test_images" / "am.jpg"))
    h, w = ref.shape[:2]
    base = Fretboard(CORNERS, far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
    res = make_landmarker(num_hands=1).detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)))
    lms = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)
    fingers = np.float32([r["point"] for r in check_chord(base, lms, CHORD, K)])

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_mask = fretboard_mask(base, ref.shape, lms)
    kp_r, des_r = SIFT.detectAndCompute(ref_gray, ref_mask)
    cv2.imwrite(str(OUT / "tracking_features.png"),
                cv2.drawKeypoints(ref, kp_r, None, color=(0, 255, 255)))

    hand_c = fingers.mean(axis=0)
    u, v = base.to_world([hand_c])[0]
    px_across = base.to_image([[u, v + 1.0]])[0] - base.to_image([[u, v]])[0]
    px_along = base.to_image([[u + 1.0, v]])[0] - base.to_image([[u, v]])[0]
    pivot = hand_c + px_along * PIVOT_MM

    def move_matrix(kind, a):
        if kind == "across":
            t = px_across * a
            return np.float32([[1, 0, t[0]], [0, 1, t[1]]])
        if kind == "along":
            t = px_along * a
            return np.float32([[1, 0, t[0]], [0, 1, t[1]]])
        if kind == "rotate":
            return cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), a, 1.0)
        return cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 1.0 + a / 100.0)

    sweeps = {
        "across": np.round(np.arange(0, 8.01, 0.5), 2),
        "along": np.round(np.arange(0, 16.01, 1.0), 2),
        "rotate": np.round(np.arange(0, 1.01, 0.1), 2),
        "scale": np.round(np.arange(0, 10.01, 1.0), 2),
    }
    out, inliers_all = {}, []
    for kind, amounts in sweeps.items():
        rows = []
        for a in amounts:
            errs, ok = [], 0
            for sign in (+1, -1):
                M = move_matrix(kind, sign * a)
                moved_fingers = cv2.transform(fingers.reshape(-1, 1, 2), M).reshape(-1, 2)
                moved_lms = cv2.transform(lms.reshape(-1, 1, 2), M).reshape(-1, 2)
                true_w = base.to_world(fingers)
                for _ in range(5):
                    moved = degrade(cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=(40, 40, 40)), rng)
                    # search where the fretboard was last seen, grown a little, without the hand
                    mask = fretboard_mask(base, moved.shape, moved_lms, grow_px=40)
                    kp_m, des_m = SIFT.detectAndCompute(cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY), mask)
                    if des_m is None or len(kp_m) < 8:
                        continue
                    pairs = MATCHER.knnMatch(des_r, des_m, k=2)
                    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
                    if len(good) < 8:
                        continue
                    src = np.float32([kp_r[m.queryIdx].pt for m in good])
                    dst = np.float32([kp_m[m.trainIdx].pt for m in good])
                    Hrm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                    if Hrm is None or inl.sum() < 8:
                        continue
                    ok += 1
                    inliers_all.append(int(inl.sum()))
                    back = cv2.perspectiveTransform(moved_fingers.reshape(-1, 1, 2), np.linalg.inv(Hrm)).reshape(-1, 2)
                    errs.append(np.abs(base.to_world(back) - true_w))
            e = np.concatenate(errs) if errs else np.full((1, 2), np.nan)
            rows.append({"amount": float(a), "track_err_across_mm": round(float(np.nanmax(e[:, 1])), 2),
                         "track_err_along_mm": round(float(np.nanmax(e[:, 0])), 2), "tracked": f"{ok}/10"})
        out[kind] = rows
        print(kind, [(r["amount"], r["track_err_across_mm"], r["track_err_along_mm"], r["tracked"]) for r in rows][::2])

    worst = [max(r["track_err_across_mm"], r["track_err_along_mm"]) for rows in out.values() for r in rows]
    tracked = sum(int(r["tracked"].split("/")[0]) for rows in out.values() for r in rows)
    total = sum(int(r["tracked"].split("/")[1]) for rows in out.values() for r in rows)
    summary = {"rows": out, "reference_features": len(kp_r), "tracked": f"{tracked}/{total}",
               "median_inliers": int(np.median(inliers_all)) if inliers_all else 0,
               "worst_err_mm": max(worst), "median_condition_worst_err_mm": float(np.median(worst)),
               "conditions_over_1mm": int(sum(x > 1 for x in worst)), "conditions": len(worst),
               "p_correct_at_worst_steel": round(p_correct(max(worst), HALF["steel"]), 3)}
    (OUT / "tracking_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
