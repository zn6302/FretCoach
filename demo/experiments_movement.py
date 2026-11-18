"""E6/E7: how far can the guitar move before a one-time calibration breaks, and do ArUco markers fix it?

The movement problem is geometric, so it is measured without MediaPipe noise:
E6  calibration error at the three fingertips after the guitar moves, when the calibration from the
    original position is kept (pure geometry: where the old grid says the finger is vs where it really is).
E7  the same error when three ArUco markers on the fretboard are detected in every frame and the
    homography is recomputed. Frames get blur, sensor noise and JPEG artefacts so marker detection is
    not trivially perfect. Markers are composited synthetically (no strings over them).
The tolerance is then derived by adding the calibration error as a bias on top of the MediaPipe error
measured in E2 (sigma 2.9 mm per frame, ~0.9 mm after averaging 10 frames).
"""
import json
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import erf, sqrt

from fretcoach import CHORDS, Fretboard, check_chord, fret_distance
from probe_hands import make_landmarker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
font_manager.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

CORNERS = {"nut_6": (790, 167), "nut_1": (790, 347), "far_6": (27, 213), "far_1": (39, 454)}
CHORD, K = "Am", 0.4
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
MARKERS = {0: (12.0, -1.0), 1: (140.0, -1.0), 2: (140.0, 33.0)}  # top-left (u, v) of 10 mm squares
SIZE_MM = 10.0
PIVOT_MM = 450.0          # the neck swings around the guitar body, ~45 cm from the hand
SIGMA_10 = 2.9 / sqrt(10)  # E2 per-frame sigma across strings, averaged over 10 frames (optimistic)
HALF = {"steel": 7.3 / 2, "classical": 8.6 / 2}


def marker_world(mid):
    u0, v0 = MARKERS[mid]
    return np.float32([[u0 + SIZE_MM, v0], [u0, v0], [u0, v0 + SIZE_MM], [u0 + SIZE_MM, v0 + SIZE_MM]])


def add_markers(bgr, board):
    out = bgr.copy()
    for mid in MARKERS:
        side, quiet = 120, 24
        tile = np.full((side + 2 * quiet,) * 2, 255, np.uint8)
        tile[quiet:quiet + side, quiet:quiet + side] = cv2.aruco.generateImageMarker(DICT, mid, side)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        w = marker_world(mid)
        c = w.mean(axis=0)
        dst = board.to_image(c + (w - c) * (side + 2 * quiet) / side).astype(np.float32)
        src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        size = (out.shape[1], out.shape[0])
        warped = cv2.warpPerspective(tile, Hm, size)
        mask = cv2.warpPerspective(np.full(tile.shape[:2], 255, np.uint8), Hm, size)
        out[mask > 0] = warped[mask > 0]
    return out


def board_from_markers(img, template):
    corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    img_pts = [c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in MARKERS]
    world_pts = [marker_world(int(i)) for i in ids.ravel() if int(i) in MARKERS]
    if len(img_pts) < 2:
        return None
    H, _ = cv2.findHomography(np.concatenate(img_pts), np.concatenate(world_pts))
    b = Fretboard(template.corners, far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
    b.H, b.H_inv = H, np.linalg.inv(H)
    return b


def degrade(img, rng):
    """Camera-like degradation so that marker detection is not on a perfect image."""
    k = int(rng.choice([1, 3, 3, 5]))
    f = cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img
    f = np.clip(f.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 91))])[1], cv2.IMREAD_COLOR)


def p_correct(bias, half, sigma=SIGMA_10):
    """P(|bias + N(0, sigma)| < half)."""
    phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return phi((half - bias) / sigma) - phi((-half - bias) / sigma)


def main():
    rng = np.random.default_rng(11)
    base = Fretboard(CORNERS, far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
    photo = add_markers(cv2.imread(str(ROOT / "test_images" / "am.jpg")), base)
    cv2.imwrite(str(OUT / "am_with_markers.png"), photo)
    h, w = photo.shape[:2]

    # contact points of the three fingers in the original photo (k = 0.4)
    bgr = cv2.imread(str(ROOT / "test_images" / "am.jpg"))
    res = make_landmarker(num_hands=1).detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    lms = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)
    fingers = np.float32([r["point"] for r in check_chord(base, lms, CHORD, K)])
    hand_c = fingers.mean(axis=0)
    u, v = base.to_world([hand_c])[0]
    px_across = base.to_image([[u, v + 1.0]])[0] - base.to_image([[u, v]])[0]   # px per mm
    px_along = base.to_image([[u + 1.0, v]])[0] - base.to_image([[u, v]])[0]
    pivot = hand_c + px_along * PIVOT_MM   # in pixels, avoids extrapolating the homography

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
        "across": ("橫跨弦方向平移 (mm)", np.round(np.arange(0, 8.01, 0.5), 2)),
        "along": ("沿琴頸方向平移 (mm)", np.round(np.arange(0, 16.01, 1.0), 2)),
        "rotate": ("琴頸以琴身為軸擺動 (°)", np.round(np.arange(0, 1.01, 0.1), 2)),
        "scale": ("靠近／遠離鏡頭 (%)", np.round(np.arange(0, 10.01, 1.0), 2)),
    }
    out = {}
    for kind, (_, amounts) in sweeps.items():
        rows = []
        for a in amounts:
            fixed_err, marker_err, detected = [], [], 0
            for sign in (+1, -1):
                M = move_matrix(kind, sign * a)
                Minv = cv2.invertAffineTransform(M)
                moved_fingers = cv2.transform(fingers.reshape(-1, 1, 2), M).reshape(-1, 2)
                true_w = base.to_world(fingers)  # where the fingers really are on the fretboard
                fixed_err.append(np.abs(base.to_world(moved_fingers) - true_w))
                for _ in range(5):
                    moved = cv2.warpAffine(photo, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
                    mb = board_from_markers(degrade(moved, rng), base)
                    if mb is None:
                        continue
                    detected += 1
                    marker_err.append(np.abs(mb.to_world(moved_fingers) - true_w))
            fe = np.concatenate(fixed_err)
            me = np.concatenate(marker_err) if marker_err else np.full((1, 2), np.nan)
            rows.append({
                "amount": float(a),
                "fixed_err_across_mm": round(float(fe[:, 1].max()), 2),
                "fixed_err_along_mm": round(float(fe[:, 0].max()), 2),
                "marker_err_across_mm": round(float(np.nanmax(me[:, 1])), 2),
                "marker_err_along_mm": round(float(np.nanmax(me[:, 0])), 2),
                "marker_detected": f"{detected}/10",
            })
        out[kind] = rows

    # tolerance: largest movement that keeps P(correct string) >= 90% for a finger (10-frame average)
    tol = {}
    for kind, rows in out.items():
        for g, half in HALF.items():
            ok = [r["amount"] for r in rows if p_correct(r["fixed_err_across_mm"], half) >= 0.9]
            tol[f"{kind}_{g}"] = max(ok) if ok else 0.0
    # along the neck the limit is the fret judgement: distance from each finger to its two fret wires
    sigma_along = 3.0 / sqrt(10)  # E2 per-frame sigma along the neck (1.8-4.7 mm, mean ~3), 10-frame average
    wires = fret_distance(np.arange(0, 13), 650.0)
    margins = []
    for r in check_chord(base, lms, CHORD, K):
        lo, hi = wires[r["fret"] - 1], wires[r["fret"]]
        margins.append({"finger": r["finger"], "to_body_side_wire_mm": round(hi - r["u_mm"], 1),
                        "to_nut_side_wire_mm": round(r["u_mm"] - lo, 1)})
    def p_fret(d):
        """Worst-case P(finger stays between its two fret wires) when the guitar shifts d mm along the neck."""
        phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        ps = []
        for m in margins:
            for s_ in (+1, -1):
                ps.append(phi((m["to_body_side_wire_mm"] - s_ * d) / sigma_along)
                          - phi((-m["to_nut_side_wire_mm"] - s_ * d) / sigma_along))
        return min(ps)
    tol["along_fret"] = max(d for d in np.arange(0, 16.01, 0.5) if p_fret(d) >= 0.9)
    summary = {"rows": out, "tolerance_for_90pct": tol, "fret_wire_margins": margins,
               "marker_worst_err_mm": max(max(r["marker_err_across_mm"], r["marker_err_along_mm"])
                                          for rows in out.values() for r in rows),
               "sigma_10frames_mm": round(SIGMA_10, 2), "pivot_mm": PIVOT_MM}
    (OUT / "movement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({"tolerance_for_90pct": tol, "marker_worst_err_mm": summary["marker_worst_err_mm"]}, ensure_ascii=False))
    for kind in out:
        print(kind, [(r["amount"], r["fixed_err_across_mm"], r["marker_err_across_mm"], r["marker_detected"]) for r in out[kind]][::2])

    # figure: P(correct string) for the steel guitar, fixed vs marker, across-string shift and neck swing
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, sharey=True)
    for ax, kind in zip(axes, ("across", "rotate")):
        rows = out[kind]
        x = [r["amount"] for r in rows]
        ax.plot(x, [p_correct(r["fixed_err_across_mm"], HALF["steel"]) for r in rows], "o-", color="#c62828",
                label="只校正一次")
        ax.plot(x, [p_correct(r["marker_err_across_mm"], HALF["steel"]) for r in rows], "s--", color="#2e7d32",
                label="ArUco 標記每格重新校正")
        t = tol[f"{kind}_steel"]
        ax.axvline(t, color="#c62828", lw=0.8, ls=":")
        ax.text(t, 0.08, f" 容忍上限 ≈ {t:g}{' mm' if kind == 'across' else '°'}", color="#c62828", fontsize=10)
        ax.axhline(0.9, color="#999", lw=0.6, ls="--")
        ax.set_xlabel(sweeps[kind][0])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("單指判斷到正確弦的機率\n（民謠吉他，平均 10 個畫面）")
    axes[0].legend(fontsize=9, loc="center right")
    fig.suptitle("E6／E7　只校正一次：吉他移動約 2 mm、琴頸擺動約 0.3° 就開始判錯；每格用標記重新校正則不受影響",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "E6_movement.png")


if __name__ == "__main__":
    main()
