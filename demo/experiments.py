"""Feasibility experiments for FretCoach.

E1  k sweep         : how far past the MediaPipe TIP landmark the real contact point is.
E2  robustness      : 60 perturbed copies of the photo (shift, scale, rotation, brightness, noise, JPEG)
                      -> spread of the contact point in mm, and chord-check accuracy.
E3  error budget    : P(correct string) as a function of random error sigma and bias,
                      for classical (8.6 mm) and steel-string (7.3 mm) string spacing.
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

from fretcoach import CHORDS, Fretboard, check_chord, contact_point
from probe_hands import make_landmarker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

font_manager.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

IMG = ROOT / "test_images" / "am.jpg"
CORNERS = {"nut_6": (790, 167), "nut_1": (790, 347), "far_6": (27, 213), "far_1": (39, 454)}
CHORD = "Am"
rng = np.random.default_rng(7)


def landmarks(lm, bgr):
    h, w = bgr.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not res.hand_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)


def perturb(bgr):
    h, w = bgr.shape[:2]
    ang, sc = rng.uniform(-3, 3), rng.uniform(0.92, 1.08)
    tx, ty = rng.uniform(-15, 15, size=2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * alpha + beta + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(50, 91))
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img, cv2.invertAffineTransform(M)


def main():
    board = Fretboard(CORNERS, far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
    bgr = cv2.imread(str(IMG))
    lm = make_landmarker(num_hands=1, min_conf=0.3)
    base = landmarks(lm, bgr)
    summary = {}

    # ---------- E1: k sweep ----------
    ks = np.round(np.arange(0.0, 1.21, 0.05), 2)
    sweep = {f: [] for f in CHORDS[CHORD]}
    for k in ks:
        for r in check_chord(board, base, CHORD, float(k)):
            want_s = r["want"][0]
            v_target = board.string_v(want_s, r["u_mm"])
            sweep[r["finger"]].append(r["v_mm"] - v_target)  # signed distance to the TARGET string
    spacing = board.span(60) / 5
    ok_k = [float(k) for i, k in enumerate(ks) if all(abs(sweep[f][i]) < spacing / 2 for f in sweep)]
    summary["E1_k_all_correct_range"] = [min(ok_k), max(ok_k)] if ok_k else None
    summary["E1_raw_offsets_mm"] = {f: round(sweep[f][0], 2) for f in sweep}
    summary["E1_spacing_mm"] = round(spacing, 2)

    # ---------- E2: robustness under perturbation ----------
    K = 0.4 if not ok_k else round((min(ok_k) + max(ok_k)) / 2, 2)
    summary["E2_k_used"] = K
    pts = {f: [] for f in CHORDS[CHORD]}
    detected, all_ok, finger_ok = 0, 0, {f: 0 for f in CHORDS[CHORD]}
    N = 60
    for _ in range(N):
        img, Minv = perturb(bgr)
        lms = landmarks(lm, img)
        if lms is None:
            continue
        detected += 1
        lms = cv2.transform(lms.reshape(-1, 1, 2), Minv).reshape(-1, 2)  # back to original pixels
        res = check_chord(board, lms, CHORD, K)
        all_ok += all(r["status"] == "ok" for r in res)
        for r in res:
            finger_ok[r["finger"]] += r["status"] == "ok"
            pts[r["finger"]].append(board.to_world([r["point"]])[0])
    spread = {}
    for f, arr in pts.items():
        a = np.array(arr)
        spread[f] = {"sd_across_mm": round(float(a[:, 1].std()), 2), "sd_along_mm": round(float(a[:, 0].std()), 2)}
    summary["E2"] = {"variants": N, "hand_detected": detected, "chord_all_correct": all_ok,
                     "finger_correct": finger_ok, "spread_mm": spread}
    sigma_meas = float(np.mean([s["sd_across_mm"] for s in spread.values()]))
    summary["E2_mean_sd_across_mm"] = round(sigma_meas, 2)

    # E2b: average the contact point over W detections (proxy for temporal smoothing over W frames).
    # Perturbations are independent here, so this is an optimistic bound for real, correlated video frames.
    n_det = len(next(iter(pts.values())))
    window = {}
    for W in (1, 3, 5, 10):
        ok = 0
        trials = 500
        for _ in range(trials):
            idx = rng.choice(n_det, size=W, replace=False)
            good = True
            for f, (want_s, want_f) in CHORDS[CHORD].items():
                w_mean = np.array(pts[f])[idx].mean(axis=0)
                loc = board.locate(board.to_image([w_mean])[0])
                good &= (loc["string"], loc["fret"]) == (want_s, want_f)
            ok += good
        window[W] = round(ok / trials, 3)
    summary["E2b_chord_correct_by_window"] = window

    # ---------- E3: error budget ----------
    sig = np.linspace(0.2, 6, 60)
    n_mc = 200_000

    def p_correct(spacing_mm, sigma, bias=0.0, avg=1):
        e = bias + rng.normal(0, sigma / np.sqrt(avg), n_mc)
        return float(np.mean(np.abs(e) < spacing_mm / 2))

    curves = {}
    for name, sp in (("古典吉他 弦距 8.6 mm", 8.6), ("民謠吉他 弦距 7.3 mm", 7.3)):
        curves[name] = [p_correct(sp, s) for s in sig]
    curves_bias = {b: [p_correct(7.3, s, bias=b) for s in sig] for b in (2.0, 3.0)}
    curves_avg = [p_correct(7.3, s, avg=10) for s in sig]
    summary["E3_steel_single_frame_at"] = {f"{s} mm": round(p_correct(7.3, s), 3) for s in (1, 2, 3, 4, 5)}
    summary["E3_steel_bias3_at"] = {f"{s} mm": round(p_correct(7.3, s, bias=3.0), 3) for s in (1, 2, 3)}

    # ---------- figures ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.axhspan(-spacing / 2, spacing / 2, color="#2e7d32", alpha=0.08, label="判斷正確的範圍（±半個弦距）")
    names = {"index": "食指", "middle": "中指", "ring": "無名指"}
    for f, ys in sweep.items():
        ax.plot(ks, ys, marker="o", ms=3, label=names[f])
    ax.axhline(0, color="#555", lw=0.8)
    if ok_k:
        ax.axvspan(min(ok_k), max(ok_k), color="#1565c0", alpha=0.08)
        ax.text((min(ok_k) + max(ok_k)) / 2, spacing / 2 + 0.6, f"k = {min(ok_k)}–{max(ok_k)} 三指全對",
                ha="center", fontsize=9, color="#1565c0")
    ax.set_xlabel("修正量 k（沿 DIP→TIP 方向外推的比例；0 = MediaPipe 原始指尖）")
    ax.set_ylabel("與目標弦的距離 (mm)")
    ax.set_title("E1　原始指尖系統性偏向高音側約半個弦距，外推修正後落回目標弦", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "E1_k_sweep.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for name, ys in curves.items():
        ax.plot(sig, ys, lw=2, label=f"{name}，單一畫面")
    ax.plot(sig, curves_avg, lw=2, ls="--", label="民謠吉他，平均 10 個畫面")
    for b, ys in curves_bias.items():
        ax.plot(sig, ys, lw=1.4, ls=":", label=f"民謠吉他，另有 {b:.0f} mm 系統偏差")
    ax.axvline(sigma_meas, color="#c62828", lw=1)
    ax.text(sigma_meas + 0.08, 0.32, f"E2 實測擾動標準差\n≈ {sigma_meas:.1f} mm", color="#c62828", fontsize=8)
    ax.axhline(0.9, color="#999", lw=0.6, ls="--")
    ax.set_xlabel("指尖位置的隨機誤差 σ（mm，橫跨弦的方向）")
    ax.set_ylabel("判斷到正確弦的機率")
    ax.set_ylim(0, 1.02)
    ax.set_title("E3　隨機誤差 σ 要壓到 2 mm 以下，且系統偏差要先修掉，才分得出相鄰的弦", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "E3_error_budget.png")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
