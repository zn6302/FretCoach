"""Figures for the movement problem (E6/E7/E8): report version, slide version, and the tracking illustration."""
import json
from math import erf, sqrt
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from experiments_tracking import SIFT, fretboard_mask
from fretcoach import Fretboard
from probe_hands import make_landmarker

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FONT = "/Library/Fonts/Arial Unicode.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE, GREY = "#c62828", "#1565c0", "#9e9e9e"

mv = json.loads((RES / "movement_summary.json").read_text())
tr = json.loads((RES / "tracking_summary.json").read_text())
sigma, half = mv["sigma_10frames_mm"], 7.3 / 2
phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
p = lambda b: phi((half - b) / sigma) - phi((-half - b) / sigma)
PANELS = (("across", "吉他橫跨弦方向移動 (mm)", "mm"), ("rotate", "琴頸上下擺動 (°)", "°"))


def plot(path, with_marker, title=None, big=False):
    fs = 15 if big else 11
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3) if big else (11, 4.2), dpi=170 if big else 150, sharey=True)
    for ax, (kind, label, unit) in zip(axes, PANELS):
        x = [r["amount"] for r in mv["rows"][kind]]
        ax.plot(x, [p(r["fixed_err_across_mm"]) for r in mv["rows"][kind]], "o-", color=RED, lw=2.5, ms=6,
                label="只校正一次")
        ax.plot(x, [p(r["track_err_across_mm"]) for r in tr["rows"][kind]], "s-", color=BLUE, lw=2.5, ms=6,
                label="每格追蹤指板（不貼標記）")
        if with_marker:
            ax.plot(x, [p(r["marker_err_across_mm"]) for r in mv["rows"][kind]], "^--", color=GREY, lw=1.5, ms=5,
                    label="ArUco 標記（備案）")
        t = mv["tolerance_for_90pct"][f"{kind}_steel"]
        ax.axvline(t, color=RED, lw=1.2, ls=":")
        ax.text(t, 0.55, f"≈ {t:g} {unit} \n就開始判錯 ", color=RED, fontsize=fs - 1, ha="right")
        ax.set_xlabel(label, fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("判斷到正確弦的機率" + ("" if big else "\n（民謠吉他，平均 10 個畫面）"), fontsize=fs)
    handles, labels = axes[0].get_legend_handles_labels()
    if title:
        fig.suptitle(title, fontsize=fs, y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=fs - 2,
               bbox_to_anchor=(0.5, 0.95 if title else 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 0.9))
    fig.savefig(path, facecolor="white")


plot(RES / "E6_movement.png", with_marker=True,
     title="E6–E8　只校正一次：移動約 2 mm、琴頸擺動約 0.3° 就判錯；每格追蹤指板則不受移動影響")
plot(RES / "slide_movement.png", with_marker=False, big=True)
Image.open(RES / "slide_movement.png").convert("RGB").save(RES / "slide_movement.jpg", quality=92)

# tracking illustration: the texture features the tracker follows, drawn large enough for a slide
ref = cv2.imread(str(ROOT / "test_images" / "am.jpg"))
h, w = ref.shape[:2]
base = Fretboard({"nut_6": (790, 167), "nut_1": (790, 347), "far_6": (27, 213), "far_1": (39, 454)},
                 far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
res = make_landmarker(num_hands=1).detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                   data=cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)))
lms = np.array([[q.x * w, q.y * h] for q in res.hand_landmarks[0]], dtype=np.float32)
kps = SIFT.detect(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), fretboard_mask(base, ref.shape, lms))
pil = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil)
for k in kps:
    x, y = k.pt
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 214, 0), outline=(0, 0, 0), width=2)
W, H = pil.size
bar = 70
canvas = Image.new("RGB", (W, H + bar), (122, 28, 32))
canvas.paste(pil, (0, 0))
dr = ImageDraw.Draw(canvas)
f = ImageFont.truetype(FONT, 36)
cap = "追蹤指板上的紋理（黃點：琴格、弦、琴枕），每個畫面更新指板位置"
dr.text(((W - dr.textlength(cap, font=f)) / 2, H + 14), cap, font=f, fill="white")
canvas.save(RES / "slide_tracking.jpg", quality=90)
print("ok", len(kps), "features")
