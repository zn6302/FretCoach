"""FretCoach system architecture diagram (16:9, for the report and the slides)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

font_manager.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams["font.family"] = "Arial Unicode MS"

import sys
SLIDE = "--slide" in sys.argv  # slide version: no title, the slide has its own
OUT = Path(__file__).resolve().parent.parent / "results" / ("architecture_slide.png" if SLIDE else "architecture.png")
RED, INK, GREY, TINT, LANE = "#7A1C20", "#2B2B2B", "#6B6B6B", "#F6E9EA", "#FAF7F7"
W, H = 2.3, 1.35
BOLD = font_manager.FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

fig, ax = plt.subplots(figsize=(16, 7.55) if SLIDE else (16, 9), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.55 if SLIDE else 9)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, title, body, accent=False, dashed=False, w=W):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - H / 2), w, H, boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=TINT if accent else "white", edgecolor=RED, linewidth=3 if accent else 1.8,
        linestyle=(0, (5, 3)) if dashed else "solid", zorder=3))
    ax.text(cx, cy + 0.25, title, ha="center", va="center", fontsize=17, fontproperties=BOLD, color=RED, zorder=4)
    ax.text(cx, cy - 0.25, body, ha="center", va="center", fontsize=12.5, color=INK, zorder=4, linespacing=1.4)


def arrow(p, q, style="-|>", color=RED, rad=0.0, lw=2.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=22, color=color, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=0, shrinkB=0))


def note(cx, y, text):
    ax.text(cx, y, text, ha="center", va="top", fontsize=12, color=GREY, style="italic")


# title
if not SLIDE:
    ax.text(0.5, 8.45, "FretCoach 系統架構", fontsize=26, fontproperties=BOLD, color=RED, va="center")
if not SLIDE:
    ax.text(0.5, 7.9, "每個畫面約 3 ms 完成「找手指 → 換算弦格 → 比對 → 回饋」", fontsize=14, color=GREY, va="center")

# lanes
ax.add_patch(FancyBboxPatch((0.2, 5.55), 15.6, 1.85, boxstyle="round,pad=0,rounding_size=0.2",
                            facecolor=LANE, edgecolor="none", zorder=0))
ax.text(0.45, 7.2, "每個畫面：找指板", fontsize=12, color=GREY, va="center")
ax.add_patch(FancyBboxPatch((0.2, 2.45), 15.6, 2.85, boxstyle="round,pad=0,rounding_size=0.2",
                            facecolor=LANE, edgecolor="none", zorder=0))
ax.text(0.45, 5.1, "每個畫面：找手指", fontsize=12, color=GREY, va="center")

ROW, TOP = 3.75, 6.45
xs = [1.55, 4.15, 6.75, 9.35, 11.95, 14.55]

box(xs[0], ROW, "鏡頭畫面", "webcam・30 fps")
box(xs[1], ROW, "手部追蹤", "MediaPipe\n21 個關節點")
box(xs[2], ROW, "接觸點修正", "TIP + 0.4 ×\n(TIP − DIP)", accent=True)
box(xs[3], ROW, "座標轉換", "指尖 →（弦, 格）")
box(xs[4], ROW, "平滑與停穩", "平均 10 個畫面\n停穩 0.6 秒才判斷", accent=True)
box(xs[5], ROW, "和弦比對", "對照 8 個\n初學和弦")
TW = 3.1
box(xs[3], TOP, "指板定位", "追蹤指板上的紋理\n每格更新 homography", w=TW, accent=True)
ax.text(xs[3] + TW / 2 + 0.25, TOP + 0.2, "只校正一次的話，吉他移動約 2 mm\n或琴頸擺動 0.3° 就會判錯", fontsize=12, color=GREY, va="center")
ax.text(xs[3] + TW / 2 + 0.25, TOP - 0.5, "每格追蹤：誤差 < 0.6 mm（模擬）\n不用在吉他上貼標記", fontsize=12, color=GREY, va="center", style="italic")

# main flow
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + W / 2, ROW), (b - W / 2, ROW))
# calibration: one frame goes up, the fretboard model comes down into the coordinate step
arrow((xs[0], ROW + H / 2), (xs[3] - TW / 2, TOP), rad=-0.18, color=GREY, lw=1.8)
ax.text(3.1, 6.55, "同一個畫面", fontsize=12, color=GREY, ha="center")
arrow((xs[3], TOP - H / 2), (xs[3], ROW + H / 2))
ax.text(xs[3] + 0.12, 5.1, "指板座標（mm）", fontsize=12, color=GREY, va="center")

# feedback + loop back to the user
FB = 1.15
box(xs[5], FB, "回饋介面", "綠點＝對、紅叉＝錯\n一次只提示一句")
arrow((xs[5], ROW - H / 2), (xs[5], FB + H / 2))
ax.add_patch(FancyArrowPatch((xs[5] - W / 2, FB), (xs[0], FB), arrowstyle="-", color=RED, linewidth=2.2, zorder=2))
arrow((xs[0], FB), (xs[0], ROW - H / 2))
ax.text((xs[0] + xs[5] - W / 2) / 2, FB + 0.18, "使用者看提示調整手指，畫面即時更新", fontsize=13.5, color=RED,
        ha="center", va="bottom")

# evidence from the feasibility study
note(xs[1], ROW - H / 2 - 0.12, "每格約 3 ms")
note(xs[2], ROW - H / 2 - 0.12, "修正前偏約 5 mm\n修正後三指全對")
note(xs[4], ROW - H / 2 - 0.12, "準確率 58% → 94%\n避免 Midas touch")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.25, facecolor="white")
print(OUT)
