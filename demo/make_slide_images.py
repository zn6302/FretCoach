"""Presentation-sized before/after images for the slides (large markers, Chinese captions)."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from annotate_image import detect_landmarks
from fretcoach import FINGERS, Fretboard, check_chord

ROOT = Path(__file__).resolve().parent.parent
FONT = "/Library/Fonts/Arial Unicode.ttf"
CORNERS = {"nut_6": (790, 167), "nut_1": (790, 347), "far_6": (27, 213), "far_1": (39, 454)}
CROP = (300, 110, 900, 470)  # x0, y0, x1, y1 in the original photo
SCALE = 2
NAMES = {"index": "食指", "middle": "中指", "ring": "無名指"}
GREEN, RED = (46, 160, 67), (220, 38, 38)


def render(k, caption, out):
    bgr = cv2.imread(str(ROOT / "test_images" / "am.jpg"))
    board = Fretboard(CORNERS, far_fret=5, scale_mm=650.0, span_nut_mm=43.0)
    lms = detect_landmarks(bgr)
    res = check_chord(board, lms, "Am", k)

    img = bgr.copy()
    board.draw(img, max_fret=5)
    x0, y0, x1, y1 = CROP
    img = cv2.resize(img[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_lab = ImageFont.truetype(FONT, 30)
    f_cap = ImageFont.truetype(FONT, 40)

    def P(p):
        return ((p[0] - x0) * SCALE, (p[1] - y0) * SCALE)

    for r in res:
        cx, cy = P(r["point"])
        ok = r["status"] == "ok"
        col = GREEN if ok else RED
        if ok:
            d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=col, outline="white", width=4)
        else:
            for a, b in (((-16, -16), (16, 16)), ((-16, 16), (16, -16))):
                d.line([cx + a[0], cy + a[1], cx + b[0], cy + b[1]], fill="white", width=11)
                d.line([cx + a[0], cy + a[1], cx + b[0], cy + b[1]], fill=col, width=7)
        s, f = r["got"]
        ws, wf = r["want"]
        label = f"{NAMES[r['finger']]} 第{s}弦" + ("" if ok else f"（應為第{ws}弦）")
        tx, ty = cx + 24, cy - 18
        if tx + d.textlength(label, font=f_lab) > pil.size[0] - 12:  # would be cut off: put it on the left
            tx = cx - 24 - d.textlength(label, font=f_lab)
            ty = cy + 22
        box = d.textbbox((tx, ty), label, font=f_lab)
        d.rounded_rectangle([box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5], 8, fill=(255, 255, 255))
        d.text((tx, ty), label, font=f_lab, fill=col)

    W, H = pil.size
    bar = 74
    canvas = Image.new("RGB", (W, H + bar), (122, 28, 32))
    canvas.paste(pil, (0, 0))
    cd = ImageDraw.Draw(canvas)
    tw = cd.textlength(caption, font=f_cap)
    cd.text(((W - tw) / 2, H + 13), caption, font=f_cap, fill="white")
    canvas.save(out)
    print(out, canvas.size, [(r["finger"], r["status"]) for r in res])


if __name__ == "__main__":
    render(0.0, "修正前：指尖偏約 5 mm，三指都判到隔壁弦", ROOT / "results" / "slide_before.png")
    render(0.4, "修正後：三指全部判對", ROOT / "results" / "slide_after.png")
