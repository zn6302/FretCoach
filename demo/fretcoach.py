"""FretCoach core: fretboard geometry, fingertip -> (string, fret) mapping, chord checking.

Coordinate conventions
----------------------
World (fretboard) coordinates are in millimetres:
  u = distance along the neck from the nut toward the body (u = 0 at the nut)
  v = distance across the neck, measured from the 6th (low E) string
Strings are numbered like guitarists do: 6 = low E ... 1 = high E.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# MediaPipe hand landmark indices: (tip, dip) per fretting finger.
FINGERS = {
    "index": (8, 7),
    "middle": (12, 11),
    "ring": (16, 15),
    "pinky": (20, 19),
}

# Open-position beginner chords: finger -> (string, fret).
CHORDS = {
    "C": {"index": (2, 1), "middle": (4, 2), "ring": (5, 3)},
    "G": {"index": (5, 2), "middle": (6, 3), "ring": (1, 3)},
    "D": {"index": (3, 2), "middle": (1, 2), "ring": (2, 3)},
    "A": {"index": (4, 2), "middle": (3, 2), "ring": (2, 2)},
    "E": {"index": (3, 1), "middle": (5, 2), "ring": (4, 2)},
    "Am": {"index": (2, 1), "middle": (4, 2), "ring": (3, 2)},
    "Em": {"middle": (5, 2), "ring": (4, 2)},
    "Dm": {"index": (1, 1), "middle": (3, 2), "ring": (2, 3)},
}


def fret_distance(n: int | np.ndarray, scale_mm: float) -> np.ndarray:
    """Distance from the nut to fret wire n under 12-tone equal temperament."""
    return scale_mm * (1.0 - 2.0 ** (-np.asarray(n, dtype=float) / 12.0))


@dataclass
class Fretboard:
    """Fretboard model calibrated from four image points.

    corners: image pixels of the 6th and 1st strings at the nut and at fret wire `far_fret`:
        {"nut_6": (x, y), "nut_1": (x, y), "far_6": (x, y), "far_1": (x, y)}
    """

    corners: dict
    far_fret: int = 5
    scale_mm: float = 650.0          # 650 classical, ~645-648 steel-string
    span_nut_mm: float = 43.0        # 6th-to-1st string distance at the nut
    span_per_fret_mm: float = 0.35   # string span widens toward the body
    H: np.ndarray = field(init=False)
    H_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        u_far = float(fret_distance(self.far_fret, self.scale_mm))
        span_far = self.span(u_far)
        img = np.float32([self.corners[k] for k in ("nut_6", "nut_1", "far_6", "far_1")])
        world = np.float32([
            [0.0, 0.0],
            [0.0, self.span_nut_mm],
            [u_far, -(span_far - self.span_nut_mm) / 2],
            [u_far, self.span_nut_mm + (span_far - self.span_nut_mm) / 2],
        ])
        self.H = cv2.getPerspectiveTransform(img, world)
        self.H_inv = np.linalg.inv(self.H)

    # ---- geometry ------------------------------------------------------
    def span(self, u: float) -> float:
        frets_equiv = u / (self.scale_mm / 17.817)  # rough mm per fret near the nut
        return self.span_nut_mm + self.span_per_fret_mm * frets_equiv

    def string_v(self, string: int, u: float) -> float:
        """v (mm) of a string (6..1) at distance u along the neck."""
        span = self.span(u)
        offset = (span - self.span_nut_mm) / 2
        return -offset + (6 - string) * span / 5.0

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def locate(self, pt_img) -> dict:
        """Map one image point to string/fret, with margins in mm."""
        u, v = self.to_world([pt_img])[0]
        vs = np.array([self.string_v(s, u) for s in range(6, 0, -1)])  # strings 6..1
        i = int(np.argmin(np.abs(vs - v)))
        string = 6 - i
        spacing = self.span(u) / 5.0
        frets = fret_distance(np.arange(0, 13), self.scale_mm)
        fret = int(np.searchsorted(frets, u))  # 0 = behind the nut, k = between wire k-1 and k
        return {
            "u_mm": float(u),
            "v_mm": float(v),
            "string": string,
            "fret": fret,
            "string_offset_mm": float(v - vs[i]),          # signed distance to the matched string
            "string_margin_mm": float(spacing / 2 - abs(v - vs[i])),  # >0 = safely inside
            "spacing_mm": float(spacing),
        }

    def mm_per_px(self, pt_img) -> float:
        """Local scale: how many mm one image pixel covers near this point (across the strings)."""
        p = np.float32(pt_img)
        w0, w1 = self.to_world([p, p + np.float32([0, 1])])
        w2 = self.to_world([p + np.float32([1, 0])])[0]
        return float((np.linalg.norm(w1 - w0) + np.linalg.norm(w2 - w0)) / 2)

    # ---- drawing -------------------------------------------------------
    def draw(self, img, max_fret: int | None = None, color=(0, 200, 255)):
        max_fret = max_fret or self.far_fret
        for n in range(0, max_fret + 1):
            u = float(fret_distance(n, self.scale_mm))
            a = self.to_image([[u, self.string_v(6, u) - 3], [u, self.string_v(1, u) + 3]])
            cv2.line(img, tuple(map(int, a[0])), tuple(map(int, a[1])), color, 2 if n == 0 else 1, cv2.LINE_AA)
            if n > 0:
                cv2.putText(img, str(n), tuple(map(int, a[0] + [-4, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        u_end = float(fret_distance(max_fret, self.scale_mm))
        for s in range(6, 0, -1):
            pts = self.to_image([[0, self.string_v(s, 0)], [u_end, self.string_v(s, u_end)]])
            cv2.line(img, tuple(map(int, pts[0])), tuple(map(int, pts[1])), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"s{s}", tuple(map(int, pts[0] + [6, 4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)


def contact_point(landmarks_px: np.ndarray, finger: str, k: float) -> np.ndarray:
    """Estimated string-contact point: extend past the TIP landmark along DIP->TIP by factor k.

    k = 0 uses the raw MediaPipe fingertip. The raw landmark sits inside the fingertip,
    so the pressing pad is usually a little further along the finger.
    """
    tip, dip = FINGERS[finger]
    t, d = landmarks_px[tip], landmarks_px[dip]
    return t + k * (t - d)


def check_chord(board: Fretboard, landmarks_px: np.ndarray, chord: str, k: float = 0.0) -> list[dict]:
    """Compare each required finger with the chord template."""
    out = []
    for finger, (want_s, want_f) in CHORDS[chord].items():
        p = contact_point(landmarks_px, finger, k)
        loc = board.locate(p)
        if loc["string"] == want_s and loc["fret"] == want_f:
            status, hint = "ok", ""
        elif loc["fret"] != want_f:
            status, hint = "wrong_fret", f"move to fret {want_f}"
        else:
            status = "wrong_string"
            # string numbers grow toward the 6th string
            hint = f"move toward string {want_s} ({'bass' if want_s > loc['string'] else 'treble'} side)"
        out.append({"finger": finger, "want": (want_s, want_f), "got": (loc["string"], loc["fret"]),
                    "status": status, "hint": hint, "point": p, **loc})
    return out
