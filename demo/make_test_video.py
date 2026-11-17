"""Build a short synthetic 'webcam' clip from a still photo (sensor noise, flicker, 1-px shake)."""
import sys
import cv2
import numpy as np

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(1)
img = cv2.imread(src)
h, w = img.shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for _ in range(n):
    M = np.float32([[1, 0, rng.uniform(-1, 1)], [0, 1, rng.uniform(-1, 1)]])
    f = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    f = f * rng.uniform(0.93, 1.07) + rng.normal(0, 5, f.shape)
    vw.write(np.clip(f, 0, 255).astype(np.uint8))
vw.release()
