"""Retry detection with lower confidence and padding/downscaling, to test the 'palm must be visible' hypothesis."""
import sys
import cv2, mediapipe as mp
from probe_hands import make_landmarker

lm = make_landmarker(num_hands=2, min_conf=0.05)
for p in sys.argv[1:]:
    bgr = cv2.imread(p)
    for pad in (0, 0.25, 0.5, 1.0):
        h, w = bgr.shape[:2]
        b = int(max(h, w) * pad)
        img = cv2.copyMakeBorder(bgr, b, b, b, b, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for flip in (False, True):
            im = cv2.flip(img, 1) if flip else img
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            print(p.split('/')[-1], 'pad', pad, 'flip', flip, '->', len(res.hand_landmarks))
