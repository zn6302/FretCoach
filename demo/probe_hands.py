"""Quick probe: run MediaPipe Hand Landmarker on still images and print fingertip pixels."""
import sys
from pathlib import Path

import cv2
import mediapipe as mp

MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
TIPS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}


def make_landmarker(num_hands=2, min_conf=0.3):
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


def main(paths):
    lm = make_landmarker()
    for p in paths:
        bgr = cv2.imread(p)
        h, w = bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = lm.detect(img)
        print(f"== {Path(p).name} ({w}x{h}): {len(res.hand_landmarks)} hand(s)")
        for hi, hand in enumerate(res.hand_landmarks):
            side = res.handedness[hi][0]
            tips = ", ".join(f"{n}=({hand[i].x*w:.0f},{hand[i].y*h:.0f})" for i, n in TIPS.items())
            print(f"   hand {hi} {side.category_name} {side.score:.2f}: {tips}")


if __name__ == "__main__":
    main(sys.argv[1:])
