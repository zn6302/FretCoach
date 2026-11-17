# FretCoach demo

## 安裝（第一次）

實驗都在 Python 3.11 跑過。用 uv 建立獨立環境，不會動到系統的 Python：

```bash
cd demo
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models && curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## 用 webcam 即時測試（拿自己的吉他）

```bash
.venv/bin/python webcam_demo.py --guitar steel --chord Am      # 民謠吉他
.venv/bin/python webcam_demo.py --guitar classical --chord C   # 古典吉他
```

1. 鏡頭放在你左前方約 40–60 cm、略高於琴頸，**畫面要拍到手掌和手腕**。
2. 按 `c` 校正：畫面會定格，依序點四下：第 6 弦在琴枕處 → 第 1 弦在琴枕處 → 第 6 弦在第 5 格琴格 → 第 1 弦在第 5 格琴格。校正結果存在 `calib.json`，下次自動載入。
3. 按 `1`–`8` 切換和弦（C G D A E Am Em Dm），`m` 鏡像，`q` 離開。
4. 手指停穩 0.6 秒後才會判斷：綠點 = 對，紅色叉 = 錯（會連到白色空心圈的目標位置），灰點 = 還在移動。

校正後吉他不能移動；移動了就再按一次 `c`。

## 重現研究報告裡的實驗

```bash
.venv/bin/python annotate_image.py ../test_images/am.jpg \
  --corners 790,167,790,347,27,213,39,454 --chord Am --k 0 0.4 --out ../results/am
.venv/bin/python experiments.py          # E1、E2、E3，輸出 ../results/*.png 與 summary.json
.venv/bin/python make_test_video.py ../test_images/am.jpg ../results/am_test_clip.mp4 90
.venv/bin/python webcam_demo.py --video ../results/am_test_clip.mp4 --guitar classical --chord Am \
  --corners 790,167,790,347,27,213,39,454 --headless
```

## 檔案

| 檔案 | 內容 |
| --- | --- |
| `fretcoach.py` | 核心：指板模型（homography + 十二平均律）、指尖→（弦, 格）、和弦比對 |
| `webcam_demo.py` | 即時 demo：校正、多格平均、dwell、回饋畫面 |
| `annotate_image.py` | 對單張照片跑完整流程並輸出疊圖 |
| `experiments.py` | 可行性實驗 E1–E3 |
| `probe_hands.py` | 測試 MediaPipe 能否在照片中找到手 |
| `make_test_video.py` | 由照片產生模擬 webcam 的測試影片 |
