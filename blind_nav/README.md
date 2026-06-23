# blind_navigation
**Git Commands overview for contribution:**
🔧 Step 1: Clone the Repository
git clone https://github.com/Sanjeev-R-B/blind_navigation.git
cd blind_navigation

🌿 Step 2: Create a New Branch
git checkout -b feature-branch-name

✍️ Step 3: Make Changes & Commit
git add .
git commit -m "Describe your changes clearly"

⬆️ Step 4: Push Your Branch
git push origin feature-branch-name

🔁 Step 5: Create a Pull Request (PR)
Go to:
Blind Navigation Repo
Click Compare & Pull Request
Add proper description
Submit PR


# blind-nav

AI-powered navigation assistant for visually impaired users. This project uses real-time object detection to identify obstacles and navigation-relevant objects through a camera feed.

---

## Phase 1 — Core Detection: Camera + YOLO (In Progress 🔄)

> Goal: Get reliable real-time object detection running on the phone at ≥12–15 FPS — the non-negotiable foundation.

### Task status

| # | Task | Status |
|---|------|--------|
| 01 | Python + OpenCV live camera feed setup | ✅ Done |
| 02 | YOLOv8n model integration via ONNX Runtime | ✅ Done |
| 03 | Bounding boxes + class labels overlay on frames | ✅ Done |
| 04 | FPS measurement loop + profiling | ✅ Done |
| 05 | Input resolution tuning: 320×320 → 416×416 | ✅ Done |
| 06 | Native Android camera via CameraX API | ⏳ Pending |
| 07 | Validate: person, car, door, stairs, pole, wall | ⏳ Pending |

### Tech stack

| Tool | Role |
|------|------|
| YOLOv8n (nano) | Object detection model |
| ONNX Runtime | Cross-platform inference engine |
| OpenCV | Camera feed + frame rendering |
| Python / Chaquopy | Desktop pipeline + Android Python bridge |
| Kivy | UI framework (Android) |
| Android CameraX | Native Android camera integration |

### What was built (tasks 01–05)

A fully functional real-time object detection pipeline running on a laptop webcam using YOLOv8n via ONNX Runtime. The pipeline detects 80 COCO object classes, with special highlighting for navigation-relevant objects (people, vehicles, furniture, common indoor objects).

### Folder structure

```
blind-nav/
│
├── phase1_detection/
│   ├── camera/
│   │   ├── camera_feed.py          # Main entry point — webcam loop with detection + FPS overlay
│   │   └── camera_android.py       # CameraX bridge placeholder (Phase 2)
│   │
│   ├── detection/
│   │   ├── yolo_detector.py        # Clean detect(frame) interface, NAV_CLASSES filter
│   │   ├── onnx_inference.py       # ONNX Runtime session, pre/post-processing, NMS
│   │   └── bbox_renderer.py        # Draws color-coded bounding boxes and labels on frames
│   │
│   ├── utils/
│   │   ├── fps_counter.py          # Non-blocking FPS measurement utility
│   │   └── resolution_tuner.py     # Benchmarks multiple input resolutions, prints FPS summary
│   │
│   └── models/
│       └── yolov8n.onnx            # Downloaded model — NOT committed (see setup below)
│
├── .gitignore
├── requirements.txt
└── README.md
```

### How to run

**Prerequisites:** Python 3.10+, a webcam

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and export the YOLO model (one-time setup)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', dynamic=True, imgsz=640)"
# Move the generated yolov8n.onnx into phase1_detection/models/

# 4. Run the detection feed
python phase1_detection/camera/camera_feed.py
```

Press `Q` to quit.

### Dependencies

```
opencv-python
ultralytics
onnxruntime
numpy
```

### Key implementation decisions

**Input resolution — 416×416**
Benchmarked across four resolutions on a mid-range CPU:

| Resolution | FPS  |
|------------|------|
| 320×320    | 47.1 |
| 416×416    | 30.6 ← locked in |
| 480×480    | 24.1 |
| 640×640    | 13.7 |

416×416 was chosen because it delivers the best accuracy/speed balance and leaves ~18 FPS of budget for the depth estimation model in Phase 2 while staying well above the 12 FPS minimum target.

**Confidence threshold — 0.25**
The default 0.40 threshold missed small or flat-surface objects (books, phones). Lowering to 0.25 recovers these detections reliably for navigation use cases.

**NMS (Non-Maximum Suppression)**
Raw YOLO output produces duplicate boxes for the same object. NMS is applied via `cv2.dnn.NMSBoxes` with an IoU threshold of 0.45, keeping only the highest-confidence box per object.

**Dynamic ONNX export**
The model is exported with `dynamic=True` so it accepts variable input resolutions. A fixed-size export (default) only accepts 640×640 and prevents resolution tuning.

**NAV_CLASSES filter**
`yolo_detector.py` tags detections with `is_nav_relevant: True/False` based on a curated set of classes meaningful for blind navigation — vehicles, people, furniture, and common indoor objects. Non-nav objects are still detected but rendered in grey.

### Exit criteria

| Criteria | Status |
|----------|--------|
| Live detection ≥ 12 FPS on actual target phone | ⏳ Desktop: ✅ 30.6 FPS — Android: pending task 06 |
| Critical obstacles reliably detected at 1–5m range | ⏳ Pending task 07 (door, stairs, pole, wall) |
| Clean bounding boxes with NMS | ✅ |
| Nav-relevant classes visually highlighted | ✅ color-coded by class |
| Small/indoor objects detected | ✅ book, cell phone confirmed at conf=0.25 |

---

## Model setup

The ONNX model is not committed to this repo (binary files excluded via `.gitignore`).

To download and generate it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', dynamic=True, imgsz=640)"
```

Then move `yolov8n.onnx` into `phase1_detection/models/`. The `.pt` file can be deleted after export.

---

## .gitignore

```
models/
__pycache__/
*.pyc
*.onnx
*.tflite
.env
venv/
```

---

## Pending Tasks — Help Wanted 🙋

The desktop pipeline (tasks 01–05) is fully working. The two remaining tasks to close Phase 1 are below. Each has enough detail to pick up and implement independently.

---

### Task 06 — Native Android camera via CameraX API

**Goal:** Replace `camera_feed.py` (OpenCV webcam) with a native Android camera feed so the detection pipeline runs on-device on a phone.

**Approach — Chaquopy + Kivy + CameraX**

The roadmap uses Chaquopy (Python-in-Android) and Kivy (UI). The rough integration path:

1. **Set up Android Studio project with Chaquopy**
   - Create a new Android project in Android Studio
   - Add Chaquopy to `build.gradle` — it lets you run Python inside an Android app
   - Add `onnxruntime`, `numpy`, `opencv-python` as Chaquopy pip dependencies in `build.gradle`
   - Copy `phase1_detection/detection/` and `phase1_detection/utils/` into the Android assets

2. **Wire CameraX to deliver frames to Python**
   - Set up a `CameraX ImageAnalysis` use case in Kotlin/Java
   - In the `analyze(ImageProxy)` callback, convert the `ImageProxy` to a byte array
   - Pass the byte array across the Chaquopy bridge into Python as a numpy array
   - Run `detector.detect(frame)` exactly as in `camera_feed.py`

3. **Render results with Kivy**
   - Use a Kivy `Canvas` overlay to draw bounding boxes on top of the camera preview
   - Mirror the logic in `bbox_renderer.py` — same colors, same label format

4. **Target FPS:** ≥12 FPS on a mid-range Android phone (Snapdragon 600-series or equivalent)
   - Start with 416×416 input (already benchmarked as optimal)
   - If FPS is too low, drop to 320×320 — the dynamic ONNX model supports it

**Files to create:**
```
blind-nav/
└── phase1_detection/
    └── camera/
        └── camera_android.py    # Python side: receives frame bytes, runs detector, returns results
android/
├── app/
│   ├── build.gradle             # Chaquopy config + pip deps
│   └── src/main/
│       ├── java/.../
│       │   └── CameraActivity.kt  # CameraX setup + Chaquopy bridge
│       └── res/layout/
│           └── activity_main.xml
```

**Useful references:**
- Chaquopy docs: https://chaquo.com/chaquopy/doc/current/android.html
- CameraX ImageAnalysis: https://developer.android.com/training/camerax/analyze
- ONNX Runtime Android: https://onnxruntime.ai/docs/tutorials/mobile/

---

### Task 07 — Validate critical obstacle classes at 1–5m range

**Goal:** Confirm the detector reliably catches the objects that matter most for blind navigation — at real-world distances (1 to 5 metres), not just close up.

**The problem with COCO classes**

YOLOv8n is trained on COCO's 80 classes. The roadmap lists `door`, `stairs`, `pole`, and `wall` as targets — but none of these are COCO classes. Here's the recommended handling:

| Target class | COCO situation | Recommended approach |
|---|---|---|
| person | ✅ In COCO | Validate directly |
| car | ✅ In COCO | Validate directly |
| door | ❌ Not in COCO | Proxy: detect as part of `wall` context, or use a fine-tuned model |
| stairs | ❌ Not in COCO | Proxy: no good COCO match — needs custom class (see below) |
| pole | ❌ Not in COCO | Proxy: sometimes detected as `parking meter` or `fire hydrant` — unreliable |
| wall | ❌ Not in COCO | Not detectable — needs depth model (Phase 2) or custom class |

**Option A — Validate COCO classes only (fastest, do this first)**

Run a structured test session with the desktop pipeline (`camera_feed.py`) and log results:

```
Test grid:
- Object: person / car / bicycle / chair / dining table / bottle
- Distances: 1m, 2m, 3m, 5m
- Lighting: good light / low light
- Angles: frontal / side / partial occlusion

Pass criteria: detected in ≥ 8/10 frames at conf ≥ 0.25
```

Create `phase1_detection/utils/validation_logger.py` — logs each detection (label, confidence, distance estimate) to a CSV for review.

**Option B — Add custom classes for door/stairs/pole (proper fix)**

Fine-tune YOLOv8n on a small custom dataset:

1. Collect ~200 images per class (door, stairs, pole) — use [Roboflow](https://roboflow.com) to label them
2. Fine-tune: `yolo train model=yolov8n.pt data=custom.yaml epochs=50 imgsz=416`
3. Export to ONNX with `dynamic=True`
4. Drop the new model into `phase1_detection/models/` and update `NAV_CLASSES` in `yolo_detector.py`

**Recommended path:** Do Option A first to close Phase 1 quickly, then add Option B as a Phase 1.5 / early Phase 2 improvement before Android validation.

**File to create:**
```
phase1_detection/utils/validation_logger.py   # logs detections to CSV with timestamp + confidence
```

---

## Roadmap

- **Phase 1** — Core Detection: Camera + YOLO 🔄 (tasks 01–05 done, 06–07 pending)
- **Phase 2** — Distance estimation (depth model, spatial audio cues)
- **Phase 3** — Android port (CameraX, TFLite)
- **Phase 4** — Navigation assistant (path guidance, voice output)

