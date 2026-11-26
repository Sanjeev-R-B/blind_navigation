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

## Phase 1 — Core Detection: Camera + YOLO (Desktop ✅ · Android validation pending 🔄)

> Goal: Get reliable real-time object detection running on the phone at ≥12–15 FPS — the non-negotiable foundation.

### Task status

| # | Task | Status |
|---|------|--------|
| 01 | Python + OpenCV live camera feed setup | ✅ Done |
| 02 | YOLOv8n model integration via ONNX Runtime | ✅ Done |
| 03 | Bounding boxes + class labels overlay on frames | ✅ Done |
| 04 | FPS measurement loop + profiling | ✅ Done |
| 05 | Input resolution tuning: 320×320 → 416×416 | ✅ Done |
| 06 | Native Android camera via CameraX API | 🔄 App builds (`assembleDebug` succeeds); on-device FPS check pending |
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

**Demo class whitelist (added during Phase 2)**
Indoor false positives (for example a wardrobe detected as a refrigerator) made the demo unreliable, so `detect()` now only returns `DEMO_CLASSES`: person, chair, book, cell phone, backpack, bicycle, motorcycle, car, truck, dining table. Edit the set in `yolo_detector.py` to widen it.

**Optional low-confidence output (added for tracking)**
`detect(frame, include_low=True)` returns boxes down to 0.10 confidence. The tracker uses them to keep an existing ID alive when a score briefly dips. The default call is unchanged (0.25).

### Exit criteria

| Criteria | Status |
|----------|--------|
| Live detection ≥ 12 FPS on actual target phone | ⏳ Desktop: ✅ 30.6 FPS — Android: app builds, on-device FPS measurement pending (task 06) |
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

> **Status update:** an Android project now exists under `android/` (Kotlin `CameraActivity`, CameraX, ONNX Runtime for Android, Chaquopy for the Python layer). `gradlew assembleDebug` succeeds and produces a ~185 MB debug APK. **Still to do:** measure FPS on the target phone (POCO C65, Android 15) against the ≥12 FPS gate. The Phase 2 logic (distance, zones, speech) and the tracker are desktop-only so far and still need porting.

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

> **Status update:** not started. Desktop checks so far covered person, chair, book, cell phone and backpack only.

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

## Phase 2 — Distance Estimation + Basic Alerts (Desktop ✅ · Depth fusion ⚠️ deferred to device)

> Goal: Tell the user what is ahead and how far away it is, by voice and on screen, from a single RGB camera.

### Task status

| # | Task | Status |
|---|------|--------|
| 01 | Focal length calibration routine (A4 sheet at 1 m) | ✅ Done |
| 02 | Known-heights lookup table | ✅ Done (navigation / demo classes) |
| 03 | Pinhole formula: distance = (focal × height) / pixels | ✅ Done |
| 04 | Depth Anything v2 Small conversion | ✅ Done as ONNX (no TFLite — `tflite-runtime` has no Windows build) |
| 05 | Fusion: pinhole for known objects, depth fallback | ⚠️ Code written, not real-time on CPU (see limitations) |
| 06 | Clipping / too-close detection at frame bottom | ✅ Done |
| 07 | Non-blocking TTS thread with priority queue | ✅ Done (Windows SAPI) |

### Tech stack

| Tool | Role |
|------|------|
| Pinhole camera model | Distance from bounding-box height |
| Depth Anything v2 Small (ONNX Runtime) | Dense depth map, fallback for unknown objects |
| Windows SAPI via a persistent PowerShell process | Offline text-to-speech |
| Python threading + queue | Non-blocking speech |
| OpenCV | Zone indicator and live dashboard |
| NumPy | Maths |

### What was built

**Calibration (`distance/calibration.py`).** Hold an A4 sheet (0.297 m tall) at exactly 1 m, press SPACE to capture, click its top and bottom edge, and the script computes `focal_length = pixel_height × 1.0 / 0.297` and saves it to `calibration.json`. The first A4 run gave 888.89 px. The camera was later recalibrated against a seated person at 1 m for the laptop demo, and `known_heights.json` uses a head-height value for `person` (0.23 m) to match.

**Pinhole estimator (`distance/pinhole_estimator.py`).** Applies the formula to every box, adds a `left / centre / right` zone from the box centre, clamps results to 0.1–10 m, and marks boxes under 20 px tall as unreliable. Results are sorted nearest first.

**Clip detector (`utils/clip_detector.py`).** When an object is so close that its box touches the bottom of the frame, the pinhole formula stops being trustworthy. The "too close" signal now needs both conditions: box bottom ≥ 99% of the frame height **and** estimated distance below 0.5 m (`CLIP_MAX_DISTANCE_M`). The distance condition was added because a seated user's box always touches the bottom edge.

**Speech engine (`audio/tts_engine.py`).** One long-lived PowerShell process hosts the Windows speech synthesizer. Text is sent over stdin and the worker waits for a `done` reply, so sentences never overlap and the main loop is never blocked. A one-slot queue drops stale normal messages and a priority message flushes the queue. The engine waits for a `ready` signal at startup so the slow first load does not delay the first alert.

**Alert formatter (`audio/alert_formatter.py`).** Turns estimates into spoken text and decides when to speak:

| Situation | Behaviour |
|---|---|
| New object | Speak immediately |
| Moves into a closer zone | Speak immediately |
| Distance changes by ≥ 0.6 m | Speak |
| Still in DANGER, nothing changed | Repeat every 3.5 s (priority) |
| CAUTION / SAFE, nothing changed | Stay silent |
| Not seen for 3 s, or beyond 3 m | Forget it, so it alerts again as new |
| Clip alert | Own 5 s cooldown per label |

Wording follows the zone ("Person, very close." / "Person, 1.2 metres." / "Person detected at 2.5 metres."), vehicles use stronger phrasing, and a risk score (distance, class weight, left/centre/right position) orders alerts when several fire together.

**Live dashboard (`run_phase2.py`).** Split window: camera feed with boxes, distances and a colour border on the left; on the right a panel with FPS, an object list with distance and risk level, and a rule-based scene summary refreshed every 4 s.

### Zones (tuned for a laptop webcam demo)

| Distance | Zone | Colour |
|---|---|---|
| ≤ 0.8 m | DANGER | Red |
| 0.8 – 1.5 m | CAUTION | Orange |
| 1.5 – 3.0 m | SAFE | Green |
| > 3.0 m | No indicator / no alert | — |

For walking use outdoors, raise these thresholds. They are constants at the top of `run_phase2.py` and `alert_formatter.py`.

### Folder structure

```
blind-nav/
└── phase2_distance/
    ├── distance/
    │   ├── calibration.py          # Focal length calibration (A4 sheet at 1 m)
    │   ├── pinhole_estimator.py    # distance = (focal × real height) / pixel height
    │   ├── calibration.json        # Saved focal length
    │   └── known_heights.json      # Real-world heights per class (metres)
    ├── depth/
    │   ├── depth_anything.py       # Depth Anything v2 Small ONNX wrapper
    │   ├── depth_fusion.py         # Pinhole + depth fusion, background thread
    │   └── models/
    │       └── depth_v2_small.onnx # ~94 MB — NOT committed (covered by models/ in .gitignore)
    ├── audio/
    │   ├── tts_engine.py           # Non-blocking TTS (persistent PowerShell SAPI process)
    │   └── alert_formatter.py      # When and what to speak
    ├── utils/
    │   └── clip_detector.py        # Too-close detection
    ├── tracking/                   # See "Tracking" section below
    │   ├── tracker.py
    │   └── events.py
    └── run_phase2.py               # Entry point: detection → tracking → distance → alerts → dashboard
```

### How to run

```bash
venv\Scripts\activate

# One-time, optional: recalibrate the focal length
python phase2_distance/distance/calibration.py

# Full Phase 2 pipeline (Q to quit)
python phase2_distance/run_phase2.py

# Depth model on its own (optional, ~1 FPS on CPU)
python phase2_distance/depth/depth_anything.py
```

No new pip packages are needed beyond the Phase 1 list. Download the depth model once:

```bash
python -c "import os,urllib.request; os.makedirs('phase2_distance/depth/models',exist_ok=True); urllib.request.urlretrieve('https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx','phase2_distance/depth/models/depth_v2_small.onnx')"
```

### Problems hit and how they were solved

| Problem | Cause | Fix |
|---|---|---|
| Only one of four test sentences was spoken | `pyttsx3` on Windows drops queued messages across threads | Replaced with Windows SAPI called through PowerShell |
| 4–9 s delay before urgent speech | Every sentence started a new PowerShell process | One persistent PowerShell process, plus a ready-wait at startup |
| "Very close, stop" repeated for everything | A seated person's box always touches the bottom of the frame | Clip alert now also needs distance < 0.5 m, plus a per-label cooldown |
| Silence for 40 s with a person in view | Alert state only compared distance, so a still person was suppressed forever | Zone-change, danger-repeat and forget rules (table above) |
| Wardrobe detected as a refrigerator | COCO false positive | Class whitelist (`DEMO_CLASSES`) |
| Distance read 3.9 m at 0.8 m | Box covers the upper body only, calibration assumed full body | Recalibrated on a seated person at 1 m |

### Depth Anything v2 — what was tried and why it is deferred

Depth Anything v2 Small (ONNX, 518×518 input) produces a correct depth map: near objects bright, far objects dark. On a CPU one inference takes 700–950 ms, so running it next to YOLO does not reach real time:

| Approach | Result |
|---|---|
| Depth on every frame, main thread | ~1 FPS |
| Depth every 10th frame, main thread | 4–7 FPS |
| Depth in a background thread, continuous | 4–6 FPS |
| Background thread, 2 s pause between runs | 4–6 FPS |
| Input reduced from 518 to 224 | Still below 12 FPS |

Both models compete for the same CPU cores, and Python's GIL limits parallelism between them. This is a hardware limit and not a code bug. The desktop demo therefore runs the pinhole pipeline only (22–30 FPS). `depth_anything.py` and `depth_fusion.py` are complete and ready to be wired in once a faster execution path exists.

**Planned fix (untested):** on the phone, run YOLO on the CPU and the depth model through ONNX Runtime's NNAPI execution provider so the two do not compete for the same cores. Alternatively use a smaller or quantized depth model. The execution-provider change is one line (`providers=["NNAPIExecutionProvider", "CPUExecutionProvider"]`), but speed on the POCO C65 has to be measured before relying on it.

### Exit criteria

| Criteria | Status |
|----------|--------|
| ≤ 15% distance error in the 1–4 m range | ⏳ Looks right at laptop distances; formal tape-measure test (1, 2, 3, 4 m) still to do |
| Works in real indoor and outdoor lighting | ⏳ Indoor only so far |
| Non-blocking voice alerts | ✅ |
| DANGER / CAUTION / SAFE indicator | ✅ |
| Clipping detection | ✅ |
| Depth model integrated | ✅ Code complete, deferred to device |
| Depth fusion real-time on desktop | ❌ CPU limit |

### Known limitations

- Speech uses Windows SAPI, so it is Windows-only; the Android build needs the platform TTS.
- Distances depend on one known height per class. A person whose full body is visible and a person seated at a desk are measured against the same height (see calibration note above).
- Alert state is keyed by class label, so two people in view share one record. This is replaced by track IDs (see Objective 4).
- `door`, `stairs`, `pole` and `wall` are not COCO classes and are not detected (Task 07, Option B).

---

## Tracking — first of the new objectives (Objective 1 🔄)

> Goal: give every object a stable ID so the system can reason about motion and stop repeating itself. Built after Phase 2 and described in `phase1_obj1.docx`.

### Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Detector can return low-confidence boxes (≥ 0.10) | ✅ Verified on webcam |
| 2 | Tracker assigns persistent IDs, drawn as `#id` on boxes | ✅ Verified on webcam |
| 3 | ENTER / EXIT events with exit reason | ✅ Verified on webcam |
| 4 | Track registry with distance history, distance-gated events | 🔄 Implemented and tested on simulated data; webcam check pending |

### Design

The tracker is pure NumPy (no torch or scipy), so the same logic can run on desktop and be ported to the phone. It follows the BoT-SORT / ByteTrack idea; `track_buffer` is replaced by a time-based `max_age_s` because FPS varies on mobile.

| Setting | Value |
|---|---|
| `track_high_thresh` | 0.25 |
| `track_low_thresh` | 0.10 |
| `new_track_thresh` | 0.25 |
| `match_iou` (stage 1 / stage 2) | 0.30 / 0.50 |
| `n_init` (frames to confirm) | 3 |
| `max_age_s` (lost track memory) | 1.5 s |
| Camera-motion compensation (`gmc_method`) | off — turn on only if ID switches appear |
| Re-identification (`with_reid`) | off — turn on only if ID switches appear |

Matching works in two stages. Stage 1 matches high-confidence boxes to all tracks by IoU. Stage 2 matches the leftover low-confidence boxes to the remaining non-lost tracks. New tracks start only from unmatched high-confidence boxes. A Kalman filter on `[cx, cy, w, h]` predicts each box between frames.

**Lifecycle:** `TENTATIVE → CONFIRMED → LOST → deleted`. A tentative track that misses a frame is deleted silently. A lost track that is re-matched within `max_age_s` keeps its ID.

**Events (`tracking/events.py`).** ENTER fires once when a track becomes CONFIRMED (held back until the object is within 3 m). EXIT fires once when a confirmed track is deleted, tagged `left_frame` if its last box touched the left, right or top border, otherwise `lost`. The bottom border is ignored because close objects always touch it.

```
{ event: ENTER | EXIT,
  track_id, class, confidence, distance,
  timestamp, frame_index, bbox, exit_reason (EXIT only) }
```

**Registry.** `tracker.registry` returns `{track_id: track}`. Each track holds its state, hit and streak counts, first and last seen time, last detected box, `class_votes` (summed confidence per class, so a flickering label does not change the track's class), and a rolling `history` of `(time, bbox, distance)`.

### Simulated test results

| Case | Result |
|---|---|
| Person tracked for 140 frames with a confidence dip and a 6-frame miss | Same ID throughout |
| New object | Confirmed after exactly 3 frames |
| Label flickering between two classes | No new ID |
| One-frame false positive | No track, no event |
| Two people crossing paths | No ID swap |
| Far object approaching | ENTER only once it is within 3 m |

### Known issues

- Matching uses box overlap only, so a fast lean or an overlap of two people can still switch an ID. Next options are camera-motion compensation and appearance re-identification.
- `left_frame` is decided only by the box touching a border. An object that sits against an edge for its whole life is tagged `left_frame` when it disappears.
- The registry does not yet store `last_seen_frame`.
- FPS dropped to about 15 in one run after the low-confidence boxes were enabled (it was 22–30 before). The cost is being measured per stage (detect vs tracker) before changing anything.

---

## Future work — what is left to achieve

Build order: **1 → 2 → 4 → 3**. Tracking feeds both the collision estimate and the alert cooldown; scene description is independent.

### Objective 1 — Object tracking (finish)
- [x] Configuration, lifecycle, ENTER / EXIT events, registry with history
- [ ] Verify step 4 on the webcam (distance in events, far objects silent)
- [ ] Add `last_seen_frame` to the registry
- [ ] Skip the 3-frame confirmation for objects already in the DANGER zone (the `fast_confirm` hook exists in `Tracker`, it only needs wiring)
- [ ] Make `left_frame` direction-aware using the last velocity before the track was lost
- [ ] Turn on camera-motion compensation (sparse optical flow) if ID switches show up on a handheld camera, and benchmark it on the phone
- [ ] Port the tracker to the Android build

### Objective 2 — Approach speed and time-to-collision
- [ ] `kinematics/ttc.py`: speed = (previous − current distance) ÷ Δt, using the median of the readings in the last ~0.5 s from `track.distance_window()`
- [ ] Ignore speeds under 0.1 m/s so noise never fakes a collision; TTC = distance ÷ speed
- [ ] Alerts such as "Car approaching fast, 1.3 seconds" instead of a bare distance
- [ ] Validate with a tape measure and a walking tester

### Objective 4 — Smart alert cooldown per object
- [ ] Switch `alert_formatter.py` state from class label to track ID
- [ ] New ID → alert; moved ≥ 0.5 m or changed zone → alert; unchanged → stay silent; DANGER → repeat on a timer
- [ ] Closing speed above about 0.8 m/s overrides the cooldown
- [ ] Use ENTER / EXIT events in place of the "not seen for 3 s" rule

### Objective 3 — Scene description on demand
- [ ] Key press (later a voice command) captures a frame and encodes it as JPEG, then base64
- [ ] Send it to a vision language model (Claude Vision API) and speak the reply
- [ ] Pause obstacle alerts while describing and resume afterwards
- [ ] Offline fallback and caching of repeated scenes (Phase 4 of the roadmap)

### Carried over from Phases 1 and 2
- [ ] Task 06: measure FPS on the POCO C65 (gate: ≥ 12 FPS)
- [ ] Task 07: validate classes at 1–5 m; plan custom classes for door, stairs, pole, wall
- [ ] Formal distance accuracy test (target ≤ 15% error between 1 and 4 m), including outdoor light
- [ ] Port Phase 2 and tracking to the Android app, including platform TTS
- [ ] Depth fusion on the device (NNAPI or a smaller model), measured before it is enabled
- [ ] Replace the single head-height value for `person` with a rule that handles full-body and seated boxes

---

## Roadmap

- **Phase 1** — Core Detection: Camera + YOLO ✅ desktop · 🔄 Android on-device validation
- **Phase 2** — Distance estimation + basic alerts ✅ desktop · depth fusion deferred to the device
- **Phase 2.5** — Motion-aware alerts: tracking 🔄 → approach speed and TTC ⏳ → smart cooldown ⏳ → scene description ⏳
- **Phase 3** — Intelligent audio and priority system (threat scoring, three-tier priority queue, haptics)
- **Phase 4** — On-demand OCR and scene description
- **Phase 5** — Voice control and full integration (wake word, voice commands, GPS)