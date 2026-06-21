"""
camera_android.py — Phase 1 + Phase 2 pipeline for on-device blind navigation.

Architecture:
  Kotlin   : CameraX frame capture → ONNX inference (Java ORT API) → raw float tensor
  Python   : postprocess → distance estimate → clipping check → alert text generation
  Kotlin   : Android TextToSpeech API → spoken audio  |  OverlayView → visual overlay

No pip packages needed beyond numpy. TTS is handled by Kotlin/Android.
"""
import numpy as np

# ── COCO class list (80 classes) ─────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Classes relevant for blind navigation
NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "couch", "bed", "dining table", "toilet",
    "bottle", "bowl", "laptop", "cell phone", "book"
}

# ── Phase 2: Known real-world heights (metres) ────────────────────────────────
# Used for pinhole distance estimation: distance = (focal_length * real_height) / pixel_height
KNOWN_HEIGHTS = {
    "person":        1.70,
    "bicycle":       1.10,
    "car":           1.50,
    "motorcycle":    1.10,
    "bus":           3.20,
    "truck":         3.50,
    "chair":         0.90,
    "couch":         0.85,
    "bed":           0.60,
    "dining table":  0.75,
    "toilet":        0.75,
    "bottle":        0.25,
    "bowl":          0.12,
    "laptop":        0.30,
    "cell phone":    0.15,
    "book":          0.22,
    "backpack":      0.50,
    "umbrella":      1.00,
    "handbag":       0.35,
    "suitcase":      0.65,
    "cup":           0.12,
    "dog":           0.55,
    "cat":           0.30,
    "traffic light": 0.60,
    "fire hydrant":  0.55,
    "stop sign":     0.75,
    "bench":         0.85,
    "potted plant":  0.40,
    "tv":            0.55,
    "microwave":     0.35,
    "oven":          0.90,
    "sink":          0.25,
    "refrigerator":  1.80,
    "clock":         0.30,
    "vase":          0.30,
}

# Focal length from calibration (standard mobile camera baseline fallback)
# This is overwritten at runtime by set_focal_length() from CameraActivity.kt.
FOCAL_LENGTH_PX = 500.0


def set_focal_length(focal_px: float):
    """
    Override the focal length with the device's actual value.

    Called once from Kotlin/CameraActivity after reading Camera2 characteristics:
        focal_length_px = (focal_mm / sensor_width_mm) * image_width_px

    Args:
        focal_px : float — focal length in pixels for this device's main camera.
    """
    global FOCAL_LENGTH_PX
    FOCAL_LENGTH_PX = float(focal_px)
    print(f"[calibration] Focal length set to {FOCAL_LENGTH_PX:.2f} px")


# Zone thresholds (metres)
DANGER_THRESHOLD  = 0.8
CAUTION_THRESHOLD = 1.5
SAFE_THRESHOLD    = 3.0

# Alert suppression: minimum change in metres before re-alerting same object
MIN_CHANGE_TO_ALERT = 0.5
MAX_ALERT_DISTANCE  = 5.0

# Clipping: fraction of frame height at which an object is considered too close
CLIP_THRESHOLD = 0.90

# Module-level state for alert suppression
_last_alerted = {}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Detection postprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes, scores, iou_threshold=0.45):
    """Vectorized numpy Non-Maximum Suppression for fast mobile execution."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return kept


def postprocess_output(raw_floats, num_rows, num_cols, frame_w, frame_h,
                        conf_threshold=0.25, iou_threshold=0.45):
    """
    Decode raw YOLOv8 ONNX output tensor into detection dicts.

    Parameters
    ----------
    raw_floats : list[float] — flattened tensor (num_rows=84, num_cols=8400),
                               already scaled to frame pixel coordinates by Kotlin.
    frame_w, frame_h : int  — upright frame dimensions.

    Returns list of dicts: {label, score, box:[x1,y1,x2,y2], is_nav_relevant}
    """
    output = np.array(raw_floats, dtype=np.float32).reshape(num_rows, num_cols).T
    # output shape: (8400, 84)

    scores_all = np.max(output[:, 4:], axis=1)
    class_ids  = np.argmax(output[:, 4:], axis=1)

    mask = scores_all > conf_threshold
    output    = output[mask]
    scores    = scores_all[mask]
    class_ids = class_ids[mask]

    if len(output) == 0:
        return []

    cx = output[:, 0]; cy = output[:, 1]
    bw = output[:, 2]; bh = output[:, 3]
    x1 = np.clip((cx - bw / 2).astype(int), 0, frame_w)
    y1 = np.clip((cy - bh / 2).astype(int), 0, frame_h)
    x2 = np.clip((cx + bw / 2).astype(int), 0, frame_w)
    y2 = np.clip((cy + bh / 2).astype(int), 0, frame_h)

    boxes = list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))

    final_detections = []
    for cls_id in set(class_ids.tolist()):
        cls_mask   = (class_ids == cls_id)
        cls_boxes  = [boxes[i] for i, m in enumerate(cls_mask) if m]
        cls_scores = scores[cls_mask].tolist()
        for k in _nms(cls_boxes, cls_scores, iou_threshold):
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            final_detections.append({
                "label":          label,
                "score":          float(cls_scores[k]),
                "box":            [int(cls_boxes[k][0]), int(cls_boxes[k][1]),
                                   int(cls_boxes[k][2]), int(cls_boxes[k][3])],
                "is_nav_relevant": label in NAV_CLASSES,
            })
    return final_detections


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Distance estimation
# ═══════════════════════════════════════════════════════════════════════════════

def _horizontal_zone(box, frame_w):
    """Returns 'left', 'centre', or 'right' based on box centre-x."""
    x1, _, x2, _ = box
    cx = (x1 + x2) / 2
    third = frame_w / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "centre"
    return "right"


def _estimate_distance(label, box, frame_w):
    """
    Pinhole distance estimate for a single detection.
    Returns dict: {label, distance(m or None), zone, reliable, box}
    """
    x1, y1, x2, y2 = box
    pixel_height = y2 - y1
    zone = _horizontal_zone(box, frame_w)

    if label not in KNOWN_HEIGHTS or pixel_height < 20:
        return {"label": label, "distance": None, "zone": zone, "reliable": False}

    real_h = KNOWN_HEIGHTS[label]
    dist   = (FOCAL_LENGTH_PX * real_h) / pixel_height
    dist   = round(min(max(dist, 0.1), 10.0), 1)
    return {"label": label, "distance": dist, "zone": zone, "reliable": True}


def estimate_distances(detections, frame_w, frame_h):
    """
    Add distance + zone info to every detection. Returns augmented list
    sorted nearest-first. Also adds 'score' and 'box' keys to each entry.

    Called by Kotlin after postprocess_output().
    Returns list of dicts: {label, score, box, is_nav_relevant, distance, zone, reliable}
    """
    results = []
    for det in detections:
        est = _estimate_distance(det["label"], det["box"], frame_w)
        est["score"]          = det["score"]
        est["box"]            = det["box"]
        est["is_nav_relevant"] = det["is_nav_relevant"]
        results.append(est)

    results.sort(key=lambda x: x["distance"] if x["distance"] is not None else 999.0)
    return results


def zone_status(estimates):
    """
    Return the overall danger zone status for the overlay.
    Returns dict: {status: 'danger'|'caution'|'safe'|'clear', distance: float|None}
    """
    reliable = [e for e in estimates if e.get("distance") is not None and e.get("reliable")]
    if not reliable:
        return {"status": "clear", "distance": None}

    d = reliable[0]["distance"]
    if d <= DANGER_THRESHOLD:
        return {"status": "danger",  "distance": d}
    elif d <= CAUTION_THRESHOLD:
        return {"status": "caution", "distance": d}
    elif d <= SAFE_THRESHOLD:
        return {"status": "safe",    "distance": d}
    return {"status": "clear", "distance": d}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Clipping detection
# ═══════════════════════════════════════════════════════════════════════════════

def check_clipping(estimates, frame_h):
    """
    Return list of labels whose boxes clip the bottom of the frame
    (object is dangerously close / touching bottom edge).
    """
    clipping = []
    for est in estimates:
        x1, y1, x2, y2 = est["box"]
        box_clips = (y2 / frame_h) > CLIP_THRESHOLD
        if not box_clips:
            continue
        dist = est.get("distance")
        if dist is not None and dist > 1.5:
            continue
        clipping.append(est["label"])
    return clipping


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Alert text generation
# ═══════════════════════════════════════════════════════════════════════════════

def format_alerts(estimates, clipping_labels):
    """
    Generate spoken alert strings for this frame.

    Returns list of dicts: {text: str, priority: bool}
      priority=True  → QUEUE_FLUSH (interrupt current speech)
      priority=False → QUEUE_ADD   (append to queue)

    Called every N frames (rate-limited in Kotlin).
    """
    global _last_alerted
    alerts = []

    # 1. Clipping alerts — highest priority, interrupt speech
    for label in clipping_labels:
        alerts.append({
            "text":     f"{label.capitalize()}, very close, stop",
            "priority": True
        })

    # 2. Distance alerts — nearest first, already sorted
    for est in estimates:
        label    = est["label"]
        distance = est.get("distance")
        zone     = est.get("zone", "centre")
        reliable = est.get("reliable", False)

        if label in clipping_labels:
            continue  # already handled
        if not reliable or distance is None:
            continue
        if distance > MAX_ALERT_DISTANCE:
            continue

        # Suppress if not changed enough
        last = _last_alerted.get(label)
        if last is not None and abs(last - distance) < MIN_CHANGE_TO_ALERT:
            continue

        _last_alerted[label] = distance
        zone_str = f", {zone}" if zone != "centre" else ""
        alerts.append({
            "text":     f"{label.capitalize()}, {distance} metres{zone_str}",
            "priority": False
        })

    return alerts
