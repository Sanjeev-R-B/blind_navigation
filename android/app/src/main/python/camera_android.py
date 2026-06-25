"""
camera_android.py — Phase 1 + Phase 2 pipeline for on-device blind navigation.
This file acts as a bridge between Android Kotlin (CameraX/Java ORT) and the 
pure python pipeline from blind_nav/.
"""

import sys
import numpy as np

# ── Mock dependencies unavailable in Android ─────────────────────────────────
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['cv2'] = MockModule()
sys.modules['onnxruntime'] = MockModule()

# Now we can safely import from blind_nav
from blind_nav.phase1_detection.detection.yolo_detector import COCO_CLASSES, NAV_CLASSES, DEMO_CLASSES
import blind_nav.phase2_distance.distance.pinhole_estimator as pinhole_estimator
import blind_nav.phase2_distance.utils.clip_detector as clip_detector
from blind_nav.phase2_distance.audio.alert_formatter import AlertFormatter

# ── Zone thresholds (metres) ────────────────────────────────────────────────
DANGER_THRESHOLD  = 0.8
CAUTION_THRESHOLD = 1.5
SAFE_THRESHOLD    = 3.0

formatter = AlertFormatter()

def set_focal_length(focal_px: float):
    """
    Override the focal length with the device's actual value.
    Called once from Kotlin/CameraActivity after reading Camera2 characteristics.
    """
    pinhole_estimator.FOCAL_LENGTH = float(focal_px)
    print(f"[calibration] Focal length set to {float(focal_px):.2f} px")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Detection postprocessing (Android Numpy NMS)
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(box_a, box_b):
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
    if len(boxes) == 0: return []
    boxes = np.array(boxes); scores = np.array(scores)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return kept

def postprocess_output(raw_floats, num_rows, num_cols, frame_w, frame_h,
                        conf_threshold=0.25, iou_threshold=0.45):
    """
    Decode raw YOLOv8 ONNX output tensor into detection dicts.
    """
    output = np.array(raw_floats, dtype=np.float32).reshape(num_rows, num_cols).T
    scores_all = np.max(output[:, 4:], axis=1)
    class_ids  = np.argmax(output[:, 4:], axis=1)

    mask = scores_all > conf_threshold
    output    = output[mask]
    scores    = scores_all[mask]
    class_ids = class_ids[mask]

    if len(output) == 0: return []

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
            
            # Whitelist filter just like yolo_detector.py
            if label not in DEMO_CLASSES:
                continue
                
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

def estimate_distances(detections, frame_w, frame_h):
    # Delegate to the real implementation in blind_nav
    estimates = pinhole_estimator.estimate_all(detections, frame_w)
    
    # Kotlin expects "is_nav_relevant" to be present, but pinhole_estimator 
    # doesn't carry it over, so we need to inject it back.
    det_map = {str(d['box']): d.get('is_nav_relevant', False) for d in detections}
    for est in estimates:
        est['is_nav_relevant'] = det_map.get(str(est['box']), False)
        
    return estimates

def zone_status(estimates):
    reliable = [e for e in estimates if e.get("distance") is not None and e.get("reliable")]
    if not reliable:
        return {"status": "clear", "distance": None}

    d = reliable[0]["distance"]
    if d <= DANGER_THRESHOLD:
        return {"status": "danger",  "distance": float(d)}
    elif d <= CAUTION_THRESHOLD:
        return {"status": "caution", "distance": float(d)}
    elif d <= SAFE_THRESHOLD:
        return {"status": "safe",    "distance": float(d)}
    return {"status": "clear", "distance": float(d)}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Clipping detection
# ═══════════════════════════════════════════════════════════════════════════════

def check_clipping(estimates, frame_h):
    return clip_detector.check_all(estimates, frame_h)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Alert text generation
# ═══════════════════════════════════════════════════════════════════════════════

def format_alerts(estimates, clipping_labels):
    # Get tuples of (text, is_priority) from the shared formatter
    raw_alerts = formatter.format_all(estimates, clipping_labels)
    
    # Map them to the list of dictionaries expected by Kotlin/CameraActivity
    alerts = []
    for text, priority in raw_alerts:
        alerts.append({
            "text": text,
            "priority": priority
        })
    return alerts
