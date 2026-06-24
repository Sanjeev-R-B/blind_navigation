import cv2
import sys
import os
import time
import textwrap
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from phase1_detection.detection.yolo_detector import YOLODetector
from phase1_detection.detection.bbox_renderer import render
from phase1_detection.utils.fps_counter import FPSCounter
from phase2_distance.distance.pinhole_estimator import estimate_all
from phase2_distance.utils.clip_detector import check_all
from phase2_distance.audio.alert_formatter import AlertFormatter
from phase2_distance.audio.tts_engine import TTSEngine

MODEL_PATH = os.path.join(ROOT, 'phase1_detection', 'models', 'yolov8n.onnx')

ALERT_EVERY_N_FRAMES = 60        # ~2 seconds at 30fps
SUMMARY_EVERY_SECONDS = 4.0

PANEL_W = 360
GAP = 15

DANGER_THRESHOLD = 0.8
CAUTION_THRESHOLD = 1.5
SAFE_THRESHOLD = 3.0


def get_risk_color(distance):
    if distance is None:
        return (180, 180, 180), "UNKNOWN"
    if distance <= DANGER_THRESHOLD:
        return (0, 0, 255), "DANGER"
    if distance <= CAUTION_THRESHOLD:
        return (0, 140, 255), "CAUTION"
    if distance <= SAFE_THRESHOLD:
        return (0, 200, 80), "SAFE"
    return (180, 180, 180), "OK"


def draw_distances(frame, estimates):
    for det in estimates:
        if det['distance'] is None:
            continue
        x1, y1, _, _ = det['box']
        distance = det['distance']
        zone = det['zone']
        color, _ = get_risk_color(distance)
        text = f"{distance:.1f}m {zone}"
        cv2.putText(frame, text, (x1, max(20, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame


def draw_zone_indicator(frame, estimates, clipping_labels):
    h, w = frame.shape[:2]
    status = None
    color = None

    reliable = [e for e in estimates if e['distance'] is not None and e['reliable']]
    closest = reliable[0]['distance'] if reliable else None

    if clipping_labels and closest is not None:
        status = f"DANGER  {closest:.1f}m  TOO CLOSE"
        color = (0, 0, 255)
    elif reliable:
        d = reliable[0]['distance']
        if d <= DANGER_THRESHOLD:
            status = f"DANGER   {d:.1f}m"
            color = (0, 0, 255)
        elif d <= CAUTION_THRESHOLD:
            status = f"CAUTION  {d:.1f}m"
            color = (0, 140, 255)
        elif d <= SAFE_THRESHOLD:
            status = f"SAFE     {d:.1f}m"
            color = (0, 200, 80)

    if status and color:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
        cv2.rectangle(frame, (0, h - 50), (w, h), color, -1)
        cv2.putText(frame, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def wrap_lines(text, width=32):
    lines = []
    for para in str(text).split("\n"):
        wrapped = textwrap.wrap(para, width=width)
        lines.extend(wrapped if wrapped else [""])
    return lines


def local_scene_summary(estimates, clipping):
    reliable = [e for e in estimates if e.get("distance") is not None and e.get("reliable")]
    if not reliable:
        return "No reliable objects detected."

    nearest = reliable[0]
    label = str(nearest.get("label", "object"))
    dist = nearest["distance"]

    parts = [f"Nearest {label} at {dist:.1f}m"]
    if clipping:
        parts.append("clipping risk detected")
    if dist <= DANGER_THRESHOLD:
        parts.append("danger")
    elif dist <= CAUTION_THRESHOLD:
        parts.append("caution")
    elif dist <= SAFE_THRESHOLD:
        parts.append("safe")

    return ", ".join(parts) + "."


def draw_dashboard(dashboard_area, estimates, scene_summary, fps_value):
    """Draw dashboard content onto the right panel"""
    h, w = dashboard_area.shape[:2]
    
    cv2.rectangle(dashboard_area, (0, 0), (w, h), (20, 20, 22), -1)

    y = 35
    cv2.putText(dashboard_area, "LIVE DASHBOARD", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    y += 30

    cv2.putText(dashboard_area, f"FPS: {fps_value:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    y += 38

    cv2.putText(dashboard_area, "Objects", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28

    shown = 0
    for det in estimates:
        if det['distance'] is None:
            continue
        color, risk = get_risk_color(det['distance'])
        label = str(det.get('label', 'object'))
        distance = det['distance']

        cv2.putText(dashboard_area, label[:20], (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        y += 19
        cv2.putText(dashboard_area, f"{distance:.1f}m   {risk}", (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 2, cv2.LINE_AA)
        y += 27
        shown += 1
        if shown >= 7:
            break

    y += 12
    cv2.putText(dashboard_area, "Scene Summary", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    y += 26

    for line in wrap_lines(scene_summary, width=32)[:6]:
        cv2.putText(dashboard_area, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        y += 21

    return dashboard_area


def run():
    cap = cv2.VideoCapture(0)
    detector = YOLODetector(MODEL_PATH, input_size=416, conf_threshold=0.25)
    fps = FPSCounter()
    formatter = AlertFormatter()
    tts = TTSEngine(rate=150)

    tts.start()

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("Phase 2 running. Press Q to quit.")
    frame_count = 0
    scene_summary = "Starting up..."
    last_summary_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_count += 1

        detections = detector.detect(frame)
        estimates = estimate_all(detections, w)
        clipping = check_all(estimates, h)

        frame = render(frame, detections)
        frame = draw_distances(frame, estimates)
        frame = draw_zone_indicator(frame, estimates, clipping)

        now = time.time()
        if now - last_summary_time >= SUMMARY_EVERY_SECONDS:
            scene_summary = local_scene_summary(estimates, clipping)
            last_summary_time = now

        fps.tick()

        # ====================== CREATE SPLIT LAYOUT ======================
        total_width = w + PANEL_W + GAP
        combined = np.zeros((h, total_width, 3), dtype=np.uint8)

        # Left: Camera feed
        combined[:, :w] = frame

        # Right: Dashboard panel
        dashboard_area = combined[:, w + GAP : w + GAP + PANEL_W]
        dashboard_area = draw_dashboard(dashboard_area, estimates, scene_summary, fps.get())

        # Vertical separator
        cv2.line(combined, (w + GAP//2, 0), (w + GAP//2, h), (60, 60, 70), 2)

        cv2.imshow("blind-nav | Phase 2", combined)

        # ====================== ALERTS ======================
        if frame_count % ALERT_EVERY_N_FRAMES == 0:
            alerts = formatter.format_all(estimates, clipping)
            for text, priority in alerts:
                if text and text.strip():
                    tts.speak(text, priority=priority)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tts.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()