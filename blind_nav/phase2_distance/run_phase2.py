import cv2
import sys
import os

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

# How many frames to skip between TTS alerts — avoids speech flood
ALERT_EVERY_N_FRAMES = 45


def draw_distances(frame, estimates):
    """Overlay distance text on each bounding box."""
    for det in estimates:
        if det['distance'] is None:
            continue
        x1, y1, _, _ = det['box']
        zone     = det['zone']
        distance = det['distance']
        text     = f"{distance}m {zone}"
        cv2.putText(
            frame, text,
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 1, cv2.LINE_AA
        )
    return frame


def run():
    cap       = cv2.VideoCapture(0)
    detector  = YOLODetector(MODEL_PATH, input_size=416, conf_threshold=0.25)
    fps       = FPSCounter()
    formatter = AlertFormatter()
    tts       = TTSEngine(rate=150)

    tts.start()

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("Phase 2 running. Press Q to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_count += 1

        # --- Detection ---
        detections = detector.detect(frame)

        # --- Distance estimation ---
        estimates  = estimate_all(detections, w)

        # --- Clipping check ---
        clipping   = check_all(estimates, h)

        # --- Render boxes + distances ---
        frame = render(frame, detections)
        frame = draw_distances(frame, estimates)
        frame = draw_zone_indicator(frame, estimates, clipping)

        # --- Voice alerts every N frames ---
        if frame_count % ALERT_EVERY_N_FRAMES == 0:
            alerts = formatter.format_all(estimates, clipping)
            for text, priority in alerts:
                tts.speak(text, priority=priority)

        # --- FPS overlay ---
        fps.tick()
        cv2.putText(
            frame,
            f"FPS: {fps.get():.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        cv2.imshow("blind-nav | Phase 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tts.stop()
    cap.release()
    cv2.destroyAllWindows()

DANGER_THRESHOLD  = 0.8   # metres
CAUTION_THRESHOLD = 1.5   # metres
SAFE_THRESHOLD    = 3.0   # metres

def draw_zone_indicator(frame, estimates, clipping_labels):
    h, w = frame.shape[:2]

    status  = None
    color   = None

    if clipping_labels:
        reliable = [e for e in estimates if e['distance'] is not None and e['reliable']]
        closest  = reliable[0]['distance'] if reliable else 0.0
        status   = f"DANGER  {closest}m  TOO CLOSE"
        color    = (0, 0, 255)

    else:
        reliable = [e for e in estimates if e['distance'] is not None and e['reliable']]
        if reliable:
            d = reliable[0]['distance']  # nearest object

            if d <= DANGER_THRESHOLD:
                status = f"DANGER   {d}m"
                color  = (0, 0, 255)       # red
            elif d <= CAUTION_THRESHOLD:
                status = f"CAUTION  {d}m"
                color  = (0, 140, 255)     # orange
            elif d <= SAFE_THRESHOLD:
                status = f"SAFE     {d}m"
                color  = (0, 200, 80)      # green
            # beyond 5m — no indicator at all

    if status and color:
        # Border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
        # Bottom bar
        cv2.rectangle(frame, (0, h - 50), (w, h), color, -1)
        cv2.putText(
            frame, status,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

    return frame
if __name__ == "__main__":
    run()