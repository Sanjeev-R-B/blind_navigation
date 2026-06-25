import cv2
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection.yolo_detector import YOLODetector

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.onnx')
RESOLUTIONS = [320, 416, 480, 640]

def benchmark(resolution):
    cap = cv2.VideoCapture(0)
    detector = YOLODetector(MODEL_PATH, input_size=resolution, conf_threshold=0.25)

    frame_times = []
    frame_count = 0

    print(f"Testing {resolution}×{resolution} — warming up...")
    # warmup
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)

    print(f"Benchmarking {resolution}×{resolution}...")
    start = time.time()
    while time.time() - start < 5.0:  # 5 second test per resolution
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        detector.detect(frame)
        frame_times.append(time.time() - t0)
        frame_count += 1

    cap.release()
    avg_ms  = (sum(frame_times) / len(frame_times)) * 1000
    avg_fps = 1000 / avg_ms
    print(f"  {resolution}×{resolution} → {avg_fps:.1f} FPS  ({avg_ms:.1f}ms per frame)")
    return resolution, avg_fps

if __name__ == "__main__":
    print("=" * 45)
    print("  Resolution Benchmark — blind-nav Phase 1")
    print("=" * 45)
    results = []
    for res in RESOLUTIONS:
        r, fps = benchmark(res)
        results.append((r, fps))

    print("\n── Summary ──")
    for res, fps in results:
        bar    = "█" * int(fps / 2)
        marker = " ← recommended" if fps >= 12 and res == min(r for r, f in results if f >= 12) else ""
        print(f"  {res}×{res:3}  {fps:5.1f} FPS  {bar}{marker}")