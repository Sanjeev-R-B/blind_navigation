import numpy as np
import os
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from phase2_distance.depth.depth_anything import DepthAnything

DEPTH_CLOSE_THRESHOLD = 0.65


class DepthFusion:
    def __init__(self):
        self.depth_model  = DepthAnything()
        self._depth_cache = None
        self._running     = False
        self._thread      = None
        self._lock        = threading.Lock()
        self._latest_frame = None
        self._frame_lock  = threading.Lock()

    def start(self):
        """Start background depth inference thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("Depth thread started.")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _worker(self):
        """Runs depth inference every 2 seconds in background."""
        import time
        while self._running:
           with self._frame_lock:
               frame = self._latest_frame

           if frame is None:
            time.sleep(0.1)
            continue

        depth_map = self.depth_model.infer(frame)

        with self._lock:
            self._depth_cache = depth_map

        # Wait 2 seconds before next depth inference
        time.sleep(2.0)

    def update_frame(self, frame):
        """Feed latest frame to depth thread — non-blocking."""
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def get_depth_map(self):
        """Get latest cached depth map — non-blocking."""
        with self._lock:
            return self._depth_cache

    def fuse(self, estimates, depth_map):
        """
        Fuse pinhole estimates with depth map.
        Known objects → keep pinhole distance.
        Unknown/unreliable → use depth as fallback.
        """
        if depth_map is None:
            return estimates

        for est in estimates:
            box         = est['box']
            depth_value = self.depth_model.get_region_depth(depth_map, box)
            depth_close = depth_value > DEPTH_CLOSE_THRESHOLD

            est['depth_value'] = round(depth_value, 3)
            est['depth_close'] = depth_close

            if not est['reliable']:
                est['depth_fallback'] = depth_close
                if depth_close and est['distance'] is None:
                    est['distance'] = 0.8
                    est['reliable'] = True
            else:
                est['depth_fallback'] = False

        return estimates

    def colorize_overlay(self, frame, depth_map, alpha=0.3):
        import cv2
        colored = self.depth_model.colorize(depth_map)
        blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
        return blended


if __name__ == "__main__":
    import cv2
    sys.path.insert(0, ROOT)

    from phase1_detection.detection.yolo_detector import YOLODetector
    from phase1_detection.detection.bbox_renderer import render
    from phase1_detection.utils.fps_counter import FPSCounter
    from phase2_distance.distance.pinhole_estimator import estimate_all

    MODEL_PATH = os.path.join(ROOT, 'phase1_detection', 'models', 'yolov8n.onnx')

    cap      = cv2.VideoCapture(0)
    detector = YOLODetector(MODEL_PATH, input_size=416, conf_threshold=0.25)
    fusion   = DepthFusion()
    fps      = FPSCounter()

    fusion.start()

    print("Fusion pipeline running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Feed frame to depth thread — non-blocking
        fusion.update_frame(frame)

        # Detection — runs at full speed
        detections = detector.detect(frame)
        estimates  = estimate_all(detections, w)

        # Get cached depth map — non-blocking
        depth_map  = fusion.get_depth_map()
        estimates  = fusion.fuse(estimates, depth_map)

        # Render
        frame = render(frame, detections)

        for est in estimates:
            if est['distance'] is None:
                continue
            x1, y1, _, _ = est['box']
            dv    = est.get('depth_value', 0)
            close = est.get('depth_close', False)
            color = (0, 0, 255) if close else (0, 255, 0)
            cv2.putText(
                frame,
                f"{est['distance']}m d:{dv:.2f}",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA
            )

        fps.tick()
        cv2.putText(
            frame,
            f"FPS: {fps.get():.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        cv2.imshow("blind-nav | Depth Fusion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fusion.stop()
    cap.release()
    cv2.destroyAllWindows()