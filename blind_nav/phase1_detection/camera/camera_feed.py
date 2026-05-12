import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.fps_counter import FPSCounter
from detection.yolo_detector import YOLODetector
from detection.bbox_renderer import render

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n.onnx')

def run():
    cap      = cv2.VideoCapture(0)
    detector = YOLODetector(MODEL_PATH)
    fps      = FPSCounter()

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        detections = detector.detect(frame)
        frame      = render(frame, detections)

        fps.tick()
        cv2.putText(
            frame,
            f"FPS: {fps.get():.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        cv2.imshow("blind-nav | Phase 1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()