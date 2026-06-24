import cv2
import numpy as np
import os
import sys

# Ensure detection packages can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection.yolo_detector import YOLODetector

detector = None

def init_detector(model_path):
    global detector
    # Use 416x416 resolution as optimized in benchmarking
    detector = YOLODetector(model_path, input_size=416, conf_threshold=0.25)
    print("YOLO Detector initialized with model:", model_path)

def process_frame(frame_bytes, width, height, rotation_degrees):
    global detector
    if detector is None:
        return []

    # 1. Convert byte array to numpy array
    # Frame shape from CameraX RGBA_8888 is (height, width, 4)
    data = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_rgba = data.reshape((height, width, 4))

    # 2. Convert to BGR format for detector/OpenCV
    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

    # 3. Rotate the frame if necessary
    if rotation_degrees == 90:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_degrees == 180:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 4. Perform object detection
    detections = detector.detect(frame_bgr)

    # 5. Format detections for Java/Kotlin consumption
    formatted_results = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        formatted_results.append({
            "label": str(det["label"]),
            "score": float(det["score"]),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "is_nav_relevant": bool(det["is_nav_relevant"])
        })

    return formatted_results
