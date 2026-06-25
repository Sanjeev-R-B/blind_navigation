import cv2
import json
import sys
import os

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from phase1_detection.detection.yolo_detector import YOLODetector
from phase1_detection.detection.bbox_renderer import render

# A4 paper real height in metres
A4_HEIGHT_M = 0.297
CALIBRATION_OUTPUT = os.path.join(os.path.dirname(__file__), 'calibration.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'phase1_detection', 'models', 'yolov8n.onnx')


def draw_guide(frame):
    """Draw a horizontal guide line at centre to help align A4 sheet."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Centre crosshair
    cv2.line(frame, (cx - 60, cy), (cx + 60, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 60), (cx, cy + 60), (0, 255, 255), 1)

    # Instructions
    cv2.putText(frame, "Hold A4 paper FLAT facing camera at exactly 1 metre",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame, "Press SPACE to capture | Press Q to quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return frame


def get_pixel_height_manual(frame):
    """Let user draw a line on A4 paper to measure pixel height manually."""
    points = []
    clone = frame.copy()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (0, 0, 255), 2)
            cv2.imshow("Calibration — click top and bottom of A4", clone)

    cv2.imshow("Calibration — click top and bottom of A4", clone)
    cv2.setMouseCallback("Calibration — click top and bottom of A4", click)

    print("Click the TOP edge of the A4 paper, then the BOTTOM edge.")
    while len(points) < 2:
        cv2.waitKey(50)

    cv2.destroyAllWindows()
    pixel_height = abs(points[1][1] - points[0][1])
    return pixel_height


def calibrate():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("Calibration started.")
    print("Hold an A4 sheet flat, facing the camera, at exactly 1 metre distance.")
    print("Press SPACE when ready to capture.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_guide(frame)
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_frame = frame.copy()
            print("Frame captured.")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print("No frame captured.")
        return

    # Manual measurement — click top and bottom of A4
    print("\nNow click the TOP and BOTTOM edges of the A4 paper in the captured frame.")
    pixel_height = get_pixel_height_manual(captured_frame)
    print(f"Measured pixel height: {pixel_height}px")

    if pixel_height < 10:
        print("ERROR: Pixel height too small — try again closer to the paper.")
        return

    # Compute focal length
    focal_length = (pixel_height * 1.0) / A4_HEIGHT_M
    print(f"Computed focal length: {focal_length:.2f} pixels")

    # Save to JSON
    data = {
        "focal_length_px": round(focal_length, 2),
        "calibration_distance_m": 1.0,
        "a4_height_m": A4_HEIGHT_M,
        "measured_pixel_height": pixel_height,
        "camera": "default (index 0)"
    }

    with open(CALIBRATION_OUTPUT, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nCalibration saved to {CALIBRATION_OUTPUT}")
    print(f"Focal length: {focal_length:.2f}px — use this for all distance estimates.")


if __name__ == "__main__":
    calibrate()