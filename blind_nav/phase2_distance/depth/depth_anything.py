import onnxruntime as ort
import numpy as np
import cv2
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'depth_v2_small.onnx')

# Depth Anything v2 input size
INPUT_SIZE = 224


class DepthAnything:
    def __init__(self):
        print("Loading Depth Anything v2 Small...")
        self.session    = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print("Depth model loaded.")

    def preprocess(self, frame):
        """Resize + normalize frame for Depth Anything input."""
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization — required by Depth Anything
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # CHW → 1CHW
        return img.astype(np.float32)

    def infer(self, frame):
        """
        Run depth inference on a frame.

        Returns:
            depth_map : 2D numpy array (H, W), normalized 0–1
                        where 1.0 = closest, 0.0 = farthest
        """
        h, w      = frame.shape[:2]
        tensor    = self.preprocess(frame)
        outputs   = self.session.run(None, {self.input_name: tensor})
        depth_raw = outputs[0].squeeze()  # shape: (518, 518)

        # Normalize to 0–1
        d_min = depth_raw.min()
        d_max = depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        # Resize back to original frame size
        depth_resized = cv2.resize(depth_norm, (w, h))
        return depth_resized

    def get_region_depth(self, depth_map, box):
        """
        Get average depth value inside a bounding box region.

        Args:
            depth_map : 2D array from infer()
            box       : (x1, y1, x2, y2)

        Returns:
            float 0–1, where higher = closer
        """
        x1, y1, x2, y2 = box
        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float(np.mean(region))

    def colorize(self, depth_map):
        """Convert depth map to a colorized BGR image for visualization."""
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored     = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return colored


if __name__ == "__main__":
    import time

    depth_model = DepthAnything()
    cap         = cv2.VideoCapture(0)

    print("Depth visualization running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0        = time.time()
        depth_map = depth_model.infer(frame)
        ms        = (time.time() - t0) * 1000

        colored   = depth_model.colorize(depth_map)

        # Stack original + depth side by side
        combined  = np.hstack([frame, colored])
        cv2.putText(
            combined,
            f"Depth inference: {ms:.0f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        cv2.imshow("Depth Anything v2 Small", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()