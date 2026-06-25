import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection.onnx_inference import ONNXInference

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

# Classes relevant to blind navigation
NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "couch", "bed", "dining table", "toilet",
    "bottle", "bowl", "laptop", "cell phone", "book"
}
DEMO_CLASSES = {'person', 'chair', 'book', 'cell phone', 'backpack'}

class YOLODetector:
    def __init__(self, model_path: str, input_size: int = 416, conf_threshold: float = 0.25):
        self.engine = ONNXInference(model_path, input_size)
        self.conf_threshold = conf_threshold

    # Only these classes will be detected — everything else ignored
    

    def detect(self, frame):
        h, w = frame.shape[:2]
        raw_output = self.engine.run(frame)
        detections = self.engine.postprocess(raw_output, w, h, self.conf_threshold)

        results = []
        for det in detections:
            label = COCO_CLASSES[det["class_id"]] if det["class_id"] < len(COCO_CLASSES) else "unknown"

        # Whitelist filter — only demo classes pass
            if label not in DEMO_CLASSES:
               continue

            results.append({
               "label": label,
               "score": det["score"],
               "box":   det["box"],
               "is_nav_relevant": label in NAV_CLASSES
        })

        return results
