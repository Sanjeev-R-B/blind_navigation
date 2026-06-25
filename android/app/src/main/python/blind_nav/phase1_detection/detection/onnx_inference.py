import onnxruntime as ort
import numpy as np
import cv2

class ONNXInference:
    def __init__(self, model_path: str, input_size: int = 640):
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # CHW → 1CHW
        return img

    def run(self, frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]  # shape: (1, 84, 8400)

    def postprocess(self, outputs, frame_w, frame_h, conf_threshold=0.4, nms_threshold=0.45):
      predictions = np.squeeze(outputs).T  # (8400, 84)
      scores = np.max(predictions[:, 4:], axis=1)
      mask = scores > conf_threshold
      predictions = predictions[mask]
      scores = scores[mask]

      if len(predictions) == 0:
        return []

      class_ids = np.argmax(predictions[:, 4:], axis=1)
      boxes_cx = predictions[:, 0] / self.input_size * frame_w
      boxes_cy = predictions[:, 1] / self.input_size * frame_h
      boxes_w  = predictions[:, 2] / self.input_size * frame_w
      boxes_h  = predictions[:, 3] / self.input_size * frame_h

      x1 = (boxes_cx - boxes_w / 2).astype(int)
      y1 = (boxes_cy - boxes_h / 2).astype(int)
      w  = boxes_w.astype(int)
      h  = boxes_h.astype(int)

    # NMS via OpenCV
      import cv2
      indices = cv2.dnn.NMSBoxes(
        list(zip(x1, y1, w, h)),
        scores.tolist(),
        conf_threshold,
        nms_threshold
    )

      detections = []
      for i in indices.flatten():
        detections.append({
            "class_id": int(class_ids[i]),
            "score":    float(scores[i]),
            "box":      (x1[i], y1[i], x1[i] + w[i], y1[i] + h[i])
        })

      return detections