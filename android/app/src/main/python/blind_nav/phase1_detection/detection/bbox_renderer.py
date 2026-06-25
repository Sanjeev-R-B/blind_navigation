import cv2

# Color per class — nav-relevant classes get bright colors, others muted
CLASS_COLORS = {
    "person":       (0, 255, 120),    # green
    "bicycle":      (0, 220, 255),    # yellow
    "car":          (0, 60, 255),     # red
    "motorcycle":   (0, 140, 255),    # orange
    "bus":          (0, 0, 255),      # bright red
    "truck":        (0, 0, 200),      # dark red
    "chair":        (255, 180, 0),    # blue
    "couch":        (255, 160, 20),   # blue variant
    "bed":          (255, 140, 40),   # blue variant
    "dining table": (255, 120, 60),   # blue variant
    "toilet":       (200, 100, 200),  # purple
}
DEFAULT_COLOR = (80, 80, 80)  # grey for non-nav objects

def render(frame, detections):
    for det in detections:
        label  = det["label"]
        score  = det["score"]
        x1, y1, x2, y2 = det["box"]
        color  = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text    = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            frame, text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

    return frame