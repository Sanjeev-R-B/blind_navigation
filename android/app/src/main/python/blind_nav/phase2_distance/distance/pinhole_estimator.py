import json
import os

# Load calibration
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), 'calibration.json')
KNOWN_HEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'known_heights.json')

with open(CALIBRATION_PATH, 'r') as f:
    _calib = json.load(f)

with open(KNOWN_HEIGHTS_PATH, 'r') as f:
    _known_heights = json.load(f)

FOCAL_LENGTH = _calib['focal_length_px']


def get_horizontal_zone(box, frame_w):
    """Returns 'left', 'centre', or 'right' based on box centre x position."""
    x1, _, x2, _ = box
    cx = (x1 + x2) / 2
    third = frame_w / 3
    if cx < third:
        return 'left'
    elif cx < 2 * third:
        return 'centre'
    else:
        return 'right'


def estimate(label, box, frame_w):
    """
    Estimate distance for a single detection.

    Args:
        label    : class label string e.g. 'person'
        box      : (x1, y1, x2, y2) bounding box in pixels
        frame_w  : frame width in pixels (for left/centre/right zone)

    Returns dict:
        {
            'label':    'person',
            'distance': 2.4,          # metres, None if unknown class
            'zone':     'left',       # horizontal position
            'reliable': True          # False if box is too small to trust
        }
    """
    x1, y1, x2, y2 = box
    pixel_height = y2 - y1

    zone = get_horizontal_zone(box, frame_w)

    # Unknown class — no height in lookup table
    if label not in _known_heights:
        return {
            'label':    label,
            'distance': None,
            'zone':     zone,
            'reliable': False
        }

    real_height = _known_heights[label]

    # Box too small — unreliable estimate
    if pixel_height < 20:
        return {
            'label':    label,
            'distance': None,
            'zone':     zone,
            'reliable': False
        }

    distance = (FOCAL_LENGTH * real_height) / pixel_height

    # Clamp to sensible range — beyond 10m is not useful for navigation
    distance = min(distance, 10.0)
    distance = max(distance, 0.1)

    return {
        'label':    label,
        'distance': round(distance, 1),
        'zone':     zone,
        'reliable': True
    }


def estimate_all(detections, frame_w):
    """
    Run estimate() on every detection from YOLO.

    Args:
        detections : list of dicts from YOLODetector.detect()
        frame_w    : frame width in pixels

    Returns list of estimate dicts, sorted nearest first.
    """
    results = []
    for det in detections:
        result = estimate(det['label'], det['box'], frame_w)
        result['score'] = det['score']
        result['box']   = det['box']
        results.append(result)

    # Sort by distance — nearest obstacle first
    results.sort(key=lambda x: x['distance'] if x['distance'] is not None else 999)
    return results

