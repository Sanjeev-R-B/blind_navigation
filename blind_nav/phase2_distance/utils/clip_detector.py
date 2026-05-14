# Percentage of frame height at which an object is considered "too close"
CLIP_THRESHOLD = 0.99


def is_clipping(box, frame_h, distance=None):
    """
    Check if a bounding box is clipping the bottom of the frame.
    Only triggers if distance is also close (under 1.5m) or unknown.
    """
    _, _, _, y2 = box
    box_clips = (y2 / frame_h) > CLIP_THRESHOLD
    
    if not box_clips:
        return False
    
    # If we have a distance estimate, only clip-alert if actually close
    if distance is not None and distance > 1.5:
        return False
    
    return True

def check_all(detections, frame_h):
    clipping = []
    for det in detections:
        distance = det.get('distance')
        if is_clipping(det['box'], frame_h, distance):
            clipping.append(det['label'])
    return clipping