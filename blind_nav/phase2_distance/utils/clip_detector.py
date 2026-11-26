# Fraction of frame height — box bottom must reach this to count as touching the edge.
# 0.90 catches real clips without being as brittle as 0.99.
CLIP_THRESHOLD = 0.90
CLIP_MAX_DISTANCE_M = 0.5


def is_clipping(box, frame_h, distance=None):
    """
    True only when the box touches the bottom of the frame AND the
    estimated distance is known and under CLIP_MAX_DISTANCE_M.
    """
    _, _, _, y2 = box
    touches_bottom = (y2 / frame_h) > CLIP_THRESHOLD

    if not touches_bottom:
        return False

    # No distance estimate -> don't trust the clip signal
    if distance is None:
        return False

    return distance < CLIP_MAX_DISTANCE_M

def check_all(detections, frame_h):
    clipping = []
    for det in detections:
        distance = det.get('distance')
        if is_clipping(det['box'], frame_h, distance):
            clipping.append(det['label'])
    return clipping