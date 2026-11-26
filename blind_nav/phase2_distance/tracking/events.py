"""
phase2_distance/tracking/events.py

Turns tracker lifecycle changes into ENTER / EXIT events.

  ENTER : fired once, when a track becomes CONFIRMED
  EXIT  : fired once, when a confirmed track is deleted
          exit_reason = "left_frame"  last real box touched the left/right/top border
                        "lost"        vanished mid-frame (occlusion / missed detections)

The bottom border is deliberately ignored: a close object or a seated person
always touches the bottom edge, so it says nothing about leaving the frame.

Event record:
  {event, track_id, class, confidence, distance, timestamp, frame_index, bbox, exit_reason}
"""


class EventEmitter:
    def __init__(self, frame_w, frame_h, edge_margin=0.03, max_distance=None):
        """
        max_distance: if set, ENTER is held back until the track's estimated
        distance is within this range (metres), and EXIT is only emitted for
        tracks that had an ENTER. Keeps far-away flicker out of the event stream.
        An unknown distance (None) is never held back.
        """
        self.max_distance = max_distance
        self.w = frame_w
        self.h = frame_h
        self.mx = frame_w * edge_margin
        self.my = frame_h * edge_margin
        self._entered = set()

    def _left_frame(self, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= self.mx or x2 >= self.w - self.mx or y1 <= self.my

    def process(self, confirmed_tracks, removed_tracks, t, frame_index):
        """
        confirmed_tracks : list returned by Tracker.update()
        removed_tracks   : Tracker.removed_this_frame
        Returns a list of event dicts (usually empty).
        """
        events = []

        for tr in confirmed_tracks:
            if tr.id in self._entered:
                continue
            d = tr.last_distance
            if self.max_distance is not None and d is not None and d > self.max_distance:
                continue                       # too far to matter yet
            self._entered.add(tr.id)
            events.append({
                "event": "ENTER",
                "track_id": tr.id,
                "class": tr.label,
                "confidence": round(tr.score, 2),
                "distance": d,
                "timestamp": round(t, 3),
                "frame_index": frame_index,
                "bbox": tuple(int(v) for v in tr.last_bbox),
                "exit_reason": None,
            })

        for tr in removed_tracks:
            if tr.id not in self._entered:
                continue                       # never announced, so no EXIT either
            self._entered.discard(tr.id)
            events.append({
                "event": "EXIT",
                "track_id": tr.id,
                "class": tr.label,
                "confidence": round(tr.score, 2),
                "distance": tr.last_distance,
                "timestamp": round(t, 3),
                "frame_index": frame_index,
                "bbox": tuple(int(v) for v in tr.last_bbox),
                "exit_reason": "left_frame" if self._left_frame(tr.last_bbox) else "lost",
            })

        return events