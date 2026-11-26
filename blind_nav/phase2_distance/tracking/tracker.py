"""
phase2_distance/tracking/tracker.py

Pure-NumPy multi-object tracker (no torch, no scipy), so the same file can run
on desktop and be ported to the phone.

Design follows the two-stage idea of ByteTrack/BoT-SORT:
  stage 1: high-confidence boxes are matched to ALL tracks by IoU
  stage 2: low-confidence boxes are matched to the still-unmatched, non-lost
           tracks (keeps an ID alive when the score briefly dips)
  new tracks start ONLY from unmatched high-confidence boxes

Lifecycle:  TENTATIVE -> CONFIRMED -> LOST -> deleted
  - TENTATIVE -> CONFIRMED after n_init consecutive matched frames
  - a tentative track that misses a frame is deleted silently
  - CONFIRMED -> LOST on a missed frame (keeps coasting on the Kalman prediction)
  - LOST -> CONFIRMED if re-matched within max_age_s
  - LOST -> deleted after max_age_s (time based, because FPS varies)

Input to Tracker.update(): list of dicts {"label", "score", "box": (x1,y1,x2,y2)}
exactly as returned by YOLODetector.detect(frame, include_low=True).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

TENTATIVE, CONFIRMED, LOST = "tentative", "confirmed", "lost"


@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.25   # boxes >= this are "high confidence"
    track_low_thresh: float = 0.10    # boxes in [low, high) are "low confidence"
    new_track_thresh: float = 0.25    # a new track may only start from boxes >= this
    match_iou: float = 0.30           # stage 1 minimum overlap
    match_iou_low: float = 0.50       # stage 2 is stricter, low boxes are less reliable
    n_init: int = 3                   # consecutive matched frames to confirm
    max_age_s: float = 1.5            # how long a LOST track is remembered (seconds)
    cross_class_penalty: float = 0.5  # IoU is multiplied by this if labels differ


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #

def _xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = [float(v) for v in b]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h])


def _cxcywh_to_xyxy(z):
    cx, cy, w, h = z
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) in x1,y1,x2,y2 -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    iw = np.clip(np.minimum(a[:, 2:3], b[:, 2]) - np.maximum(a[:, 0:1], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3:4], b[:, 3]) - np.maximum(a[:, 1:2], b[:, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(sim, thresh):
    """
    Greedy assignment, highest similarity first.
    sim: (T, D). Returns (matches [(t, d)], unmatched_t, unmatched_d).
    """
    n_t, n_d = sim.shape
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    used_t, used_d, matches = set(), set(), []
    for ti, di in order:
        ti, di = int(ti), int(di)
        if sim[ti, di] < thresh:
            break
        if ti in used_t or di in used_d:
            continue
        matches.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    return (matches,
            [i for i in range(n_t) if i not in used_t],
            [i for i in range(n_d) if i not in used_d])


# --------------------------------------------------------------------------- #
#  Kalman filter on [cx, cy, w, h, vcx, vcy, vw, vh]
# --------------------------------------------------------------------------- #

class BoxKalman:
    _STD_POS = 1.0 / 20.0     # noise scales with box height (same idea as ByteTrack)
    _STD_VEL = 1.0 / 160.0

    def __init__(self, box):
        z = _xyxy_to_cxcywh(box)
        h = z[3]
        self.x = np.concatenate([z, np.zeros(4)])
        std = np.array([2 * self._STD_POS * h] * 4 + [10 * self._STD_VEL * h] * 4)
        self.P = np.diag(std ** 2)

    def predict(self, dt, damp_velocity=False):
        dt = max(float(dt), 0.0)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        h = self.x[3]
        q = np.array([self._STD_POS * h] * 4 + [self._STD_VEL * h] * 4) ** 2
        Q = np.diag(q) * max(dt * 30.0, 1e-3)    # uncertainty grows with elapsed time
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if damp_velocity:                        # a lost object shouldn't drift away forever
            self.x[4:] *= 0.9 ** (dt * 30.0)
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)

    def update(self, box):
        z = _xyxy_to_cxcywh(box)
        h = self.x[3]
        H = np.hstack([np.eye(4), np.zeros((4, 4))])
        R = np.diag([self._STD_POS * h] * 4) ** 2
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T      # P H^T S^-1
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)

    @property
    def box(self):
        return _cxcywh_to_xyxy(self.x[:4])


# --------------------------------------------------------------------------- #
#  Track
# --------------------------------------------------------------------------- #

class Track:
    def __init__(self, track_id, det, t, state=TENTATIVE):
        self.id = track_id
        self.kf = BoxKalman(det["box"])
        self.state = state
        self.ever_confirmed = state == CONFIRMED
        self.hits = 1                    # total matched frames
        self.streak = 1                  # consecutive matched frames
        self.first_seen_t = t
        self.last_seen_t = t
        self._last_pred_t = t
        self.score = float(det["score"])
        self.last_bbox = tuple(det["box"])   # last DETECTED box (not the Kalman prediction)
        self.class_votes = {det["label"]: float(det["score"])}
        self.history = deque(maxlen=150)     # (t, bbox, distance) - filled by add_history()

    # -- read-only views ---------------------------------------------------- #
    @property
    def box(self):
        return self.kf.box

    @property
    def label(self):
        return max(self.class_votes, key=self.class_votes.get)

    # -- distance history (registry) ---------------------------------------- #
    def add_history(self, t, distance):
        """Record one observation. distance may be None (unknown class / tiny box)."""
        self.history.append((t, tuple(int(v) for v in self.box), distance))

    @property
    def last_distance(self):
        for _, _, d in reversed(self.history):
            if d is not None:
                return d
        return None

    def distance_window(self, window_s, now):
        """[(t, distance)] for valid readings in the last window_s seconds, oldest first."""
        return [(t, d) for t, _, d in self.history if d is not None and now - t <= window_s]

    # -- per-frame steps ---------------------------------------------------- #
    def predict(self, t):
        self.kf.predict(t - self._last_pred_t, damp_velocity=(self.state == LOST))
        self._last_pred_t = t

    def apply(self, det, t, cfg):
        self.kf.update(det["box"])
        self.hits += 1
        self.streak += 1
        self.last_seen_t = t
        self.score = float(det["score"])
        self.last_bbox = tuple(det["box"])
        self.class_votes[det["label"]] = self.class_votes.get(det["label"], 0.0) + float(det["score"])
        if self.state == LOST:
            self.state = CONFIRMED
        elif self.state == TENTATIVE and self.streak >= cfg.n_init:
            self.state = CONFIRMED
            self.ever_confirmed = True


# --------------------------------------------------------------------------- #
#  Tracker
# --------------------------------------------------------------------------- #

class Tracker:
    def __init__(self, cfg=None, fast_confirm=None):
        """
        fast_confirm: optional callable(det) -> bool. When it returns True a new
        track starts CONFIRMED immediately (skips the n_init wait). Intended for
        objects already inside the DANGER zone. Unused in step 2.
        """
        self.cfg = cfg or TrackerConfig()
        self.fast_confirm = fast_confirm
        self.tracks = []
        self.removed_this_frame = []     # confirmed tracks deleted this frame (EXIT, step 3)
        self._next_id = 1

    @property
    def registry(self):
        """{track_id: Track} for every live (tentative, confirmed or lost) track."""
        return {tr.id: tr for tr in self.tracks}

    def _similarity(self, tracks, dets):
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        sim = iou_matrix([tr.box for tr in tracks], [d["box"] for d in dets])
        for i, tr in enumerate(tracks):
            lbl = tr.label
            for j, d in enumerate(dets):
                if d["label"] != lbl:
                    sim[i, j] *= self.cfg.cross_class_penalty
        return sim

    def update(self, dets, t):
        """
        dets: list of {"label", "score", "box"}; t: timestamp in seconds.
        Returns the CONFIRMED tracks that were matched in this frame.
        """
        cfg = self.cfg
        self.removed_this_frame = []

        for tr in self.tracks:
            tr.predict(t)

        high = [d for d in dets if d["score"] >= cfg.track_high_thresh]
        low = [d for d in dets if cfg.track_low_thresh <= d["score"] < cfg.track_high_thresh]

        # Stage 1: every track vs high-confidence boxes
        m1, unmatched_t, unmatched_hi = greedy_match(
            self._similarity(self.tracks, high), cfg.match_iou)
        for ti, di in m1:
            self.tracks[ti].apply(high[di], t, cfg)

        # Stage 2: leftover non-lost tracks vs low-confidence boxes
        rem = [self.tracks[i] for i in unmatched_t if self.tracks[i].state != LOST]
        m2, _, _ = greedy_match(self._similarity(rem, low), cfg.match_iou_low)
        for ti, di in m2:
            rem[ti].apply(low[di], t, cfg)

        # Misses and deletions
        survivors = []
        for tr in self.tracks:
            if tr.last_seen_t == t:
                survivors.append(tr)
                continue
            tr.streak = 0
            if tr.state == TENTATIVE:
                continue                                   # silent delete
            tr.state = LOST
            if t - tr.last_seen_t > cfg.max_age_s:
                if tr.ever_confirmed:
                    self.removed_this_frame.append(tr)     # becomes an EXIT event later
                continue
            survivors.append(tr)
        self.tracks = survivors

        # New tracks: only from unmatched HIGH boxes above new_track_thresh
        for di in unmatched_hi:
            det = high[di]
            if det["score"] < cfg.new_track_thresh:
                continue
            state = CONFIRMED if (self.fast_confirm and self.fast_confirm(det)) else TENTATIVE
            self.tracks.append(Track(self._next_id, det, t, state))
            self._next_id += 1

        return [tr for tr in self.tracks if tr.state == CONFIRMED and tr.last_seen_t == t]