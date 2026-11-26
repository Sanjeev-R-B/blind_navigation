# tracking/

Persistent object IDs and ENTER / EXIT events for the blind-nav pipeline.

Without tracking, every frame is independent: a person who walks toward the camera is a new detection each time, so the system cannot say how long an object has been there, how fast it is approaching, or whether it has already been announced. This folder gives every object a stable ID and a history, which the next objectives (approach speed / time-to-collision and per-object alert cooldown) are built on.

Pure Python + NumPy. No torch, no scipy, no ultralytics, so the same code can run on the desktop and be ported to the Android app.

---

## Files

| File | What it does |
|---|---|
| `tracker.py` | `Tracker`: two-stage IoU association, Kalman prediction, track lifecycle. Also `Track`, `TrackerConfig`, `BoxKalman` |
| `events.py` | `EventEmitter`: turns lifecycle changes into ENTER / EXIT event records |
| `__init__.py` | Empty, makes the folder importable |

---

## Where it sits in the pipeline

```
camera frame
  -> YOLODetector.detect(frame, include_low=True)     boxes from 0.10 confidence upward
  -> Tracker.update(detections, t)                    stable IDs, confirmed tracks
  -> estimate(label, box, frame_w) per track          distance in metres
  -> track.add_history(t, distance)                   registry / distance history
  -> EventEmitter.process(...)                        ENTER / EXIT events
  -> (next) TTC and per-object alert cooldown
```

Distance, zones and speech in `run_phase2.py` still use the plain detections list. The tracker runs next to them and does not change their behaviour yet.

---

## Quick start

```python
from phase2_distance.tracking.tracker import Tracker
from phase2_distance.tracking.events import EventEmitter

tracker = Tracker()
emitter = EventEmitter(frame_w, frame_h, max_distance=3.0)

# every frame
t_now      = time.time()
all_dets   = detector.detect(frame, include_low=True)
tracks     = tracker.update(all_dets, t_now)             # confirmed tracks seen this frame

for tr in tracks:
    tr.add_history(t_now, estimate(tr.label, tr.box, frame_w)['distance'])

for e in emitter.process(tracks, tracker.removed_this_frame, t_now, frame_index):
    print(e)
```

Call `add_history` before `emitter.process`, because ENTER is held back until the track has a distance within range.

---

## Input and output

**Input to `Tracker.update(dets, t)`**

- `dets`: list of `{"label": str, "score": float, "box": (x1, y1, x2, y2)}` in pixels. This is exactly what `YOLODetector.detect(frame, include_low=True)` returns.
- `t`: timestamp in seconds, for example `time.time()`. Everything is time-based, so FPS can change without breaking anything.

**Output:** the list of `Track` objects that are CONFIRMED and were matched in this frame. All live tracks (including tentative and lost ones) are in `tracker.tracks`; `tracker.registry` gives them as `{track_id: track}`.

**`Track` fields and helpers**

| Member | Meaning |
|---|---|
| `id` | Persistent integer ID |
| `box` | Kalman-smoothed box `(x1, y1, x2, y2)` (use this for distance) |
| `last_bbox` | Last box that was actually detected (not predicted) |
| `label` | Class with the highest summed confidence (`class_votes`) |
| `score` | Confidence of the latest matched detection |
| `state` | `tentative`, `confirmed` or `lost` |
| `hits` / `streak` | Total matched frames / consecutive matched frames |
| `first_seen_t` / `last_seen_t` | Times in seconds |
| `class_votes` | `{class: summed confidence}`, so a flickering label does not change the track's class |
| `history` | Rolling list of `(t, bbox, distance)` (last 150 readings) |
| `add_history(t, distance)` | Record a reading (`distance` may be `None`) |
| `last_distance` | Most recent non-`None` distance |
| `distance_window(window_s, now)` | `[(t, distance)]` from the last `window_s` seconds, oldest first (used for TTC) |

---

## How tracking works

**1. Predict.** Each track has a Kalman filter on `[cx, cy, w, h, vx, vy, vw, vh]`. Before matching, every track is moved forward by the elapsed time since the last frame. A lost track's velocity is damped so it does not drift away.

**2. Two-stage matching.** Matching is greedy by highest IoU (no scipy).

- Stage 1: high-confidence boxes (>= `track_high_thresh`) against all tracks, minimum IoU `match_iou`.
- Stage 2: low-confidence boxes (`track_low_thresh` to `track_high_thresh`) against the tracks still unmatched and not lost, minimum IoU `match_iou_low`. This keeps an ID alive when a score dips for a moment.
- If the box and track classes differ, the IoU is multiplied by `cross_class_penalty`, so two overlapping objects of different classes are less likely to swap IDs.

**3. New tracks.** Only an unmatched high-confidence box with a score of at least `new_track_thresh` starts a track. Low-confidence boxes can never create one.

**4. Lifecycle.**

```
            n_init matched frames in a row
TENTATIVE ------------------------------------> CONFIRMED
    |                                              |   ^
    | misses a frame                    miss a     |   | re-matched within max_age_s
    v                                   frame      v   |
 deleted silently                                  LOST
                                                     |
                                   unmatched for > max_age_s
                                                     v
                                                  deleted  (EXIT event)
```

- A tentative track that fails early disappears with no event.
- `max_age_s` is in seconds, not frames. In the original BoT-SORT settings `track_buffer / fps` must be at least the memory time you want; using seconds removes that check.
- `Tracker(fast_confirm=fn)`: if `fn(det)` returns `True`, a new track starts CONFIRMED immediately and skips the `n_init` wait. Intended for objects already in the DANGER zone. Not wired in yet.

---

## Events

`EventEmitter(frame_w, frame_h, edge_margin=0.03, max_distance=None)`

| Event | When it fires |
|---|---|
| `ENTER` | Once, when a track is CONFIRMED. With `max_distance` set, it is held back until the track's distance is within range (an unknown distance is not held back) |
| `EXIT` | Once, when a confirmed track is deleted. Only for tracks that had an ENTER, so objects that were never announced never "exit" |

**Exit reason**

| `exit_reason` | Meaning |
|---|---|
| `left_frame` | The last detected box was within `edge_margin` (3%) of the left, right or top border |
| `lost` | Vanished in the middle of the frame (occlusion or missed detections) |

The bottom border is ignored on purpose: a close object or a seated person always touches it, so it says nothing about leaving.

**Event record**

```
{ event: ENTER | EXIT,
  track_id, class, confidence, distance,
  timestamp, frame_index, bbox, exit_reason (EXIT only) }
```

---

## Configuration (`TrackerConfig`)

| Setting | Default | Effect |
|---|---|---|
| `track_high_thresh` | 0.25 | Boxes at or above this are high confidence |
| `track_low_thresh` | 0.10 | Boxes below this are ignored completely |
| `new_track_thresh` | 0.25 | Minimum score to start a track |
| `match_iou` | 0.30 | Minimum overlap in stage 1 |
| `match_iou_low` | 0.50 | Minimum overlap in stage 2 |
| `n_init` | 3 | Matched frames in a row before ENTER |
| `max_age_s` | 1.5 | Seconds a lost track is remembered |
| `cross_class_penalty` | 0.5 | IoU multiplier when classes differ |

Not included from the BoT-SORT settings: camera-motion compensation (`gmc_method`) and re-identification (`with_reid`). Both stay off until ID switches show they are needed.

### Tuning guide

| Symptom | Try |
|---|---|
| ID changes when you move or lean fast | Lower `match_iou` (for example 0.2) |
| Same object gets a new ID after a short disappearance | Raise `max_age_s`, lower `match_iou` |
| Ghost track stays after the object left | Lower `max_age_s` |
| ENTER / EXIT spam from flickering objects | Raise `n_init` or `new_track_thresh`, or set `max_distance` on the emitter |
| Small objects never get tracked | Lower `new_track_thresh` carefully, since false positives rise |

The detector must be called with `include_low=True`, otherwise stage 2 has no boxes to work with.

---

## Tests run

All of the following were run against simulated detections at 20 FPS:

| Case | Result |
|---|---|
| Person walking 140 frames, 4-frame confidence dip to 0.18, 6-frame full miss | Same ID the whole time |
| Object appears | Confirmed after exactly 3 frames |
| Label flickering between two classes | No new ID |
| One-frame false positive | No track, no event |
| Object vanishes mid-frame | EXIT `lost` after 1.5 s |
| Person walks off the right edge | EXIT `left_frame` |
| Two people crossing paths | No ID swap |
| Far object approaching | ENTER only once it is within 3 m |

On the webcam, steps 1 to 3 (IDs on screen, ENTER, EXIT) were checked. The distance history and distance-gated ENTER (step 4) still need a webcam run.

**Webcam checklist**

1. Sit in view: one `ENTER #1 person` within a quarter of a second, and no more after that.
2. Lean and move: the ID stays the same.
3. Walk out through a side: `EXIT ... left_frame` after about 1.5 s.
4. Cover the lens: `EXIT ... lost` after about 1.5 s.
5. A far chair (6 m or more) produces no events; one inside 3 m produces a single ENTER.

---

## Known limitations

- Matching uses box overlap only. A fast lean, or two people overlapping, can still switch an ID.
- `left_frame` only checks that the box touches a border. An object that sits against an edge for its whole life gets `left_frame` when it disappears. A direction-aware check (using the last velocity) is planned.
- Greedy matching is used instead of the Hungarian algorithm, which is slightly less optimal when many boxes overlap.
- `last_seen_frame` from the original registry design is not stored yet.
- Run time per frame is still being measured, because FPS dropped to about 15 in one run after low-confidence boxes were enabled. Compare the `detect` and `tracker` times before changing anything.

---

## Next

1. Approach speed and time-to-collision, using `track.distance_window()`.
2. Per-object alert cooldown keyed by track ID, replacing the label-based rules in `alert_formatter.py`.
3. Skip the confirmation wait for DANGER-zone objects through `fast_confirm`.
4. Port the tracker to the Android app.