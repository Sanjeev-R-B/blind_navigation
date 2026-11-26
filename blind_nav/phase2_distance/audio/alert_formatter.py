# phase2_distance/audio/alert_formatter.py

import time

# Distance beyond which we don't alert
MAX_ALERT_DISTANCE = 3.0

# Minimum distance change (metres) before re-alerting same object
MIN_CHANGE_TO_ALERT = 0.6

# Cooldown between clip alerts for the same label (seconds).
# 5 s — short enough to re-warn if the object is still there,
# long enough to avoid back-to-back repeats.
# TODO (tracker phase): switch key from label → track_id, and add a
# velocity override — if closing speed > ~0.8 m/s, skip cooldown entirely
# so a fast-approaching vehicle triggers immediately on every frame cycle.
CLIP_COOLDOWN_S = 5.0

# Zone boundaries (keep in sync with run_phase2.py)
DANGER_DISTANCE  = 0.8
CAUTION_DISTANCE = 1.5

# Still in DANGER with nothing changing: repeat this often (seconds)
DANGER_REPEAT_S = 3.5

# Forget an object not seen for this long, so it alerts again as "new"
FORGET_AFTER_S = 3.0
# Object importance weights
# Scale: 1.0 = highest danger, 0.1 = lowest
# Vehicles > moving people > fixed structures > small objects
OBJECT_WEIGHTS = {
    # --- Vehicles (highest danger) ---
    "truck":        1.0,
    "car":          1.0,
    "bike":         0.9,   # motorbike / bicycle — both mapped here

    # --- People ---
    "person":       0.85,

    # --- Fixed obstacles ---
    "wall":         0.75,
    "door":         0.7,
    "chair":        0.65,
    "dining table": 0.70,

    # --- Small / low-risk objects ---
    "backpack":     0.4,
    "book":         0.2,
    "cell phone":   0.15,
}


# Human-readable danger tier for TTS phrasing
def _tier(label):
    if label in ("truck", "car", "bike"):
        return "vehicle"
    if label in ("wall", "door"):
        return "obstacle"
    return "object"

def _zone_rank(distance):
    """0 = DANGER, 1 = CAUTION, 2 = SAFE. Lower means closer."""
    if distance <= DANGER_DISTANCE:
        return 0
    if distance <= CAUTION_DISTANCE:
        return 1
    return 2

class AlertFormatter:
    def __init__(self):
        # Per-label state: rank, dist, t_alert, t_seen
        # TODO: key by track_id once the tracker is in
        self._state: dict[str, dict] = {}

        # Last clip alert timestamp per label (time-based cooldown)
        # TODO: swap key to track_id once tracker is in
        self._last_clip_time: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  Risk scoring                                                        #
    # ------------------------------------------------------------------ #

    def calculate_risk_score(self, result):
        """
        risk = distance_score * 0.6
             + object_weight  * 0.3
             + position_score * 0.1

        distance_score : 1.0 (≤0.8 m) | 0.7 (≤1.5 m) | 0.4 (≤3.0 m)
        object_weight  : from OBJECT_WEIGHTS table
        position_score : centre=1.0, left/right=0.5
        """
        distance = result["distance"]
        label    = result["label"]
        zone     = result["zone"]

        if distance <= 0.8:
            distance_score = 1.0
        elif distance <= 1.5:
            distance_score = 0.7
        elif distance <= 3.0:
            distance_score = 0.4
        else:
            distance_score = 0.0

        object_weight  = OBJECT_WEIGHTS.get(label, 0.5)
        position_score = 1.0 if zone == "centre" else 0.5

        return (
            distance_score * 0.6 +
            object_weight  * 0.3 +
            position_score * 0.1
        )

    # ------------------------------------------------------------------ #
    #  Alert formatters                                                    #
    # ------------------------------------------------------------------ #

    def format_clip(self, label):
        """
        Clipping alert — object is filling the bottom of the frame and
        is under 0.5 m away.  Highest priority; clears the TTS queue.

        Returns (text, is_priority) or None if still inside cooldown.
        """
        now = time.time()
        last = self._last_clip_time.get(label, 0.0)
        if now - last < CLIP_COOLDOWN_S:
            return None          # still in cooldown — skip

        self._last_clip_time[label] = now

        tier = _tier(label)
        if tier == "vehicle":
            msg = f"Danger! {label.capitalize()} right in front of you! Stop now."
        elif tier == "obstacle":
            msg = f"{label.capitalize()} is directly ahead! Stop immediately."
        else:
            # Change 5: "stop" for the real clip case
            msg = f"{label.capitalize()}, stop!"
        return msg, True   # (text, is_priority)

    def format_distance(self, result):
        """
        Normal distance alert.
        Returns (text, is_priority) or None if suppressed.

        Change 5 wording:
          DANGER  (≤0.8 m) → "Person, very close"
          CAUTION (≤1.5 m) → "Person, 1.2 metres"
          SAFE    (≤3.0 m) → "Person detected at 2.5 metres"
        """
        label    = result["label"]
        distance = result["distance"]
        zone     = result["zone"]

        if distance is None or not result.get("reliable", False):
            return None
        if distance > MAX_ALERT_DISTANCE:
            self._state.pop(label, None)   # out of range: forget, so re-entry alerts
            return None

        now  = time.time()
        rank = _zone_rank(distance)
        st   = self._state.get(label)

        if st is None:
            speak = True                                   # new object
        else:
            st["t_seen"] = now
            if rank < st["rank"]:
                speak = True                               # moved into a closer zone
            elif abs(st["dist"] - distance) >= MIN_CHANGE_TO_ALERT:
                speak = True                               # distance changed enough
            elif rank == 0 and now - st["t_alert"] >= DANGER_REPEAT_S:
                speak = True                               # still in DANGER: repeat
            else:
                speak = False                              # nothing new

        if not speak:
            return None

        self._state[label] = {"rank": rank, "dist": distance,
                              "t_alert": now, "t_seen": now}
        zone_str = f" on the {zone}" if zone != "centre" else ""
        tier     = _tier(label)

        # --- DANGER zone ≤ 0.8 m ---
        if distance <= 0.8:
            if tier == "vehicle":
                text = (
                    f"Danger! {label.capitalize()} very close, "
                    f"{distance:.1f} metres{zone_str}. Stop!"
                )
            else:
                # Clean, concise: "Person, very close"
                text = f"{label.capitalize()}, very close{zone_str}."

        # --- CAUTION zone 0.8 – 1.5 m ---
        elif distance <= 1.5:
            if tier == "vehicle":
                text = (
                    f"Caution! {label.capitalize()} approaching, "
                    f"{distance:.1f} metres{zone_str}."
                )
            else:
                # Clean, concise: "Person, 1.2 metres"
                text = f"{label.capitalize()}, {distance:.1f} metres{zone_str}."

        # --- SAFE zone 1.5 – 3.0 m ---
        else:
            text = (
                f"{label.capitalize()} detected "
                f"at {distance:.1f} metres{zone_str}."
            )

        return text, rank == 0       # DANGER alerts are priority

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def format_all(self, estimates, clipping_labels):
        """
        Build the final alert list, ordered by risk:
          1. Clipping alerts (immeasurably close, under 0.5 m) — always first
          2. Distance alerts for ALL objects including clipping ones
             (Change 4: no longer skip clipping labels here)
        """
        alerts = []
        now = time.time()
        for lbl in [l for l, s in self._state.items() if now - s["t_seen"] > FORGET_AFTER_S]:
            del self._state[lbl]
        # 1. Clip alerts — subject to per-label cooldown
        for label in clipping_labels:
            result = self.format_clip(label)
            if result is not None:
                alerts.append(result)

        # 2. Distance alerts — ALL objects, sorted by risk score
        # Change 4: removed the `if result["label"] in clipping_labels: continue` guard
        distance_alerts = []
        for result in estimates:
            formatted = self.format_distance(result)
            if formatted:
                distance_alerts.append({
                    "alert": formatted,
                    "risk":  self.calculate_risk_score(result),
                })

        distance_alerts.sort(key=lambda x: x["risk"], reverse=True)
        alerts.extend(item["alert"] for item in distance_alerts)

        return alerts
