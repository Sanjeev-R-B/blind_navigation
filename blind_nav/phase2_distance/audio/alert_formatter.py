# phase2_distance/audio/alert_formatter.py

# Distance beyond which we don't alert
MAX_ALERT_DISTANCE = 3.0

# Minimum distance change (metres) before re-alerting same object
MIN_CHANGE_TO_ALERT = 0.6

# Object importance weights
# Scale: 1.0 = highest danger, 0.1 = lowest
# Vehicles > moving people > fixed structures > small objects
OBJECT_WEIGHTS = {
    # --- Vehicles (highest danger) ---
    "truck":      1.0,
    "car":        1.0,
    "bike":       0.9,   # motorbike / bicycle — both mapped here

    # --- People ---
    "person":     0.85,

    # --- Fixed obstacles ---
    "wall":       0.75,
    "door":       0.7,
    "chair":      0.65,
    "dining table":0.70,

    # --- Small / low-risk objects ---
    "backpack":   0.4,
    "book":       0.2,
    "cell phone": 0.15,
}

# Human-readable danger tier for TTS phrasing
def _tier(label):
    if label in ("truck", "car", "bike"):
        return "vehicle"
    if label in ("wall", "door"):
        return "obstacle"
    return "object"


class AlertFormatter:
    def __init__(self):
        # Track last alerted distance per label
        self._last_alerted = {}

    def calculate_risk_score(self, result):
        """
        Risk score formula:
            risk = distance_score * 0.6
                 + object_weight  * 0.3
                 + position_score * 0.1

        distance_score : 1.0 (≤0.8 m)  | 0.7 (≤1.5 m)  | 0.4 (≤3.0 m)
        object_weight  : from OBJECT_WEIGHTS table
        position_score : centre=1.0, left/right=0.5
        """
        distance = result["distance"]
        label    = result["label"]
        zone     = result["zone"]

        # Distance score
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
        Clipping alert — object is too close for distance to be measured.
        Highest priority; always spoken.
        """
        tier = _tier(label)
        if tier == "vehicle":
            msg = f"Danger! {label.capitalize()} right in front of you! Stop now."
        elif tier == "obstacle":
            msg = f"{label.capitalize()} is directly ahead! Stop immediately."
        else:
            msg = f"{label.capitalize()} is too close! Stop immediately."
        return msg, True   # (text, is_clip)

    def format_distance(self, result):
        """
        Normal distance alert.
        Returns (text, is_clip) tuple, or None if suppressed.
        """
        label    = result["label"]
        distance = result["distance"]
        zone     = result["zone"]

        if distance is None or not result.get("reliable", False):
            return None
        if distance > MAX_ALERT_DISTANCE:
            return None

        last = self._last_alerted.get(label)
        if last is not None and abs(last - distance) < MIN_CHANGE_TO_ALERT:
            return None

        self._last_alerted[label] = distance
        zone_str = f" on the {zone}" if zone != "centre" else ""
        tier     = _tier(label)

        # --- DANGER zone ≤ 0.8 m ---
        if distance <= 0.8:
            if tier == "vehicle":
                text = (
                    f"Danger! {label.capitalize()} very close, "
                    f"only {distance:.1f} meters{zone_str}. Stop!"
                )
            else:
                text = (
                    f"Warning! {label.capitalize()} is very close, "
                    f"only {distance:.1f} meters{zone_str}."
                )

        # --- CAUTION zone 0.8 – 1.5 m ---
        elif distance <= 1.5:
            if tier == "vehicle":
                text = (
                    f"Caution! {label.capitalize()} approaching, "
                    f"{distance:.1f} meters{zone_str}."
                )
            else:
                text = (
                    f"Caution. {label.capitalize()} "
                    f"at {distance:.1f} meters{zone_str}."
                )

        # --- SAFE zone 1.5 – 3.0 m ---
        else:
            text = (
                f"{label.capitalize()} detected "
                f"at {distance:.1f} meters{zone_str}."
            )

        return text, False

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def format_all(self, estimates, clipping_labels):
        """
        Build the final alert list, ordered by risk:
          1. Clipping alerts (immeasurably close) — always first
          2. Distance alerts — sorted by risk score (highest first)
        """
        alerts = []

        # 1. Clipping alerts
        for label in clipping_labels:
            alerts.append(self.format_clip(label))

        # 2. Distance alerts (skip anything already clipping)
        distance_alerts = []
        for result in estimates:
            if result["label"] in clipping_labels:
                continue
            formatted = self.format_distance(result)
            if formatted:
                distance_alerts.append({
                    "alert": formatted,
                    "risk":  self.calculate_risk_score(result)
                })

        distance_alerts.sort(key=lambda x: x["risk"], reverse=True)
        alerts.extend(item["alert"] for item in distance_alerts)

        return alerts