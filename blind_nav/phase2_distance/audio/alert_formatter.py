# phase2_distance/audio/alert_formatter.py

# Distance beyond which we don't alert
MAX_ALERT_DISTANCE = 3.0

# Minimum distance change (metres) before re-alerting same object
MIN_CHANGE_TO_ALERT = 0.6

# Object importance weights
OBJECT_WEIGHTS = {
    "person": 1.0,
    "chair": 0.8,
    "backpack": 0.6,
    "book": 0.3,
    "cell phone": 0.2
}


class AlertFormatter:
    def __init__(self):
        # Track last alerted distance per label
        self._last_alerted = {}

    def calculate_risk_score(self, result):
        """
        Calculate risk score using:
        distance + object importance + position
        """
        distance = result["distance"]
        label = result["label"]
        zone = result["zone"]

        # Distance score (most important)
        if distance <= 0.8:
            distance_score = 1.0
        elif distance <= 1.5:
            distance_score = 0.7
        elif distance <= 3.0:
            distance_score = 0.4
        else:
            distance_score = 0.0

        # Object importance
        object_weight = OBJECT_WEIGHTS.get(label, 0.5)

        # Position importance
        position_score = {
            "centre": 1.0,
            "left": 0.5,
            "right": 0.5
        }.get(zone, 0.5)

        risk_score = (
            distance_score * 0.6 +
            object_weight * 0.3 +
            position_score * 0.1
        )

        return risk_score

    def format_clip(self, label):
        """
        Format a clipping alert.
        Highest priority alert.
        """
        return f"{label.capitalize()} is too close! Stop immediately.", True

    def format_distance(self, result):
        """
        Format a normal distance alert.
        Returns (text, priority) tuple or None.
        """
        label = result["label"]
        distance = result["distance"]
        zone = result["zone"]

        # Skip if unreliable
        if distance is None or not result.get("reliable", False):
            return None

        # Skip if too far
        if distance > MAX_ALERT_DISTANCE:
            return None

        # Avoid repeating same alert too often
        last = self._last_alerted.get(label)
        if last is not None and abs(last - distance) < MIN_CHANGE_TO_ALERT:
            return None

        self._last_alerted[label] = distance

        zone_str = f" on the {zone}" if zone != "centre" else ""

        if distance <= 0.8:
            text = (
                f"Warning! {label.capitalize()} is very close, "
                f"only {distance:.1f} meters{zone_str}."
            )
        elif distance <= 1.5:
            text = (
                f"Caution. {label.capitalize()} "
                f"at {distance:.1f} meters{zone_str}."
            )
        else:
            text = (
                f"{label.capitalize()} detected "
                f"at {distance:.1f} meters{zone_str}."
            )

        return text, False

    def format_all(self, estimates, clipping_labels):
        """
        Generate alerts sorted by risk score.
        """
        alerts = []

        # 1. Clipping alerts first
        for label in clipping_labels:
            alerts.append(self.format_clip(label))

        # 2. Distance alerts
        distance_alerts = []

        for result in estimates:
            if result["label"] in clipping_labels:
                continue

            formatted = self.format_distance(result)

            if formatted:
                risk_score = self.calculate_risk_score(result)

                distance_alerts.append({
                    "alert": formatted,
                    "risk": risk_score
                })

        # Highest risk first
        distance_alerts.sort(
            key=lambda x: x["risk"],
            reverse=True
        )

        for item in distance_alerts:
            alerts.append(item["alert"])

        return alerts