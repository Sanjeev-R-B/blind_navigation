# Distance beyond which we don't alert — not immediately dangerous
MAX_ALERT_DISTANCE = 5.0

# Minimum distance change (metres) before re-alerting same object
MIN_CHANGE_TO_ALERT = 0.5


class AlertFormatter:
    def __init__(self):
        # Track last alerted distance per label to avoid repetition
        self._last_alerted = {}

    def format_clip(self, label):
        """Format a too-close / clipping alert."""
        return f"{label.capitalize()}, very close, stop", True  # (text, priority)

    def format_distance(self, result):
        """
        Format a normal distance alert.

        Args:
            result : dict from pinhole_estimator.estimate()

        Returns:
            (text, priority) tuple or None if alert should be suppressed
        """
        label    = result['label']
        distance = result['distance']
        zone     = result['zone']

        # Skip if unknown or unreliable
        if distance is None or not result['reliable']:
            return None

        # Skip if too far
        if distance > MAX_ALERT_DISTANCE:
            return None

        # Suppress if distance hasn't changed enough since last alert
        last = self._last_alerted.get(label)
        if last is not None and abs(last - distance) < MIN_CHANGE_TO_ALERT:
            return None

        # Update last alerted distance
        self._last_alerted[label] = distance

        # Format zone — skip "centre" to keep speech concise
        zone_str = f", {zone}" if zone != 'centre' else ''

        text = f"{label.capitalize()}, {distance} metres{zone_str}"
        return text, False  # (text, priority=False)

    def format_all(self, estimates, clipping_labels):
        """
        Generate all alerts for a single frame.

        Args:
            estimates       : list from pinhole_estimator.estimate_all()
            clipping_labels : list from clip_detector.check_all()

        Returns:
            list of (text, priority) tuples, priority alerts first
        """
        alerts = []

        # Clipping alerts first — highest priority
        for label in clipping_labels:
            alerts.append(self.format_clip(label))

        # Distance alerts — already sorted nearest first
        for result in estimates:
            if result['label'] in clipping_labels:
                continue  # already handled above
            formatted = self.format_distance(result)
            if formatted:
                alerts.append(formatted)

        return alerts


