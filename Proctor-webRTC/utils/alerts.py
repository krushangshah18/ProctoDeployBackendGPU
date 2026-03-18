from collections import deque
import time

from settings.alerts import WARN_DISPLAY_DURATION, ALERT_DISPLAY_DURATION


class AlertManager:
    """
    Two-tier alert system:

    warn(msg)  — soft on-screen warning (yellow).  NOT logged.  Not an API call.
                 Used for first-occurrence notices. OK to show frequently.

    alert(msg) — hard API event (red on screen + logged to report).
                 Treat as a backend API call — rate-limit via AlertEngine.
    """

    def __init__(self, warn_duration: float = WARN_DISPLAY_DURATION,
                 alert_duration: float = ALERT_DISPLAY_DURATION):
        self._warnings: deque = deque()
        self._alerts:   deque = deque()
        self.warn_duration  = warn_duration
        self.alert_duration = alert_duration

        # Set this to a callable(message) to capture alerts/warnings for the session report.
        self.on_alert = None
        self.on_warn  = None

    # ── Public write API ───────────────────────────────────────────────────

    def warn(self, message: str) -> None:
        """On-screen soft warning. No logging, no API call.
        If the same message is already active, refresh its timestamp instead of
        adding a duplicate — prevents the same warning from stacking on screen.
        """
        now = time.time()
        for entry in list(self._warnings):
            if entry["message"] == message:
                entry["timestamp"] = now
                return   # duplicate — refresh only, don't log again
        self._warnings.append({"message": message, "timestamp": now})
        if self.on_warn:
            self.on_warn(message)

    def alert(self, message: str) -> None:
        """On-screen hard alert + triggers on_alert callback (= API / report log)."""
        self._alerts.append({"message": message, "timestamp": time.time()})
        if self.on_alert:
            self.on_alert(message)

    # Backward-compat alias (used by audio event path)
    def add_alert(self, message: str) -> None:
        self.alert(message)

    # ── Public read API ────────────────────────────────────────────────────

    def get_active_warnings(self) -> list[str]:
        now = time.time()
        while self._warnings and now - self._warnings[0]["timestamp"] > self.warn_duration:
            self._warnings.popleft()
        return [w["message"] for w in self._warnings]

    def get_active_alerts(self) -> list[str]:
        now = time.time()
        while self._alerts and now - self._alerts[0]["timestamp"] > self.alert_duration:
            self._alerts.popleft()
        return [a["message"] for a in self._alerts]
