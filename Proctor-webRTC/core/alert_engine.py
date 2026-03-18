"""
AlertEngine — routes RiskEvent to warn or alert.

Rule:
  risk_added == 0  →  warn()   soft amber banner
  risk_added  > 0  →  alert()  red banner + API log

Grace (occ=1 of occurrence-based events) is enforced in RiskEngine by arming
the score cooldown without adding score — so AlertEngine always sees
risk_added=0 during grace and routes it to warn automatically.

Proof capture (snapshots / video) is handled entirely in main.py via ProofWriter.
"""

import time

import settings.alerts as A
from .risk_engine import RiskEvent


class AlertEngine:
    """
    Consumes RiskEvent objects (from RiskEngine) and decides whether to
    show a warning or fire an API alert.
    """

    def __init__(self) -> None:
        # API-alert cooldowns: key → earliest wall-time for next API alert
        self._api_cooldown_until:  dict[str, float] = {}
        # Warn cooldowns: key → earliest wall-time for next soft warning
        self._warn_cooldown_until: dict[str, float] = {}
        # Termination alert fired flag — only emit once
        self._termination_alerted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def handle(self, event: RiskEvent, alert_manager) -> None:
        """
        Main entry point. Call once per frame per detection key.

        event         — RiskEvent returned by RiskEngine.process_event()
        alert_manager — AlertManager instance
        """
        key = event.key

        # ── Terminated ────────────────────────────────────────────────────────
        if event.terminated:
            if not self._termination_alerted:
                self._termination_alerted = True
                alert_manager.alert(f"EXAM TERMINATED: {event.termination_reason}")
            return

        # ── Inactive: nothing to show ─────────────────────────────────────────
        if not event.active and not event.is_new_occurrence:
            return

        warn_msg  = A.WARN_MESSAGES.get(key,  key)
        alert_msg = A.ALERT_MESSAGES.get(key, key)
        now       = time.time()

        # ── No score added → WARNING ──────────────────────────────────────────
        if event.risk_added == 0:
            if self._warn_ok(key, now):
                alert_manager.warn(warn_msg)
                self._arm_warn_cooldown(key, now)
            return

        # ── Score added → ALERT ───────────────────────────────────────────────
        api_due = self._api_cooldown_until.get(key, 0.0)
        if now >= api_due:
            score_tag = f"  [+{event.risk_added:.0f} pts]"
            alert_manager.alert(f"{alert_msg}{score_tag}")
            self._api_cooldown_until[key] = now + A.API_COOLDOWNS.get(key, 10)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _warn_ok(self, key: str, now: float) -> bool:
        return now >= self._warn_cooldown_until.get(key, 0.0)

    def _arm_warn_cooldown(self, key: str, now: float) -> None:
        cd = A.WARN_COOLDOWNS.get(key, 5.0)
        self._warn_cooldown_until[key] = now + cd
