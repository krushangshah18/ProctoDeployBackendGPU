"""
Risk Scoring Engine — implements proctoring_risk_scoring_design.md

Single source of truth for:
  - Occurrence counts (rising-edge detections)
  - Two-bucket score: _fixed_score (non-decaying) + _decaying_score (decaying)
  - Risk scoring cooldowns (distinct from API-alert cooldowns in AlertEngine)
  - Score decay, combo bonuses, state machine, termination rules
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import settings.scoring as S


# ── State machine ─────────────────────────────────────────────────────────────

class ExamState(Enum):
    NORMAL       = "NORMAL"
    WARNING      = "WARNING"
    HIGH_RISK    = "HIGH_RISK"
    ADMIN_REVIEW = "ADMIN_REVIEW"
    TERMINATED   = "TERMINATED"


# ── RiskEvent — returned by process_event() ───────────────────────────────────

@dataclass
class RiskEvent:
    """Result of processing one event for one frame.

    AlertEngine reads this to decide warn vs API-alert.
    """
    key: str
    active: bool             = False  # current condition state
    risk_added: float        = 0.0    # score added this call (0 = warning zone / cooldown)
    is_new_occurrence: bool  = False  # True on the rising edge frame
    occurrence_count: int    = 0      # total rising-edge detections for this key
    duration: float          = 0.0    # seconds this active period has been running
    terminated: bool         = False
    termination_reason: str  = ""


# ── Constants — all values sourced from settings.scoring ──────────────────────
# Edit settings/scoring.py to change any threshold, score, or cooldown.

_GAZE_EVENTS = {"looking_away", "looking_down", "looking_up", "looking_side"}


# ── Engine ────────────────────────────────────────────────────────────────────

class RiskEngine:

    def __init__(self, session_duration_s: float = 3600.0,
                 flicker_grace_s: float = 1.5):
        # Two-bucket score
        self._fixed_score:    float = 0.0   # non-decaying
        self._decaying_score: float = 0.0   # decays every interval

        self.state     = ExamState.NORMAL
        self.terminated          = False
        self._termination_reason = ""

        # Decay config
        self._session_duration_s = session_duration_s
        self._decay_interval     = max(60.0, session_duration_s / 20.0)
        self._creation_time      = time.time()
        self._last_decay_time    = time.time()

        # Decay audit log: each tick recorded for the report
        self.decay_log: list[dict] = []

        # Occurrence tracking: rising-edge counts per key (single source of truth)
        self._occurrences: dict[str, int] = {}

        # Previous-frame active state (for edge detection)
        self._prev_active: dict[str, bool] = {}

        # Score-cooldown: key → timestamp after which scoring is allowed again
        self._score_cooldown_until: dict[str, float] = {}

        # Duration tracking: key → wall-time start of current active period
        self._active_since: dict[str, Optional[float]] = {}

        # Gaze aggregation (30-second window).
        # Stored as a deque so old entries are dropped without list rebuild.
        self._gaze_timestamps:      deque[float] = deque()
        self._last_gaze_bonus_time: float        = 0.0

        # Continuous-presence timers for termination rules
        self._multi_people_since: Optional[float] = None
        self._no_person_since:    Optional[float] = None

        # Frozen durations — set at termination so the overlay timer stops ticking
        self._frozen_multi_dur:     float = 0.0
        self._frozen_no_person_dur: float = 0.0

        # Flicker grace: timestamp when condition first went inactive.
        # Timer only resets after condition stays inactive for this many seconds.
        self._flicker_grace_s:          float          = flicker_grace_s
        self._multi_people_gone_since:  Optional[float] = None
        self._no_person_gone_since:     Optional[float] = None

        # Speaker audio continuous-presence timer (with flicker grace + full reset)
        self._speaker_since:      Optional[float] = None
        self._speaker_gone_since: Optional[float] = None
        # True episode counter — increments only on full silence reset (> flicker_grace).
        # Brief dropouts (flicker grace) do NOT increment this.
        self._speaker_occ:        int             = 0

        # Currently-active event set (for combo detection)
        self._active_set: set[str] = set()

        # Combo cooldowns: combo_key → earliest time combo can fire again
        self._combo_cooldowns: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def score(self) -> float:
        return self._fixed_score + self._decaying_score

    def continuous_duration(self, key: str) -> float:
        """Seconds the key has been continuously active (0.0 if not active)."""
        if self.terminated:
            if key == "multiple_people":
                return self._frozen_multi_dur
            if key == "no_person":
                return self._frozen_no_person_dur
            return 0.0
        now = time.time()
        if key == "multiple_people":
            return (now - self._multi_people_since) if self._multi_people_since else 0.0
        if key == "no_person":
            return (now - self._no_person_since) if self._no_person_since else 0.0
        return 0.0

    def process_event(self, key: str, active: bool,
                      confidence: float = 1.0,
                      duration: float = 0.0,
                      now: float = 0.0) -> RiskEvent:
        """
        Call once per frame per detection key.

        key        — event identifier
        active     — whether the condition is currently true (after duration-gating)
        confidence — detection confidence (YOLO: from model; head/gaze: 1.0)
        duration   — how long this active period has been running (from HeadTracker
                     or internal timer if 0 is passed for object events)
        now        — current wall time (time.time()); pass the tick's already-computed
                     value to avoid a redundant syscall for each of the 12+ events
                     processed per tick. Defaults to 0.0 which triggers an internal call.

        Returns a RiskEvent the AlertEngine uses to decide warn vs API-alert.
        """
        if self.terminated:
            return RiskEvent(key=key, terminated=True,
                             termination_reason=self._termination_reason)

        if now <= 0.0:
            now = time.time()
        prev = self._prev_active.get(key, False)
        self._prev_active[key] = active

        # ── Edge detection ────────────────────────────────────────────────────
        is_new_occurrence = active and not prev

        if is_new_occurrence:
            self._occurrences[key] = self._occurrences.get(key, 0) + 1

        # ── Active-set maintenance (for combos) ───────────────────────────────
        if active:
            self._active_set.add(key)
        else:
            self._active_set.discard(key)

        # ── Duration tracking (internal fallback when caller passes 0) ────────
        if active:
            if self._active_since.get(key) is None:
                self._active_since[key] = now
            # Prefer caller-supplied duration (more accurate for head events)
            eff_duration = duration if duration > 0 else now - self._active_since[key]
        else:
            self._active_since[key] = None
            eff_duration = 0.0

        occ = self._occurrences.get(key, 0)

        # ── Decay (applied every frame, limited by interval) ──────────────────
        self._apply_decay(now)

        # ── Inactive: reset continuous timers (with flicker grace) ───────────
        if not active:
            if key == "multiple_people" and self._multi_people_since is not None:
                if self._multi_people_gone_since is None:
                    self._multi_people_gone_since = now
                elif now - self._multi_people_gone_since >= self._flicker_grace_s:
                    self._multi_people_since     = None
                    self._multi_people_gone_since = None
            elif key == "no_person" and self._no_person_since is not None:
                if self._no_person_gone_since is None:
                    self._no_person_gone_since = now
                elif now - self._no_person_gone_since >= self._flicker_grace_s:
                    self._no_person_since     = None
                    self._no_person_gone_since = None
            elif key == "speaker_audio" and self._speaker_since is not None:
                if self._speaker_gone_since is None:
                    self._speaker_gone_since = now
                elif now - self._speaker_gone_since >= self._flicker_grace_s:
                    # Full silence — end this episode. Increment occ counter and
                    # clear tier gates so the next episode starts fresh.
                    self._speaker_since      = None
                    self._speaker_gone_since = None
                    self._speaker_occ       += 1
                    self._score_cooldown_until.pop("_speaker_t1",     None)
                    self._score_cooldown_until.pop("_speaker_repeat",  None)
            self._update_state()
            return RiskEvent(key=key, active=False, occurrence_count=occ)

        # ── Termination / special-duration checks ─────────────────────────────
        term_event = self._handle_special(key, active, occ, eff_duration,
                                          confidence, now)
        if term_event is not None:
            return term_event

        # ── Risk scoring ───────────────────────────────────────────────────────
        risk_added = self._score_event(key, confidence, eff_duration, occ, now)

        # ── Gaze aggregation bonus ─────────────────────────────────────────────
        if key in _GAZE_EVENTS and risk_added > 0:
            self._gaze_timestamps.append(now)
            while self._gaze_timestamps and now - self._gaze_timestamps[0] > S.GAZE_WINDOW_S:
                self._gaze_timestamps.popleft()
            if (len(self._gaze_timestamps) >= S.GAZE_BONUS_MIN_EVENTS
                    and now - self._last_gaze_bonus_time > S.GAZE_WINDOW_S):
                self._add(S.GAZE_BONUS_SCORE, decaying=True)
                self._last_gaze_bonus_time = now

        # ── Combo bonuses ──────────────────────────────────────────────────────
        self._check_combos(key, confidence, now)

        self._update_state()

        return RiskEvent(
            key=key,
            active=True,
            risk_added=risk_added,
            is_new_occurrence=is_new_occurrence,
            occurrence_count=occ,
            duration=eff_duration,
        )

    def handle_tab_switch(self, now: float = 0.0) -> RiskEvent:
        """
        Process one discrete tab-switch event (exam tab left / window hidden).

        Each call represents one occurrence — no active/inactive cycle needed.

        Occurrence 1 to TERMINATE-1       → +TAB_SWITCH_SCORE pts fixed (alert)
        Occurrence >= TERMINATE_COUNT     → exam terminated immediately
        """
        if self.terminated:
            return RiskEvent(key="tab_switch", terminated=True,
                             termination_reason=self._termination_reason)

        if now <= 0.0:
            now = time.time()

        # Increment occurrence counter (this IS a rising-edge event every call)
        occ = self._occurrences.get("tab_switch", 0) + 1
        self._occurrences["tab_switch"] = occ

        # Termination rule
        if occ >= S.TAB_SWITCH_TERMINATE_COUNT:
            self._terminate(f"Tab switched {occ} time{'s' if occ != 1 else ''} — exam policy violated")
            return RiskEvent(key="tab_switch", terminated=True,
                             termination_reason=self._termination_reason,
                             occurrence_count=occ)

        # Every tab switch adds fixed (non-decaying) points immediately — no grace.
        risk_added = self._add(S.TAB_SWITCH_SCORE, decaying=False)

        self._update_state()
        return RiskEvent(
            key="tab_switch",
            active=True,
            risk_added=risk_added,
            is_new_occurrence=True,
            occurrence_count=occ,
        )

    def add_audio_risk(self, base_score: float, confidence: float) -> float:
        """
        Direct score addition for audio proctoring events.
        Audio risk is always decaying.
        Returns the actual score added (0 if confidence too low).
        """
        if confidence < S.MIN_CONF:
            return 0.0
        added = base_score * confidence
        self._add(added, decaying=True)
        self._update_state()
        return added

    def occurrence_count(self, key: str) -> int:
        """Rising-edge detection count for a key (authoritative)."""
        return self._occurrences.get(key, 0)

    def get_display(self) -> dict:
        return {
            "score"    : round(self.score, 1),
            "fixed"    : round(self._fixed_score, 1),
            "decaying" : round(self._decaying_score, 1),
            "state"    : self.state.value,
            "terminated": self.terminated,
            "reason"   : self._termination_reason,
        }

    def get_summary(self) -> dict:
        return {
            "final_score"         : round(self.score, 1),
            "fixed_score"         : round(self._fixed_score, 1),
            "decaying_score"      : round(self._decaying_score, 1),
            "final_state"         : self.state.value,
            "occurrences"         : dict(self._occurrences),
            "terminated"          : self.terminated,
            "termination_reason"  : self._termination_reason,
            "decay_ticks"         : len(self.decay_log),
            "decay_log"           : self.decay_log,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add(self, amount: float, decaying: bool) -> float:
        """Add to the appropriate bucket. Returns amount added."""
        amount = max(0.0, amount)
        if decaying:
            self._decaying_score = min(S.DECAY_BUCKET_CAP, self._decaying_score + amount)
        else:
            self._fixed_score = min(S.DECAY_BUCKET_CAP, self._fixed_score + amount)
        return amount

    def _in_cooldown(self, key: str, now: float) -> bool:
        return now < self._score_cooldown_until.get(key, 0.0)

    def _arm_cooldown(self, key: str, now: float, override_cd: float = 0.0) -> None:
        cd = override_cd if override_cd > 0 else S.SCORE_COOLDOWNS.get(key, 3.0)
        self._score_cooldown_until[key] = now + cd

    def _score_event(self, key: str, confidence: float,
                     duration: float, occ: int, now: float) -> float:
        """Apply per-event scoring policy. Returns amount added to score."""
        min_conf = S.SCORE_MIN_CONF.get(key, S.MIN_CONF)
        if confidence < min_conf:
            return 0.0
        if self._in_cooldown(key, now):
            return 0.0

        decaying = key not in S.NON_DECAYING
        added    = 0.0

        # ── Phone ──────────────────────────────────────────────────────────────
        if key == "phone":
            grace = S.GRACE_OCCURRENCES.get("phone", 1)
            if occ <= grace:
                self._arm_cooldown(key, now)   # grace: arm cooldown, no score
            elif occ == grace + 1:
                added = self._add(S.PHONE_SCORE_2ND * confidence, decaying=False)
            else:
                added = self._add(S.PHONE_SCORE_3RD * confidence, decaying=False)

        # ── Headphone / earbud ────────────────────────────────────────────────
        elif key == "headphone":
            grace = S.GRACE_OCCURRENCES.get("headphone", 1)
            if occ <= grace:
                self._arm_cooldown(key, now)   # grace: arm cooldown, no score
            else:
                added = self._add(S.HEADPHONE_SCORE * confidence, decaying=True)
        elif key == "earbud":
            grace = S.GRACE_OCCURRENCES.get("earbud", 1)
            if occ <= grace:
                self._arm_cooldown(key, now)   # grace: arm cooldown, no score
            else:
                added = self._add(S.EARBUD_SCORE * confidence, decaying=True)

        # ── Multiple people ───────────────────────────────────────────────────
        elif key == "multiple_people":
            grace = S.GRACE_OCCURRENCES.get("multiple_people", 1)
            if occ <= grace:
                self._arm_cooldown(key, now)   # grace: arm cooldown, no score
            elif occ == grace + 1:
                added = self._add(S.MULTI_PEOPLE_SCORE_2ND * confidence, decaying=False)
            else:
                added = self._add(S.MULTI_PEOPLE_SCORE_3RD * confidence, decaying=False)

        # ── Book ──────────────────────────────────────────────────────────────
        elif key == "book":
            added = self._add(S.BOOK_SCORE * confidence, decaying=True)

        # ── Fake presence (duration-tiered) ───────────────────────────────────
        elif key == "fake_presence":
            if duration >= S.FAKE_PRESENCE_DUR_2:
                added = self._add(S.FAKE_PRESENCE_SCORE_2 * confidence, decaying=False)
            elif duration >= S.FAKE_PRESENCE_DUR_1:
                added = self._add(S.FAKE_PRESENCE_SCORE_1 * confidence, decaying=False)
            # < DUR_1: warning only

        # ── Face hidden (duration-tiered) ─────────────────────────────────────
        elif key == "face_hidden":
            if duration >= S.FACE_HIDDEN_DUR_2:
                added = self._add(S.FACE_HIDDEN_SCORE_2 * confidence, decaying=True)
            elif duration >= S.FACE_HIDDEN_DUR_1:
                added = self._add(S.FACE_HIDDEN_SCORE_1 * confidence, decaying=True)
            # < DUR_1: warning only

        # ── Gaze events ────────────────────────────────────────────────────────
        elif key in _GAZE_EVENTS:
            added = self._add(S.GAZE_SCORE * confidence, decaying=True)

        # ── Partial face ──────────────────────────────────────────────────────
        elif key == "partial_face":
            if duration >= S.PARTIAL_FACE_DURATION_GATE:
                added = self._add(S.PARTIAL_FACE_SCORE * confidence, decaying=True)

        # ── Exit fullscreen ───────────────────────────────────────────────────
        elif key == "exit_fullscreen":
            if occ == 1:
                pass  # warning only
            elif duration >= 2:
                added = self._add(5.0 * confidence, decaying=True)

        if added > 0:
            self._arm_cooldown(key, now)

        return added

    def _handle_special(self, key: str, active: bool, occ: int,
                         duration: float, confidence: float,
                         now: float) -> Optional[RiskEvent]:
        """
        Handle continuous-presence termination rules and no-person duration tiers.
        Returns a RiskEvent only if the exam was just terminated.
        Returns None to let normal scoring proceed.
        """
        if key == "multiple_people":
            self._multi_people_gone_since = None   # condition active — cancel grace timer
            if self._multi_people_since is None:
                self._multi_people_since = now
            elif now - self._multi_people_since >= S.MULTI_PEOPLE_TERMINATE_S:
                self._terminate(f"Multiple people continuously >{S.MULTI_PEOPLE_TERMINATE_S:.0f}s")
                return RiskEvent(key=key, terminated=True,
                                 termination_reason=self._termination_reason)

        elif key == "no_person":
            self._no_person_gone_since = None      # condition active — cancel grace timer
            no_person_added = 0.0
            if self._no_person_since is None:
                self._no_person_since = now
            else:
                dur = now - self._no_person_since
                if dur >= S.NO_PERSON_TERMINATE_S:
                    self._terminate(f"No person detected >{S.NO_PERSON_TERMINATE_S:.0f}s")
                    return RiskEvent(key=key, terminated=True,
                                     termination_reason=self._termination_reason)
                elif dur >= S.NO_PERSON_DUR_2 and not self._in_cooldown("_no_person_t2", now):
                    no_person_added = self._add(S.NO_PERSON_SCORE_2, decaying=False)
                    self._arm_cooldown("_no_person_t2", now, override_cd=15)
                elif dur >= S.NO_PERSON_DUR_1 and not self._in_cooldown("_no_person_t1", now):
                    no_person_added = self._add(S.NO_PERSON_SCORE_1, decaying=False)
                    self._arm_cooldown("_no_person_t1", now, override_cd=15)
            return RiskEvent(key=key, active=True, is_new_occurrence=(occ == 1 and duration < 1),
                             occurrence_count=occ, duration=duration,
                             risk_added=no_person_added)  # carries actual score added

        elif key == "speaker_audio":
            self._speaker_gone_since = None   # active — cancel flicker grace timer
            if self._speaker_since is None:
                self._speaker_since = now
            dur = now - self._speaker_since

            # Pick tier based on true episode count (_speaker_occ).
            # Episode 1 (_speaker_occ == 0): shorter grace, smaller first score (decaying).
            # Episode 2+ (_speaker_occ >= 1): longer grace, larger fixed scores.
            is_first  = (self._speaker_occ == 0)
            warn_s    = S.SPEAKER_OCC1_WARN_S   if is_first else S.SPEAKER_OCC2_WARN_S
            score_1   = S.SPEAKER_OCC1_SCORE    if is_first else S.SPEAKER_OCC2_SCORE
            repeat    = S.SPEAKER_OCC1_REPEAT   if is_first else S.SPEAKER_OCC2_REPEAT
            decaying1 = is_first   # first-episode threshold score is decaying; rest fixed

            speaker_added = 0.0
            if dur >= warn_s:
                if not self._in_cooldown("_speaker_t1", now):
                    # One-time score at end of grace window
                    speaker_added = self._add(score_1, decaying=decaying1)
                    self._arm_cooldown("_speaker_t1",    now, override_cd=999999)
                    self._arm_cooldown("_speaker_repeat", now, override_cd=S.SPEAKER_REPEAT_INTERVAL)
                elif not self._in_cooldown("_speaker_repeat", now):
                    # Repeating tail score every REPEAT_INTERVAL
                    speaker_added = self._add(repeat, decaying=False)
                    self._arm_cooldown("_speaker_repeat", now, override_cd=S.SPEAKER_REPEAT_INTERVAL)

            return RiskEvent(key=key, active=True, is_new_occurrence=(dur < 1),
                             occurrence_count=occ, duration=dur,
                             risk_added=speaker_added)

        return None  # proceed to normal _score_event

    def _check_combos(self, key: str, confidence: float, now: float) -> None:
        combos = [
            ({"looking_down", "book"},  S.COMBO_DOWN_BOOK,  "_combo_down_book"),
            ({"phone", "looking_down"}, S.COMBO_PHONE_DOWN, "_combo_phone_down"),
        ]
        for combo_set, bonus, cd_key in combos:
            if key in combo_set and combo_set.issubset(self._active_set):
                if now >= self._combo_cooldowns.get(cd_key, 0.0):
                    eff_conf = max(confidence, S.MIN_CONF)
                    self._add(bonus * eff_conf, decaying=True)
                    self._combo_cooldowns[cd_key] = now + 60.0

    def _apply_decay(self, now: float) -> None:
        """Decay only the decaying bucket. Records every tick in decay_log."""
        if now - self._last_decay_time >= self._decay_interval:
            before  = self._decaying_score
            self._decaying_score = max(0.0, self._decaying_score - S.DECAY_AMOUNT)
            after   = self._decaying_score
            elapsed = now - self._creation_time
            m, s    = divmod(int(elapsed), 60)
            self.decay_log.append({
                "time"             : f"{m:02d}:{s:02d}",
                "elapsed_s"        : round(elapsed, 1),
                "decayed_amount"   : round(before - after, 2),
                "decaying_before"  : round(before, 2),
                "decaying_after"   : round(after, 2),
                "total_score_after": round(self._fixed_score + after, 2),
            })
            self._last_decay_time = now

    def _update_state(self) -> None:
        if self.terminated:
            self.state = ExamState.TERMINATED
            return
        s = self.score
        if s >= S.STATE_ADMIN:
            self.state = ExamState.ADMIN_REVIEW
        elif s >= S.STATE_HIGH_RISK:
            self.state = ExamState.HIGH_RISK
        elif s >= S.STATE_WARNING:
            self.state = ExamState.WARNING
        else:
            self.state = ExamState.NORMAL

    def _terminate(self, reason: str) -> None:
        if not self.terminated:
            now = time.time()
            self._frozen_multi_dur     = (now - self._multi_people_since) if self._multi_people_since else 0.0
            self._frozen_no_person_dur = (now - self._no_person_since)    if self._no_person_since    else 0.0
            self.terminated          = True
            self._termination_reason = reason
            self.state               = ExamState.TERMINATED
