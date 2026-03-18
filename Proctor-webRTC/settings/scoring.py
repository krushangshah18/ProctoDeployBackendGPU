# ═══════════════════════════════════════════════════════════════════════════════
#  AI Proctor — Risk Scoring Settings
#
#  Every score value, cooldown, tier threshold, and scoring rule lives here.
#  Edit this file to tune scoring behaviour without touching engine logic.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Score bucket classification ───────────────────────────────────────────────
# Keys listed here go into the NON-DECAYING (fixed) bucket — their score
# contribution never shrinks over the session.  All other keys go into the
# decaying bucket.
NON_DECAYING: set = {"tab_switch", "phone", "fake_presence", "multiple_people", "no_person"}

# ── Confidence gate ────────────────────────────────────────────────────────────
# Global minimum confidence for scoring.  Per-key entries in SCORE_MIN_CONF
# take precedence.  Head/gaze events always pass confidence=1.0 so this only
# matters for YOLO-based events.
MIN_CONF: float = 0.5

SCORE_MIN_CONF: dict = {
    "phone"    : 0.60,    # matches YOLO phone detection threshold
    "book"     : 0.65,    # matches YOLO book  detection threshold
    "headphone": 0.40,    # matches YOLO headphone threshold
    "earbud"   : 0.40,    # matches YOLO earbud threshold
}

# ── Risk-scoring cooldowns (seconds) ──────────────────────────────────────────
# Minimum gap between consecutive score additions for the same event.
# RULE: api_cooldown == score_cooldown (every score event fires an alert).
#       warn_cooldown <= score_cooldown (warning shows at least as often as scoring).
# Both live in alerts.py and must be kept in sync with these values.
SCORE_COOLDOWNS: dict = {
    "looking_away"   :  5,
    "looking_down"   :  5,
    "looking_up"     :  5,
    "looking_side"   :  5,
    "partial_face"   :  5,
    "face_hidden"    :  5,
    "book"           : 30,
    "headphone"      : 30,
    "earbud"         : 30,
    "phone"          : 15,
    "multiple_people": 10,
    "no_person"      : 10,
    "fake_presence"  : 10,
    # speaker_audio not listed — managed by its own tier system below
}

# ── State machine thresholds ───────────────────────────────────────────────────
# Total score (fixed + decaying) required to enter each exam state.
STATE_WARNING   : float =  30.0   # NORMAL  → WARNING
STATE_HIGH_RISK : float =  60.0   # WARNING → HIGH_RISK
STATE_ADMIN     : float = 100.0   # HIGH_RISK → ADMIN_REVIEW
                                   # ADMIN_REVIEW → TERMINATED (via termination rules)

# ── Score decay ────────────────────────────────────────────────────────────────
# Applied to the decaying bucket on a timed interval.
# The interval is computed as max(60s, session_duration / 20) by the engine.
DECAY_AMOUNT    : float =   5.0   # points removed from decaying bucket per tick
DECAY_BUCKET_CAP: float = 150.0   # maximum value for either score bucket

# ── Grace occurrences ──────────────────────────────────────────────────────────
# Number of leading occurrences that are warning-only (no score added).
# occ <= grace → warn + arm cooldown, no score.
# occ >  grace → score added from that occurrence onwards.
GRACE_OCCURRENCES: dict = {
    "phone"          : 1,
    "headphone"      : 1,
    "earbud"         : 1,
    "multiple_people": 1,
}

# ── Phone (non-decaying) ───────────────────────────────────────────────────────
PHONE_SCORE_2ND : float = 25.0   # added on the (grace+1)th occurrence
PHONE_SCORE_3RD : float = 50.0   # added on (grace+2)th+ occurrence

# ── Book (decaying) ────────────────────────────────────────────────────────────
BOOK_SCORE      : float = 20.0   # added every SCORE_COOLDOWNS["book"] seconds

# ── Headphone / earbud (decaying) ─────────────────────────────────────────────
HEADPHONE_SCORE : float = 20.0
EARBUD_SCORE    : float = 20.0

# ── Multiple people (non-decaying) ────────────────────────────────────────────
MULTI_PEOPLE_SCORE_2ND : float = 20.0   # (grace+1)th occurrence
MULTI_PEOPLE_SCORE_3RD : float = 50.0   # (grace+2)th+ occurrence

# ── Gaze events: looking_away / down / up / side (decaying) ───────────────────
# Score is added once per SCORE_COOLDOWNS["looking_*"] seconds while active.
GAZE_SCORE      : float =  5.0

# ── Speaker audio — speech without lip movement ────────────────────────────────
# All speaker audio knobs live here (scoring + alert cooldowns in one place).
#
# Each TRUE episode (silence > flicker_grace_s = new episode) uses the tier
# matching its episode number.  Tier gates reset between episodes so each
# episode starts fresh; escalation is handled by the episode-number tier.
#
# Episode 1:
#   0 – OCC1_WARN_S        → warn only, every WARN_COOLDOWN seconds
#   at OCC1_WARN_S         → alert  +OCC1_SCORE  decaying  (one-time)
#   every REPEAT_INTERVAL  → alert  +OCC1_REPEAT fixed     (repeating)
#
# Episode 2+ :
#   0 – OCC2_WARN_S        → warn only, every WARN_COOLDOWN seconds
#   at OCC2_WARN_S         → alert  +OCC2_SCORE  fixed     (one-time)
#   every REPEAT_INTERVAL  → alert  +OCC2_REPEAT fixed     (repeating)

SPEAKER_WARN_COOLDOWN   : float =  3.0   # warn repeat cadence during grace window
SPEAKER_ALERT_COOLDOWN  : float = 10.0   # min gap between alert banners (= REPEAT_INTERVAL)

# Episode 1 tier
SPEAKER_OCC1_WARN_S     : float =  3.0   # warn-only grace (seconds)
SPEAKER_OCC1_SCORE      : float = 10.0   # score at end of grace (decaying)
SPEAKER_OCC1_REPEAT     : float = 20.0   # repeating score every REPEAT_INTERVAL (fixed)

# Episode 2+ tier
SPEAKER_OCC2_WARN_S     : float =  5.0   # warn-only grace (seconds)
SPEAKER_OCC2_SCORE      : float = 20.0   # score at end of grace (fixed)
SPEAKER_OCC2_REPEAT     : float = 20.0   # repeating score every REPEAT_INTERVAL (fixed)

SPEAKER_REPEAT_INTERVAL : float = 10.0   # cadence for all repeating tail scores

# ── Partial face — too far from camera (decaying) ─────────────────────────────
PARTIAL_FACE_SCORE         : float = 2.0   # score per scoring cooldown tick
PARTIAL_FACE_DURATION_GATE : float = 5.0   # must be active this many seconds first (= score cooldown)

# ── Face hidden — person present but no landmarks (decaying, duration-tiered) ──
FACE_HIDDEN_DUR_1   : float =  5.0   # seconds before tier-1 score
FACE_HIDDEN_DUR_2   : float = 10.0   # seconds before tier-2 score
FACE_HIDDEN_SCORE_1 : float = 10.0   # score added at tier 1
FACE_HIDDEN_SCORE_2 : float = 20.0   # score added at tier 2

# ── Fake presence — no movement / no blink (non-decaying, duration-tiered) ────
FAKE_PRESENCE_DUR_1   : float = 10.0   # seconds before tier-1 score
FAKE_PRESENCE_DUR_2   : float = 25.0   # seconds before tier-2 score
FAKE_PRESENCE_SCORE_1 : float = 30.0
FAKE_PRESENCE_SCORE_2 : float = 60.0

# ── No person — nobody detected (non-decaying, duration-tiered) ───────────────
NO_PERSON_DUR_1   : float =  5.0
NO_PERSON_DUR_2   : float = 10.0
NO_PERSON_SCORE_1 : float = 25.0
NO_PERSON_SCORE_2 : float = 50.0

# ── Termination thresholds ─────────────────────────────────────────────────────
# Exam is auto-terminated when a condition persists continuously this long.
MULTI_PEOPLE_TERMINATE_S : float = 20.0
NO_PERSON_TERMINATE_S    : float = 20.0

# ── Gaze aggregation bonus (decaying) ─────────────────────────────────────────
# When multiple gaze events cluster within the window, an extra bonus is added.
GAZE_WINDOW_S         : float = 30.0   # look-back window in seconds
GAZE_BONUS_MIN_EVENTS : int   = 3      # events within window to trigger bonus
GAZE_BONUS_SCORE      : float = 10.0   # bonus score (decaying)

# ── Combo bonuses (decaying) ──────────────────────────────────────────────────
# Extra score when two suspicious conditions occur simultaneously.
# Each combo has a 60-second internal cooldown to prevent repeated stacking.
COMBO_DOWN_BOOK  : float = 15.0   # looking_down + book
COMBO_PHONE_DOWN : float = 20.0   # phone + looking_down

# ── Tab switch — exam page left / window hidden (non-decaying) ─────────────────
# Termination rule: >= TAB_SWITCH_TERMINATE_COUNT switches → exam auto-terminated.
# Occurrence 1            → warn only (grace)
# Occurrence 2            → +TAB_SWITCH_SCORE pts fixed  (alert)
# Occurrence >= TERMINATE → TERMINATED immediately
TAB_SWITCH_SCORE           : float = 15.0   # score added from 2nd occurrence onward
TAB_SWITCH_TERMINATE_COUNT : int   = 3      # number of switches that trigger termination
