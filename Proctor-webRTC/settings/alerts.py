# ═══════════════════════════════════════════════════════════════════════════════
#  AI Proctor — Alert & Warning Settings
#
#  RULES (must be respected when editing):
#    api_cooldown  == score_cooldown   every score event directly fires an alert
#    warn_cooldown <= score_cooldown   warning shows at least as often as scoring
#
#  Grace period is enforced in the RiskEngine (occ=1 arms cooldown, no score).
#  AlertEngine rule: risk_added == 0 → WARNING,  risk_added > 0 → ALERT.
# ═══════════════════════════════════════════════════════════════════════════════

# ── On-screen display durations (seconds) ──────────────────────────────────────
WARN_DISPLAY_DURATION  : float = 3.0   # how long a warning banner stays visible
ALERT_DISPLAY_DURATION : float = 5.0   # how long an alert banner stays visible

# ── Warning cooldowns (seconds) ────────────────────────────────────────────────
# Minimum gap between consecutive soft (amber) warnings for the same key.
# Must be <= the corresponding SCORE_COOLDOWNS value.
WARN_COOLDOWNS: dict = {
    "looking_away"   :  3,    # score_cooldown =  5s
    "looking_down"   :  3,
    "looking_up"     :  3,
    "looking_side"   :  3,
    "speaker_audio"  :  3,    # ← edit in scoring.py (SPEAKER_WARN_COOLDOWN)
    "partial_face"   :  3,
    "face_hidden"    :  3,
    "fake_presence"  :  5,    # score_cooldown = 10s
    "phone"          :  8,    # score_cooldown = 15s
    "multiple_people":  5,    # score_cooldown = 10s
    "no_person"      :  5,
    "book"           : 15,    # score_cooldown = 30s
    "headphone"      : 15,
    "earbud"         : 15,
    "tab_switch"     :  0,    # discrete event — cooldown managed by engine occurrence count
}

# ── API alert cooldowns (seconds) ─────────────────────────────────────────────
# MUST equal SCORE_COOLDOWNS — every scoring event fires an alert.
# Edit both together when changing timing for a key.
API_COOLDOWNS: dict = {
    "looking_away"   :  5,
    "looking_down"   :  5,
    "looking_up"     :  5,
    "looking_side"   :  5,
    "speaker_audio"  : 10,    # ← edit in scoring.py (SPEAKER_ALERT_COOLDOWN)
    "partial_face"   :  5,
    "face_hidden"    :  5,
    "fake_presence"  : 10,
    "phone"          : 15,
    "multiple_people": 10,
    "no_person"      : 10,
    "book"           : 30,
    "headphone"      : 30,
    "earbud"         : 30,
    "tab_switch"     :  0,    # discrete — no cooldown needed (per-occurrence)
}

# ── On-screen messages ─────────────────────────────────────────────────────────
# WARN_MESSAGES — shown on amber banner (no score added yet)
# ALERT_MESSAGES — shown on red banner (score was added, pts appended automatically)
WARN_MESSAGES: dict = {
    "phone"          : "Phone visible in frame",
    "book"           : "Book visible in frame",
    "headphone"      : "Headphones visible",
    "earbud"         : "Earbuds visible",
    "looking_away"   : "Not facing screen",
    "looking_down"   : "Looking down",
    "looking_up"     : "Looking up",
    "looking_side"   : "Gaze shifted sideways",
    "face_hidden"    : "Face not clearly visible",
    "partial_face"   : "Face partially outside frame",
    "fake_presence"  : "No movement detected — please move slightly",
    "multiple_people": "Multiple people in frame",
    "no_person"      : "Candidate not visible",
    "speaker_audio"  : "Speech detected without lip movement",
    "tab_switch"     : "Please return to the exam — do not leave this tab",
}

ALERT_MESSAGES: dict = {
    "phone"          : "Mobile phone detected",
    "book"           : "Book detected",
    "headphone"      : "Headphones detected",
    "earbud"         : "Earbuds detected",
    "looking_away"   : "Candidate not facing screen",
    "looking_down"   : "Candidate looking down",
    "looking_up"     : "Candidate looking up",
    "looking_side"   : "Candidate gaze away from screen",
    "face_hidden"    : "Face obstructed or hidden",
    "partial_face"   : "Candidate too far from camera",
    "fake_presence"  : "Possible fake presence (static image)",
    "multiple_people": "Multiple people detected",
    "no_person"      : "No person detected — candidate absent",
    "speaker_audio"  : "Speaker audio detected",
    "tab_switch"     : "Tab switch detected — candidate left the exam tab",
}
