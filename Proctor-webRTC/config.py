import os

# ── YOLO Model ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "finalBestV5.pt")

# ── Coordinator ───────────────────────────────────────────────────────────────
# TICK_RATE: lower = less CPU per session, higher = more responsive.
# 10 Hz @ 3 users/container: benchmarked tick avg=64ms p99=89ms ok%=100% (100ms budget).
# Head-pose gates are 1.5–2 s → 10 Hz gives 15+ samples per gate — sufficient.
# stride=3 + 3 users = exactly 1 MP call per tick (perfect rotation).
TICK_RATE    = 10   # Hz  — 3 users/container config; revert to 14 if dropping to 2u
MAX_SESSIONS = 3    # g4dn.xlarge: benchmarked safe limit (4 users fails at 41.4% tick OK)

# ── Detection Toggles ────────────────────────────────────────────────────────
# Set False to completely skip a detection — no alert, no processing, no score.
# Per-exam overrides can be sent in the /offer request body under "detection_config".

# Head / gaze
DETECT_LOOKING_AWAY    = True
DETECT_LOOKING_DOWN    = True
DETECT_LOOKING_UP      = True
DETECT_LOOKING_SIDE    = True    # iris-based gaze — disabling skips iris landmark refinement
DETECT_FACE_HIDDEN     = True
DETECT_PARTIAL_FACE    = True
DETECT_FAKE_PRESENCE   = True
DETECT_SPEAKER_AUDIO   = True

# Objects
DETECT_PHONE           = True
DETECT_BOOK            = True
DETECT_HEADPHONE       = True
DETECT_EARBUD          = True
DETECT_MULTIPLE_PEOPLE = True

# ── Alert Cooldowns ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS       = 3
RESET_COOLDOWN_SECONDS = 1

# ── Duration Thresholds (seconds a condition must persist before alerting) ───
LOOKING_AWAY_THRESHOLD = 2.0
GAZE_THRESHOLD         = 1.5

# ── Head Pose ────────────────────────────────────────────────────────────────
LOOK_AWAY_YAW   = 0.20
LOOK_DOWN_PITCH = 0.13
LOOK_UP_PITCH   = -0.10
GAZE_LEFT       = -0.13
GAZE_RIGHT      =  0.13

# ── Partial Face ─────────────────────────────────────────────────────────────
MIN_FACE_WIDTH  = 110   # pixels
MIN_FACE_HEIGHT = 120   # pixels

# ── Face Hidden ───────────────────────────────────────────────────────────────
# face_hidden fires only within this many seconds after the face was last seen.
# Prevents false positives when YOLO sees a person but MediaPipe finds no face
# on a dark/covered camera that has never had a face at all.
FACE_HIDDEN_RECENCY_S = 4.0

# ── Blink / EAR ──────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.20
# Time-based blink window: EAR must stay below threshold for at least
# BLINK_MIN_DURATION_S to count as a blink, and no longer than
# BLINK_MAX_DURATION_S (longer = eye closed / drowsiness, not a blink).
# At 10Hz / stride=3 each session gets MediaPipe at ~3.3Hz (~300ms per sample),
# so frame-count gates were unreliable for quick blinks (150-400ms).
BLINK_MIN_DURATION_S = 0.05   # 50ms  — catches quick single-frame blinks
BLINK_MAX_DURATION_S = 0.40   # 400ms — above this = eye closed, not a blink

# ── Liveness ─────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL  = 0.2
FAKE_WINDOW      = 15.0
MIN_VARIANCE     = 0.001
NO_BLINK_TIMEOUT = 10
LIVENESS_WEIGHTS = {"yaw": 0.45, "gaze": 0.45, "pitch": 0.10}

# ── Lip Movement / Speaker Audio ─────────────────────────────────────────────
LIP_MAR_SPEAKING    = 0.05
LIP_MAR_YAWN        = 0.22
LIP_YAWN_DURATION_S = 1.5
LIP_DYNAMIC_STD_MIN = 0.010
LIP_HISTORY         = 30

AUDIO_SAMPLE_RATE   = 16_000
AUDIO_CHANNELS      = 1
AUDIO_CHUNK_SAMPLES = 512
AUDIO_SPEECH_THRESH = 0.5
SPEAKER_HOLD_S      = 0.3

# ── Object Detection (YOLO confidence thresholds) ────────────────────────────
YOLO_DEFAULT_CONF = 0.50
YOLO_PERSON_CONF  = 0.30
YOLO_PHONE_CONF   = 0.65
YOLO_BOOK_CONF    = 0.70
YOLO_AUDIO_CONF   = 0.41

# ── Object Detection (temporal stability) ────────────────────────────────────
OBJECT_WINDOW    = 15
OBJECT_MIN_VOTES = 5
PHONE_MIN_VOTES     = 9
BOOK_MIN_VOTES      = 10
HEADPHONE_MIN_VOTES = 9
EARBUD_MIN_VOTES    = 9

# ── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_SESSION_DURATION_S  = 300   # 5 minutes max exam duration
TIMER_FLICKER_GRACE_S    = 1.5

# ── Session Report ────────────────────────────────────────────────────────────
SAVE_REPORT = True
REPORT_DIR  = "reports"

# ── Proof capture ─────────────────────────────────────────────────────────────
# SAVE_PROOF = True  → single JPEG per alert, plus a WAV clip for speaker_audio.
# SAVE_PROOF = False → no proof files written at all.
SAVE_PROOF        = True
PROOF_AUDIO_PRE_S = 5.0   # seconds of audio ring-buffer to capture before the alert

# ── Inference device ──────────────────────────────────────────────────────────
# GPU-only deployment — CUDA is required.
YOLO_DEVICE = "cuda"

# ── GPU performance ───────────────────────────────────────────────────────────
# YOLO_HALF: FP16 inference — ~2× throughput, ~half VRAM, negligible accuracy loss.
#            Override at launch: python main.py --half
YOLO_HALF          = True    # FP16: ~2× throughput, ~half VRAM

# YOLO_WARMUP_FRAMES: dummy forward passes at startup to pre-compile CUDA kernels.
#            Override at launch: python main.py --warmup 3
YOLO_WARMUP_FRAMES = 3    # pre-compile CUDA kernels at startup

# YOLO_MIN_VRAM_GB: minimum free GPU VRAM required to use CUDA in "auto" mode.
# If free VRAM is below this threshold the detector falls back to CPU automatically.
YOLO_MIN_VRAM_GB   = 2.0    # RTX PRO 4500 has 32 GB; set higher threshold for auto-selection

# YOLO_IMGSZ: input resolution for YOLO inference.
# Set at MODEL LOAD TIME via model.overrides — never at inference time (breaks
# batch coord transforms).  320 → ~3× faster than 640, same bbox pixel coords.
YOLO_IMGSZ        = 640     # keep 640 — needed for earbud/small-object detection

# MEDIAPIPE_STRIDE: run FaceMesh every N ticks per session, reusing last result
# on skipped ticks.  Duration gates are 1.5–2 s so at 14 Hz / stride 3 = 4.7 Hz
# per session we get 7+ samples per gate — sufficient for accurate detection.
# stride=3 keeps every 3rd tick as a "free" tick (YOLO only, no MP) — this
# acts as a pressure-relief valve that prevents p99 spikes under GPU contention.
# Tested stride=2 @ 14Hz, 2 users: p99 spiked 44ms→242ms, tick OK% dropped
# from 100% to 98.3% — the free ticks are load-bearing, not wasted.
MEDIAPIPE_STRIDE  = 3

