"""
ProctorSession — per-candidate state container.

No drawing, no video recording. Output is data only:
  - JSON alert/warning logs
  - Lightweight proof files (JPEG per alert, WAV for audio events)
  - SSE event stream (real-time push to connected frontend clients)
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from detectors import FaceMeshProvider, HeadPoseDetector, LipDetector, merge_by_class
from core.alert_engine import AlertEngine
from core.head_tracker import HeadTracker
from core.liveness import LivenessDetector
from core.risk_engine import RiskEngine, ExamState
from core.audio_monitor import AudioMonitor, SpeakerAudioDetector
from core.object_tracker import ObjectTemporalTracker
from utils import AlertManager, ProofWriter

_DEFAULTS = {
    "DETECT_LOOKING_AWAY"    : True,
    "DETECT_LOOKING_DOWN"    : True,
    "DETECT_LOOKING_UP"      : True,
    "DETECT_LOOKING_SIDE"    : True,
    "DETECT_FACE_HIDDEN"     : True,
    "DETECT_PARTIAL_FACE"    : True,
    "DETECT_FAKE_PRESENCE"   : True,
    "DETECT_SPEAKER_AUDIO"   : True,
    "DETECT_PHONE"           : True,
    "DETECT_BOOK"            : True,
    "DETECT_HEADPHONE"       : True,
    "DETECT_EARBUD"          : True,
    "DETECT_MULTIPLE_PEOPLE" : True,
    "LOOKING_AWAY_THRESHOLD" : 2.0,
    "GAZE_THRESHOLD"         : 1.5,
    "SAMPLE_INTERVAL"        : 0.2,
    "FAKE_WINDOW"            : 15.0,
    "MIN_VARIANCE"           : 0.001,
    "NO_BLINK_TIMEOUT"       : 10,
    "LIVENESS_WEIGHTS"       : {"yaw": 0.45, "gaze": 0.45, "pitch": 0.10},
    "AUDIO_SAMPLE_RATE"      : 16_000,
    "AUDIO_CHANNELS"         : 1,
    "AUDIO_CHUNK_SAMPLES"    : 512,
    "AUDIO_SPEECH_THRESH"    : 0.5,
    "SPEAKER_HOLD_S"         : 0.3,
    "OBJECT_WINDOW"          : 15,
    "OBJECT_WINDOW_S"        : 1.5,
    "VOTE_RATIOS"            : {
        "book":       0.65,
        "phone":      0.55,
        "headphones": 0.55,
        "earbud":     0.50,
        "default":    0.55,
    },
    "RISK_SESSION_DURATION_S" : 3600,
    "TIMER_FLICKER_GRACE_S"   : 1.5,
    "SAVE_PROOF"              : True,
    "SAVE_REPORT"             : True,
    "PROOF_AUDIO_PRE_S"       : 5.0,
}


class ProctorSession:

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        config: dict | None = None,
        use_webrtc_audio: bool = True,
    ):
        self.session_id  = session_id
        self.session_dir = Path(session_dir)
        self.proof_dir   = self.session_dir / "proof"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        cfg = {**_DEFAULTS, **(config or {})}
        self._cfg = cfg

        # Shared exam-level config injected by ProctorCoordinator.add_session().
        # Admin can update this dict at runtime — changes apply to all sessions
        # immediately because all sessions share the same dict reference.
        self._exam_config: dict = {}

        self.session_start = time.time()
        self.session_clock = time.monotonic()

        # ── Latest frame slot ──────────────────────────────────────────────────
        self.latest_frame: np.ndarray | None = None
        self.observed_fps: float             = 15.0

        # ── Logs ──────────────────────────────────────────────────────────────
        self.alert_log:   list[dict] = []
        self.warning_log: list[dict] = []

        # ── SSE subscribers ───────────────────────────────────────────────────
        # Each connected SSE client gets a Queue. Alert/warning events are
        # put_nowait() onto every subscriber queue so the SSE endpoint can yield them.
        self._sse_queues: list[asyncio.Queue] = []

        # ── Head-tracker state dict ───────────────────────────────────────────
        self._states: dict = {
            "phone"          : {"active": False, "last_alert": 0},
            "multiple_people": {"active": False, "last_alert": 0},
            "no_person"      : {"active": False, "last_alert": 0},
            "book"           : {"active": False, "last_alert": 0},
            "headphone"      : {"active": False, "last_alert": 0},
            "earbud"         : {"active": False, "last_alert": 0},
            "speaker_audio"  : {"active": False, "last_alert": 0},
            "looking_away"   : {"active": False, "last_alert": 0, "start_time": None},
            "looking_down"   : {"active": False, "last_alert": 0, "start_time": None},
            "looking_up"     : {"active": False, "last_alert": 0, "start_time": None},
            "looking_side"   : {"active": False, "last_alert": 0, "start_time": None},
            "face_hidden"    : {"active": False, "last_alert": 0, "start_time": None},
            "partial_face"   : {"active": False, "last_alert": 0, "start_time": None},
            "fake_presence"  : {"active": False, "last_alert": 0, "start_time": None},
        }

        self._alert_manager = AlertManager()
        self._alert_manager.on_alert = self._on_api_alert
        self._alert_manager.on_warn  = self._on_warn_notice

        # ── Detectors ─────────────────────────────────────────────────────────
        refine_landmarks = cfg.get("DETECT_LOOKING_SIDE", True)
        self._face_mesh    = FaceMeshProvider(refine_landmarks=refine_landmarks)
        self.head_detector = HeadPoseDetector(debug=False, own_mesh=False)
        self.lip_detector  = LipDetector(own_mesh=False)

        # ── State machines ────────────────────────────────────────────────────
        window_frames = float(cfg["OBJECT_WINDOW"])
        window_s      = window_frames / 15.0
        vote_ratios   = {
            "phone"    : float(cfg["PHONE_MIN_VOTES"])  / window_frames,
            "book"     : float(cfg["BOOK_MIN_VOTES"])   / window_frames,
            "headphone": float(cfg["EARBUD_MIN_VOTES"]) / window_frames,
            "earbud"   : float(cfg["EARBUD_MIN_VOTES"]) / window_frames,
            "default"  : float(cfg["OBJECT_MIN_VOTES"]) / window_frames,
        }

        self.obj_tracker  = ObjectTemporalTracker(window_s=window_s, vote_ratios=vote_ratios)
        self.alert_engine = AlertEngine()
        self.head_tracker = HeadTracker(self._states, cfg["LOOKING_AWAY_THRESHOLD"], debug=False)
        self.liveness     = LivenessDetector(
            cfg["FAKE_WINDOW"], cfg["SAMPLE_INTERVAL"],
            cfg["MIN_VARIANCE"], cfg["NO_BLINK_TIMEOUT"],
            cfg["LIVENESS_WEIGHTS"],
        )
        self.speaker_audio = SpeakerAudioDetector(hold_s=cfg["SPEAKER_HOLD_S"])
        self.risk = RiskEngine(
            session_duration_s = cfg["RISK_SESSION_DURATION_S"],
            flicker_grace_s    = cfg["TIMER_FLICKER_GRACE_S"],
        )

        self.audio_monitor = AudioMonitor(
            sample_rate      = cfg["AUDIO_SAMPLE_RATE"],
            channels         = cfg["AUDIO_CHANNELS"],
            chunk_samples    = cfg["AUDIO_CHUNK_SAMPLES"],
            speech_threshold = cfg["AUDIO_SPEECH_THRESH"],
        )
        if use_webrtc_audio:
            self.audio_monitor.start_webrtc_mode()
        else:
            self.audio_monitor.start()

        self.proof_writer = ProofWriter(
            str(self.proof_dir),
            audio_pre_s = cfg["PROOF_AUDIO_PRE_S"],
        ) if cfg["SAVE_PROOF"] else None

        self._termination_proved   = False
        self._termination_sse_sent = False   # push session-terminated SSE only once

        # Debug overlay — toggled per-session by admin
        self.debug_mode: bool           = False
        self._last_mp_result: tuple | None = None
        self._last_detections: list      = []

        # Tracks the last wall-clock time face landmarks were actively detected.
        # Used to gate face_hidden: only fires within FACE_HIDDEN_RECENCY_S after
        # a real face was confirmed, preventing YOLO false-positives on a
        # dark/covered camera from triggering face_hidden instead of no_person.
        self._last_face_seen: float = 0.0
        # How long after losing landmarks face_hidden is still eligible to fire.
        # 4 seconds gives enough time for genuine face-turn/obstruction events
        # while excluding camera-cover scenarios where face was never recently seen.
        self._face_hidden_recency_s: float = float(cfg.get("FACE_HIDDEN_RECENCY_S", 4.0))

        self._last_tick_mono: float = 0.0
        self._tick_fps:       float = 15.0

    # ── Detection toggle helper ───────────────────────────────────────────────

    def _detect(self, key: str) -> bool:
        """
        Check whether a detection is currently enabled.
        exam_config (set by admin at runtime) overrides the session's static config.
        """
        if key in self._exam_config:
            return bool(self._exam_config[key])
        return bool(self._cfg.get(key, True))

    # ── SSE subscription ──────────────────────────────────────────────────────

    def subscribe_sse(self, queue: asyncio.Queue) -> None:
        self._sse_queues.append(queue)

    def unsubscribe_sse(self, queue: asyncio.Queue) -> None:
        try:
            self._sse_queues.remove(queue)
        except ValueError:
            pass

    def _push_sse(self, event: dict) -> None:
        for q in self._sse_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass   # slow subscriber — drop event

    # ── Tab switch (discrete event from frontend) ─────────────────────────────

    def report_tab_switch(self, now: float = 0.0) -> None:
        """
        Called when the frontend reports that the candidate left the exam tab.
        Scoring rules (set in settings/scoring.py):
          occ == 1                          → warning only
          occ == 2 to TERMINATE-1           → alert +TAB_SWITCH_SCORE pts fixed
          occ >= TAB_SWITCH_TERMINATE_COUNT → exam terminated
        """
        if now <= 0.0:
            now = time.time()
        rev   = self.risk.handle_tab_switch(now=now)
        frame = self.latest_frame
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._handle_event(rev, frame, now)

    # ── Debug overlay ─────────────────────────────────────────────────────────

    def set_debug(self, enabled: bool) -> None:
        self.debug_mode = enabled

    def get_debug_frame(self) -> "np.ndarray | None":
        """
        Return a BGR frame with full debug overlay:
          • YOLO bounding boxes (correct coords — no manual pre-resize)
          • MediaPipe landmarks, head-pose lines, gaze dots, EAR value
          • Lip MAR contour + speaking indicator
          • Active alert / warning banners (left side)
          • Risk score panel (top-right)
          • Audio MIC indicator (bottom-right)
        Returns None if no frame has been received yet.
        """
        import cv2
        from utils.draw import draw_detections, draw_alerts, draw_audio_status

        frame = self.latest_frame
        if frame is None:
            return None
        canvas = frame.copy()

        # ── 1. YOLO bounding boxes ─────────────────────────────────────────
        if self._last_detections:
            draw_detections(canvas, self._last_detections)

        # ── 2. MediaPipe — landmarks, head pose, lip MAR ───────────────────
        if self._last_mp_result is not None:
            head_result, lip_result = self._last_mp_result
            self.head_detector.draw_debug(canvas, head_result)
            self.lip_detector._draw_overlay(canvas, lip_result)

        # ── 3. Alert / warning banners (left side) ─────────────────────────
        draw_alerts(
            canvas,
            self._alert_manager.get_active_warnings(),
            self._alert_manager.get_active_alerts(),
        )

        # ── 4. Audio MIC indicator (bottom-right) ──────────────────────────
        speech_active = (
            self.audio_monitor.speech_active()
            if self.audio_monitor is not None else False
        )
        draw_audio_status(canvas, speech_active)

        # ── 5. Risk score panel (top-right) ────────────────────────────────
        self._draw_risk_panel(canvas)

        return canvas

    _STATE_COLORS = {
        "NORMAL"      : (0,   200,  0),
        "WARNING"     : (0,   200, 255),
        "HIGH_RISK"   : (0,   100, 255),
        "ADMIN_REVIEW": (0,     0, 255),
        "TERMINATED"  : (0,     0, 180),
    }

    def _draw_risk_panel(self, frame: "np.ndarray") -> None:
        """Semi-transparent risk score panel — top-right corner."""
        import cv2
        h, w  = frame.shape[:2]
        info  = self.risk.get_display()
        score = info["score"]
        state = info["state"]
        color = self._STATE_COLORS.get(state, (200, 200, 200))

        panel_w, panel_h = 230, 90
        x1 = w - panel_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + panel_h

        # Semi-transparent dark background
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            bg = roi.copy()
            bg[:] = (20, 20, 20)
            cv2.addWeighted(bg, 0.60, roi, 0.40, 0, roi)
            frame[y1:y2, x1:x2] = roi

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Score line
        cv2.putText(
            frame,
            f"Risk: {score:.0f}  F:{info['fixed']:.0f}  D:{info['decaying']:.0f}",
            (x1 + 6, y1 + 20), font, 0.45, color, 1, cv2.LINE_AA,
        )
        # State label
        cv2.putText(
            frame, state,
            (x1 + 6, y1 + 42), font, 0.55, color, 2, cv2.LINE_AA,
        )
        # Progress bar
        bx1, bx2, by = x1 + 6, x2 - 6, y1 + 62
        bar_w = bx2 - bx1
        filled = int(bar_w * min(score, 100) / 100)
        cv2.rectangle(frame, (bx1, by - 6), (bx2, by + 4), (55, 55, 55), -1)
        if filled > 0:
            cv2.rectangle(frame, (bx1, by - 6), (bx1 + filled, by + 4), color, -1)
        # Termination banner overlay
        if info.get("terminated"):
            cv2.putText(
                frame, "TERMINATED",
                (x1 + 6, y1 + 82), font, 0.48, (0, 0, 220), 2, cv2.LINE_AA,
            )

    # ── Audio ─────────────────────────────────────────────────────────────────

    def push_audio_chunk(self, pcm_bytes: bytes, timestamp: float) -> None:
        if self.audio_monitor is not None:
            with self.audio_monitor._lock:
                self.audio_monitor._audio_ring.append((timestamp, pcm_bytes))

    # ── MediaPipe gating ──────────────────────────────────────────────────────

    # Detection flags that require FaceMesh / HeadPoseDetector / LipDetector.
    # If NONE of these are enabled, all three MediaPipe calls are skipped entirely.
    _MEDIAPIPE_FLAGS = frozenset({
        "DETECT_LOOKING_AWAY",
        "DETECT_LOOKING_DOWN",
        "DETECT_LOOKING_UP",
        "DETECT_LOOKING_SIDE",
        "DETECT_PARTIAL_FACE",
        "DETECT_FAKE_PRESENCE",
        "DETECT_FACE_HIDDEN",
        "DETECT_SPEAKER_AUDIO",   # needs lip_result.is_speaking
    })

    # Null results returned when MediaPipe is skipped — same shape as real outputs.
    _NULL_HEAD = (False, False, False, False, False, False, 0.0, 0.0, 0.0, 0, False, 0)

    def needs_mediapipe(self) -> bool:
        """Return True if at least one enabled detection requires MediaPipe."""
        return any(self._detect(k) for k in self._MEDIAPIPE_FLAGS)

    # ── MediaPipe (thread pool) ───────────────────────────────────────────────

    def run_mediapipe(self, frame: np.ndarray) -> tuple:
        """
        Run FaceMesh → HeadPoseDetector → LipDetector on *frame*.
        Skipped entirely (null result) if no enabled detection needs it —
        saves ~15–40 ms per tick and one thread-pool slot per session.
        """
        from detectors.lip_detector import LipState

        if not self.needs_mediapipe():
            return (self._NULL_HEAD, LipState(face_detected=False))

        ts          = time.monotonic() - self.session_clock
        landmarks   = self._face_mesh.process(frame)
        head_result = self.head_detector.detect(frame, draw=False, landmarks=landmarks)
        lip_result  = self.lip_detector.process(frame, ts, draw=False, landmarks=landmarks)
        return head_result, lip_result

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[dict],
        mp_result:  tuple,
        frame:      np.ndarray,
        now:        float,
        fps:        float,
    ) -> None:
        # Hard guard: once the exam is terminated, no more scoring or alerts.
        # The coordinator snapshot filter also excludes terminated sessions, but a
        # batch that was already in-flight (YOLO/MediaPipe running async) can still
        # reach this call.  This is the final, cheapest gate.
        if self.risk.terminated:
            return

        self.observed_fps = fps

        now_mono = time.monotonic()
        if self._last_tick_mono > 0:
            dt = now_mono - self._last_tick_mono
            if 0 < dt < 1.0:
                self._tick_fps = 0.9 * self._tick_fps + 0.1 * (1.0 / dt)
        self._last_tick_mono = now_mono

        # Frame quality guard
        _sample = frame[::4, ::4]
        if _sample.mean() < 5 or _sample.std() < 8:
            return

        # Store for on-demand debug snapshot
        self._last_mp_result  = mp_result
        self._last_detections = detections

        head_result, lip_result = mp_result
        (
            looking_away, looking_down, looking_up,
            looking_left, looking_right,
            partial_face,
            yaw, pitch, gaze,
            _, blinked, _,
        ) = head_result

        self.liveness.update(yaw, pitch, gaze, blinked)
        fake = self.liveness.is_fake()

        speech_active = (
            self.audio_monitor.speech_active()
            if self.audio_monitor and self._detect("DETECT_SPEAKER_AUDIO")
            else False
        )
        speaker_flagged = (
            self.speaker_audio.update(
                speech_active = speech_active,
                lip_speaking  = lip_result.is_speaking,
                face_detected = lip_result.face_detected,
                timestamp     = time.monotonic() - self.session_clock,
            )
            if self._detect("DETECT_SPEAKER_AUDIO") else False
        )

        phone = book = headphone = earbud = False
        phone_conf = book_conf = hp_conf = eb_conf = 1.0
        people_count = 0
        for d in detections:
            cls  = d["class"]
            conf = d.get("confidence", 1.0)
            if   cls == "person"    : people_count += 1
            elif cls == "cell_phone": phone     = True; phone_conf = conf
            elif cls == "book"      : book      = True; book_conf  = conf
            elif cls == "headphone" : headphone = True; hp_conf    = conf
            elif cls == "earbud"    : earbud    = True; eb_conf    = conf

        landmarks_active = bool(yaw or pitch or gaze)

        # Update last-seen timestamp whenever face landmarks are actively detected.
        if landmarks_active:
            self._last_face_seen = now

        # no_person: YOLO sees nobody AND no face landmarks
        no_person_cond = people_count == 0 and not landmarks_active

        # face_hidden: person IS detected by YOLO but landmarks are absent.
        # The recency gate prevents YOLO false-positives on a dark/covered camera
        # (where people_count briefly flickers to 1) from firing face_hidden —
        # if no face was seen recently it is a no_person situation, not face_hidden.
        face_hidden_cond = (
            people_count > 0
            and not landmarks_active
            and (now - self._last_face_seen) < self._face_hidden_recency_s
        )

        # fake_presence: person detected AND face landmarks visible AND no movement/blink.
        # Requires landmarks_active so the liveness detector's variance inputs are
        # real face-tracking values, not all-zeros from a blank/covered frame.
        fake_presence_cond = (
            people_count > 0
            and landmarks_active
            and fake
        )

        head_conditions: dict[str, tuple[bool, bool]] = {
            "looking_away" : (looking_away,                  self._detect("DETECT_LOOKING_AWAY")),
            "looking_down" : (looking_down,                  self._detect("DETECT_LOOKING_DOWN")),
            "looking_up"   : (looking_up,                    self._detect("DETECT_LOOKING_UP")),
            "looking_side" : (looking_left or looking_right, self._detect("DETECT_LOOKING_SIDE")),
            "face_hidden"  : (face_hidden_cond,              self._detect("DETECT_FACE_HIDDEN")),
            "partial_face" : (partial_face,                  self._detect("DETECT_PARTIAL_FACE")),
            "fake_presence": (fake_presence_cond,            self._detect("DETECT_FAKE_PRESENCE")),
        }

        for key, (cond, enabled) in head_conditions.items():
            if not enabled:
                continue
            threshold = self._cfg["GAZE_THRESHOLD"] if key == "looking_side" else None
            triggered, dur = self.head_tracker.process(frame, key, cond, threshold=threshold)
            rev = self.risk.process_event(key, triggered, confidence=1.0, duration=dur, now=now)
            self._handle_event(rev, frame, now)

        object_flags: dict[str, tuple[bool, bool, float]] = {
            "phone"    : (phone,     self._detect("DETECT_PHONE"),     phone_conf),
            "book"     : (book,      self._detect("DETECT_BOOK"),      book_conf),
            "headphone": (headphone, self._detect("DETECT_HEADPHONE"), hp_conf),
            "earbud"   : (earbud,    self._detect("DETECT_EARBUD"),    eb_conf),
        }
        for key, (present, enabled, conf) in object_flags.items():
            if not enabled:
                continue
            stable = self.obj_tracker.update(key, present, fps=self._tick_fps)
            rev    = self.risk.process_event(key, stable, confidence=conf, now=now)
            self._handle_event(rev, frame, now)

        if self._detect("DETECT_MULTIPLE_PEOPLE"):
            rev = self.risk.process_event("multiple_people", people_count > 1, now=now)
            self._handle_event(rev, frame, now)

        rev = self.risk.process_event("no_person", no_person_cond, now=now)
        self._handle_event(rev, frame, now)

        if self._detect("DETECT_SPEAKER_AUDIO"):
            rev = self.risk.process_event("speaker_audio", speaker_flagged, now=now)
            self._handle_event(rev, frame, now)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._face_mesh.close()
        self.lip_detector.close()

        if self.audio_monitor:
            self.audio_monitor.stop()

        if self.proof_writer:
            self.proof_writer.flush()

        if self._cfg["SAVE_REPORT"]:
            self._save_report()

        # Notify SSE subscribers that the session has ended
        risk_info = self.risk.get_display()
        self._push_sse({
            "type"      : "session_end",
            "report_id" : self.session_dir.name,
            "risk"      : risk_info,
            "elapsed_s" : round(time.time() - self.session_start, 1),
        })
        # Sentinel to close SSE generators
        for q in self._sse_queues:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _elapsed(self) -> tuple[float, str]:
        elapsed = time.time() - self.session_start
        m, s    = divmod(int(elapsed), 60)
        return round(elapsed, 1), f"{m:02d}:{s:02d}"

    def _on_api_alert(self, message: str) -> None:
        elapsed, ts = self._elapsed()
        entry = {
            "time"     : ts,
            "elapsed_s": elapsed,
            "message"  : message,
        }
        self.alert_log.append(entry)
        try:
            from core.metrics import metrics as _m
            _m.inc_alert()
        except Exception:
            pass

    def _on_warn_notice(self, message: str) -> None:
        elapsed, ts = self._elapsed()
        entry = {
            "time"     : ts,
            "elapsed_s": elapsed,
            "message"  : message,
        }
        self.warning_log.append(entry)
        try:
            from core.metrics import metrics as _m
            _m.inc_warning()
        except Exception:
            pass
        # Push warning to SSE subscribers immediately
        self._push_sse({
            "type"     : "warning",
            "message"  : message,
            "time"     : ts,
            "elapsed_s": elapsed,
            "risk"     : self.risk.get_display(),
        })

    def _handle_event(self, rev, frame: np.ndarray, now: float) -> None:
        self.alert_engine.handle(rev, self._alert_manager)

        if rev.risk_added > 0 and self.alert_log:
            self.alert_log[-1]["score_added"] = round(rev.risk_added, 2)

        # ── Proof capture ─────────────────────────────────────────────────────
        proof_url: str | None = None

        if self.proof_writer is not None:
            save = (rev.terminated and not self._termination_proved) or rev.risk_added > 0
            if save:
                if rev.terminated:
                    self._termination_proved = True
                path = self.proof_writer.save_proof(
                    rev.key, frame, now, audio_monitor=self.audio_monitor,
                )
                if path:
                    try:
                        rel = Path(path).relative_to("reports")
                        proof_url = f"/proof/{rel.as_posix()}"
                    except ValueError:
                        proof_url = None
                    if self.alert_log:
                        self.alert_log[-1]["proof"] = path
                        if proof_url:
                            self.alert_log[-1]["proof_url"] = proof_url

        # ── Push alert to SSE subscribers ─────────────────────────────────────
        # Push when score was added, OR exactly once on termination.
        should_push = (
            rev.risk_added > 0
            or (rev.terminated and not self._termination_sse_sent)
        )
        if should_push:
            if rev.terminated:
                self._termination_sse_sent = True
            elapsed, ts = self._elapsed()
            event: dict = {
                "type"     : "alert",
                "key"      : rev.key,
                "message"  : self.alert_log[-1]["message"] if self.alert_log else rev.key,
                "time"     : ts,
                "elapsed_s": elapsed,
                "score_added": round(rev.risk_added, 2),
                "risk"     : self.risk.get_display(),
            }
            if proof_url:
                event["proof_url"]  = proof_url
                event["proof_type"] = "audio" if rev.key == "speaker_audio" else "image"
            if rev.terminated:
                event["terminated"] = True
            self._push_sse(event)

    def _save_report(self) -> None:
        end_time = time.time()

        alert_summary: dict[str, int] = {}
        for entry in self.alert_log:
            k = entry["message"].split("(")[0].strip()
            alert_summary[k] = alert_summary.get(k, 0) + 1

        warn_summary: dict[str, int] = {}
        for entry in self.warning_log:
            k = entry["message"]
            warn_summary[k] = warn_summary.get(k, 0) + 1

        report = {
            "session_id"      : self.session_id,
            "report_id"       : self.session_dir.name,
            "session_start"   : datetime.fromtimestamp(self.session_start).strftime("%Y-%m-%d %H:%M:%S"),
            "session_end"     : datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s"      : round(end_time - self.session_start, 1),
            "total_api_alerts": len(self.alert_log),
            "total_warnings"  : len(self.warning_log),
            "alert_summary"   : alert_summary,
            "warning_summary" : warn_summary,
            "alert_log"       : self.alert_log,
            "warning_log"     : self.warning_log,
            "risk"            : self.risk.get_summary(),
        }

        path = self.session_dir / "report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[{self.session_id}] Report saved → {path}")
