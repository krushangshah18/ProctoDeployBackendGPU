"""
ProctorCoordinator — multi-user inference orchestrator.

Owns the single shared YOLO model and drives the per-tick pipeline:

  Every tick:
  1. Snapshot latest frames from all active ProctorSessions
  2. Single YOLO batch call  → one forward pass for N candidates
  3. Parallel MediaPipe       → ThreadPoolExecutor, GIL released in C++
  4. Per-user state update    → sequential (fast CPU work)

Also owns the exam-level detection config (exam_config). This is a mutable
dict that admin can update at runtime. All active sessions read from it
via a shared reference, so toggles take effect immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from detectors.object_detector import ObjectDetector, merge_by_class
from detectors.lip_detector import LipState as _LipState
from core.proctor_session import ProctorSession

# Null MediaPipe result used when all face/head/lip detections are disabled.
# Shape matches the real (head_result, lip_result) tuple so update() unpacks cleanly.
_NULL_HEAD = (False, False, False, False, False, False, 0.0, 0.0, 0.0, 0, False, 0)
_NULL_MP   = (_NULL_HEAD, _LipState(face_detected=False))

logger = logging.getLogger("coordinator")

# Target processing rate (ticks per second).
# Actual rate is bounded by YOLO batch latency — if inference takes longer
# than 1/TICK_RATE the loop simply runs as fast as it can.
# 10 Hz is a good balance: 100 ms window fits head-pose gates (1.5–2 s)
# and gives ~50% more headroom per tick than 15 Hz.
TICK_RATE = 10   # hz


class ProctorCoordinator:
    """
    Single instance, shared across all WebRTC sessions.

    Lifecycle (called from server.py):
        coordinator = ProctorCoordinator("finalBestV5.pt", max_sessions=40)
        await coordinator.start()

        coordinator.add_session(pc_id, session)   # on new WebRTC connection
        coordinator.remove_session(pc_id)          # on disconnect

        await coordinator.stop()                   # on server shutdown
    """

    def __init__(
        self,
        model_path:    str   = "finalBestV5.pt",
        max_sessions:  int   = 5,
        tick_rate:     int   = TICK_RATE,
        device:        str   = "cpu",
        default_conf:  float = 0.50,
        person_conf:   float = 0.30,
        phone_conf:    float = 0.65,
        book_conf:     float = 0.70,
        audio_conf:    float = 0.41,
        half:          bool  = False,
        warmup_frames: int   = 0,
        min_vram_gb:   float = 1.5,
    ):
        self.detector = ObjectDetector(
            model_path    = model_path,
            device        = device,
            default_conf  = default_conf,
            person_conf   = person_conf,
            phone_conf    = phone_conf,
            book_conf     = book_conf,
            audio_conf    = audio_conf,
            half          = half,
            warmup_frames = warmup_frames,
            min_vram_gb   = min_vram_gb,
        )
        self.max_sessions = max_sessions
        self.tick_rate    = tick_rate

        # pc_id → ProctorSession
        self.sessions: dict[str, ProctorSession] = {}

        # Exam-level detection config — admin can update this at runtime.
        # All sessions hold a reference to this dict, so changes apply immediately.
        self.exam_config: dict = {}

        # Dedicated thread pool for parallel MediaPipe calls.
        # On GPU the YOLO batch finishes fast (~15ms), so all N MediaPipe calls
        # start in parallel.  MediaPipe releases the GIL during C++ compute, so
        # true parallelism is achieved up to the number of physical CPU cores.
        # Cap at 32 to match RTX PRO 4500 vCPU allocation; min 8 on any GPU pod.
        _mp_workers = min(max(max_sessions, 8), 32)
        self._mp_pool = ThreadPoolExecutor(
            max_workers=_mp_workers,
            thread_name_prefix="mediapipe",
        )

        self._running  = False
        self._task: asyncio.Task | None = None

        # Diagnostics
        self._last_tick_ms: float = 0.0
        self._tick_count:   int   = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task    = asyncio.create_task(self._tick_loop(), name="coordinator")
        logger.info(
            "ProctorCoordinator started (tick_rate=%d Hz  device=%s)",
            self.tick_rate, self.detector.device,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._mp_pool.shutdown(wait=False)
        logger.info("ProctorCoordinator stopped")

    # ── Session management ────────────────────────────────────────────────────

    def add_session(self, pc_id: str, session: ProctorSession) -> None:
        # Give the session a live reference to the shared exam config so that
        # admin toggle changes apply immediately to all active sessions.
        session._exam_config = self.exam_config
        self.sessions[pc_id] = session
        logger.info("Session added: %s  (%d/%d active)",
                    pc_id, len(self.sessions), self.max_sessions)

    def remove_session(self, pc_id: str) -> None:
        session = self.sessions.pop(pc_id, None)
        if session:
            session.close()
            logger.info("Session removed: %s  (%d/%d active)",
                        pc_id, len(self.sessions), self.max_sessions)

    def update_exam_config(self, overrides: dict) -> dict:
        """
        Apply DETECT_* overrides to the shared exam config.
        All active sessions see the change immediately (shared reference).
        Returns the full updated config.
        """
        valid = {k: bool(v) for k, v in overrides.items() if k.startswith("DETECT_")}
        self.exam_config.update(valid)
        logger.info("Exam config updated: %s", valid)
        return dict(self.exam_config)

    # ── Tick loop ─────────────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        loop     = asyncio.get_running_loop()
        interval = 1.0 / self.tick_rate

        while self._running:
            tick_start = time.perf_counter()

            if self.sessions:
                try:
                    await self._tick(loop)
                except Exception:
                    logger.exception("Error in coordinator tick")

            elapsed = time.perf_counter() - tick_start
            self._last_tick_ms = elapsed * 1000
            self._tick_count  += 1

            # Push to metrics for /metrics and /system/report endpoints
            try:
                from core.metrics import metrics as _m
                _m.record_tick_latency(self._last_tick_ms)
            except Exception:
                pass

            sleep = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep)

    async def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        # 1. Snapshot sessions — skip terminated ones (no more scoring needed)
        snapshot = {
            pc_id: session
            for pc_id, session in list(self.sessions.items())
            if session.latest_frame is not None and not session.risk.terminated
        }
        if not snapshot:
            return

        pc_ids   = list(snapshot.keys())
        sessions = [snapshot[k] for k in pc_ids]
        frames   = [s.latest_frame for s in sessions]
        fps_vals = [s.observed_fps  for s in sessions]
        now      = time.time()

        # 2. Batch YOLO — one forward pass for all N frames
        batch_detections: list[list[dict]] = await loop.run_in_executor(
            None,
            self.detector.detect_batch,
            frames,
        )

        # 3. Parallel MediaPipe — only for sessions that actually need it.
        # If ALL face/head/lip detections are disabled for a session, skip
        # FaceMesh + HeadPoseDetector + LipDetector entirely: no thread-pool
        # dispatch, no C++ compute — just an already-resolved future with _NULL_MP.
        mp_tasks  = []
        mp_needed = 0
        for session, frame in zip(sessions, frames):
            if session.needs_mediapipe():
                mp_tasks.append(
                    loop.run_in_executor(self._mp_pool, session.run_mediapipe, frame)
                )
                mp_needed += 1
            else:
                f = loop.create_future()
                f.set_result(_NULL_MP)
                mp_tasks.append(f)

        mp_t0      = time.perf_counter()
        mp_results = await asyncio.gather(*mp_tasks)
        if mp_needed > 0:
            mp_elapsed_ms = (time.perf_counter() - mp_t0) * 1000
            try:
                from core.metrics import metrics as _m
                _m.record_mediapipe_latency(mp_elapsed_ms)
            except Exception:
                pass

        # 4. Per-user state update — sequential (fast CPU work)
        for session, raw_dets, mp_result, frame, fps in zip(
            sessions, batch_detections, mp_results, frames, fps_vals
        ):
            detections = (
                merge_by_class(raw_dets, ["person", "earbud"], iou_threshold=0.5)
                if len(raw_dets) > 1
                else raw_dets
            )
            session.update(detections, mp_result, frame, now, fps)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "active_sessions" : len(self.sessions),
            "max_sessions"    : self.max_sessions,
            "tick_rate_target": self.tick_rate,
            "last_tick_ms"    : round(self._last_tick_ms, 2),
            "total_ticks"     : self._tick_count,
            "device"          : self.detector.device,
            "device_info"     : self.detector.device_info,
        }
