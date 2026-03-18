from __future__ import annotations

import logging
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProofWriter:
    """
    Captures lightweight proof evidence for proctoring alert events.

    Proof types:
      image — single JPEG frame saved synchronously for every alert.
      audio — WAV clip (last N seconds from ring buffer) for speaker_audio events.

    No rolling frame buffer, no video encoding, no ffmpeg dependency.
    All writes are fast and synchronous.

    Usage:
        pw = ProofWriter("reports/proof", audio_pre_s=5.0)
        # on alert:
        path = pw.save_proof(key, frame, time.time(), audio_monitor=am)
        # on shutdown:
        pw.flush()   # no-op, kept for API compatibility
    """

    # Keys that save a JPEG image proof
    _PROOF_IMAGE = frozenset({
        "phone", "book", "headphone", "earbud",
        "multiple_people", "no_person",
        "looking_away", "looking_down", "looking_up", "looking_side",
        "face_hidden", "partial_face", "fake_presence",
        "tab_switch",
    })

    # Keys that additionally save a WAV audio clip
    _PROOF_AUDIO = frozenset({"speaker_audio"})

    def __init__(
        self,
        proof_dir:    str,
        audio_pre_s:  float = 5.0,
    ) -> None:
        self._dir         = Path(proof_dir)
        self._audio_pre_s = audio_pre_s

    # ── Public ────────────────────────────────────────────────────────────────

    def save_proof(
        self,
        key:            str,
        frame:          np.ndarray,
        timestamp:      float,
        audio_monitor=  None,
        is_termination: bool = False,
    ) -> str | None:
        """
        Save proof for the given event key. Returns the saved file path or None.

        - Every key saves a JPEG frame (synchronous, ~1–5 ms).
        - speaker_audio additionally saves a WAV clip from the audio ring buffer.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H%M%S_%f")[:9]

        # Always save a JPEG frame
        img_path = self._save_image(key, frame, ts_str)

        # Additionally save audio WAV for speaker_audio events
        if key in self._PROOF_AUDIO and audio_monitor is not None:
            self._save_audio_clip(key, ts_str, timestamp, audio_monitor)

        return img_path

    def flush(self) -> None:
        """No-op — kept for API compatibility."""

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_image(self, key: str, frame: np.ndarray, ts_str: str) -> str:
        path = str(self._dir / f"{key}_{ts_str}.jpg")
        ok   = cv2.imwrite(path, frame)
        if not ok:
            logger.error("ProofWriter: failed to write image %s", path)
        return path

    def _save_audio_clip(
        self,
        key:           str,
        ts_str:        str,
        timestamp:     float,
        audio_monitor,
    ) -> str | None:
        """Extract the last audio_pre_s seconds from the ring buffer and write a WAV."""
        try:
            t0        = timestamp - self._audio_pre_s
            audio_raw = audio_monitor.get_audio_range(t0, timestamp)
            if not audio_raw:
                return None
            path = str(self._dir / f"{key}_{ts_str}.wav")
            self._write_wav(path, audio_raw, audio_monitor.sample_rate, audio_monitor.channels)
            return path
        except Exception as exc:
            logger.error("ProofWriter: failed to write audio clip: %s", exc)
            return None

    @staticmethod
    def _write_wav(path: str, raw: bytes, sample_rate: int, channels: int) -> None:
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(raw)
        except Exception as exc:
            logger.error("ProofWriter: failed to write WAV %s: %s", path, exc)
