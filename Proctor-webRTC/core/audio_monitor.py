from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)


class AudioMonitor:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_samples: int,
        speech_threshold: float,
        ring_duration_s: float = 30.0,
    ) -> None:
        self.sample_rate      = sample_rate
        self.channels         = channels
        self.chunk_samples    = chunk_samples
        self.speech_threshold = speech_threshold

        # How many chunks to keep in the ring (~ring_duration_s seconds).
        chunks_per_sec = sample_rate / chunk_samples
        ring_maxlen    = int(chunks_per_sec * ring_duration_s) + 1

        self._lock            = threading.Lock()
        self._stop_event      = threading.Event()
        self._speech_detected = False
        # Timestamped ring: (wall_time, raw_pcm_bytes) per chunk
        self._audio_ring: deque[tuple[float, bytes]] = deque(maxlen=ring_maxlen)
        self._thread: threading.Thread | None = None
        self._stream  = None
        self._pa      = None
        self._model   = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        """Start pyaudio capture thread (single-user / local-runner mode)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-capture")
        self._thread.start()

    def start_webrtc_mode(self) -> None:
        """
        Start a VAD-only background thread for WebRTC push mode.

        Audio chunks arrive via push_audio_chunk() instead of being read
        from a local microphone.  The VAD thread polls the ring buffer and
        updates _speech_detected without opening any pyaudio stream.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_vad_only, daemon=True, name="audio-vad")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass

    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_detected

    def get_audio_range(self, t0: float, t1: float) -> bytes:
        """Return concatenated raw PCM bytes for chunks timestamped in [t0, t1]."""
        with self._lock:
            chunks = [data for ts, data in self._audio_ring if t0 <= ts <= t1]
        return b"".join(chunks)

    # ── Private threads ───────────────────────────────────────────────────────

    def _run(self) -> None:
        """Pyaudio capture + VAD loop (local mic mode)."""
        import time as _time
        import pyaudio  # lazy — only needed for local mic mode, not on server
        try:
            self._model  = load_silero_vad()
            self._pa     = pyaudio.PyAudio()
            self._stream = self._pa.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_samples,
            )
            logger.info("AudioMonitor: local mic capture started (%d Hz)", self.sample_rate)
            while not self._stop_event.is_set():
                data   = self._stream.read(self.chunk_samples, exception_on_overflow=False)
                ts     = _time.time()
                pcm    = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                tensor = torch.from_numpy(pcm)
                t_vad  = _time.perf_counter()
                prob   = self._model(tensor, self.sample_rate).item()
                vad_ms = (_time.perf_counter() - t_vad) * 1000
                try:
                    from core.metrics import metrics as _m
                    _m.record_audio_latency(vad_ms)
                except Exception:
                    pass
                with self._lock:
                    self._speech_detected = prob >= self.speech_threshold
                    self._audio_ring.append((ts, data))
        except Exception as exc:
            self._error = str(exc)
            logger.error("AudioMonitor capture thread failed: %s", exc)

    def _run_vad_only(self) -> None:
        """
        VAD thread for WebRTC push mode.

        Reads new chunks from _audio_ring (written by push_audio_chunk),
        accumulates them into a PCM buffer, and runs silero-VAD every time
        there are at least chunk_samples samples available.

        This is chunk-size agnostic: WebRTC frames are often 480 or 960
        samples which may not match the configured chunk_samples (512).
        """
        import time as _time
        try:
            self._model = load_silero_vad()
            logger.info("AudioMonitor: WebRTC VAD thread started (%d Hz)", self.sample_rate)
            last_processed_ts: float = 0.0
            pcm_buffer = np.empty(0, dtype=np.float32)
            skipped    = 0

            while not self._stop_event.is_set():
                # Collect all new ring entries since last poll
                new_chunks: list[tuple[float, bytes]] = []
                with self._lock:
                    for ts, data in list(self._audio_ring):
                        if ts > last_processed_ts:
                            new_chunks.append((ts, data))

                if new_chunks:
                    last_processed_ts = new_chunks[-1][0]
                    for _, data in new_chunks:
                        try:
                            chunk_pcm  = (
                                np.frombuffer(data, dtype=np.int16)
                                .astype(np.float32) / 32768.0
                            )
                            pcm_buffer = np.concatenate([pcm_buffer, chunk_pcm])
                        except Exception as exc:
                            skipped += 1
                            if skipped <= 5 or skipped % 100 == 0:
                                logger.warning(
                                    "AudioMonitor: skipped bad PCM chunk #%d: %s", skipped, exc
                                )

                    # Process full windows
                    while len(pcm_buffer) >= self.chunk_samples:
                        window     = pcm_buffer[:self.chunk_samples]
                        pcm_buffer = pcm_buffer[self.chunk_samples:]
                        try:
                            tensor = torch.from_numpy(window)
                            t_vad  = _time.perf_counter()
                            prob   = self._model(tensor, self.sample_rate).item()
                            vad_ms = (_time.perf_counter() - t_vad) * 1000
                            try:
                                from core.metrics import metrics as _m
                                _m.record_audio_latency(vad_ms)
                            except Exception:
                                pass
                            with self._lock:
                                self._speech_detected = prob >= self.speech_threshold
                        except Exception as exc:
                            logger.debug("AudioMonitor: VAD inference error: %s", exc)

                _time.sleep(0.033)  # ~30 Hz polling

        except Exception as exc:
            self._error = str(exc)
            logger.error("AudioMonitor VAD thread failed: %s", exc)


class SpeakerAudioDetector:
    """Detects speech activity that is not accompanied by visible lip movement."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s           = hold_s
        self._no_lips_since: float | None = None
        self._flagged          = False

    def update(
        self,
        speech_active: bool,
        lip_speaking:  bool,
        face_detected: bool,
        timestamp:     float,
    ) -> bool:
        # Flag if audio is active AND (no face visible OR lips not moving).
        desync = speech_active and (not face_detected or not lip_speaking)

        if desync:
            if self._no_lips_since is None:
                self._no_lips_since = timestamp
            elif (not self._flagged) and (timestamp - self._no_lips_since >= self._hold_s):
                self._flagged = True
        else:
            self._no_lips_since = None
            self._flagged       = False

        return self._flagged
