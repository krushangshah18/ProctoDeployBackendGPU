from __future__ import annotations

import time
from collections import deque


class LivenessDetector:
    def __init__(self, window: float, interval: float, min_variance: float,
                 blink_timeout: float, weights: dict) -> None:
        self.window        = window
        self.interval      = interval
        self.min_variance  = min_variance
        self.blink_timeout = blink_timeout
        self.weights       = weights

        self._yaw:   deque[tuple[float, float]] = deque()
        self._pitch: deque[tuple[float, float]] = deque()
        self._gaze:  deque[tuple[float, float]] = deque()

        self.last_blink = time.time()

    def _variance(self, values: list[float]) -> float:
        if len(values) < 10:
            return 1.0  # not enough data → assume real
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def update(self, yaw: float, pitch: float, gaze: float, blinked: bool) -> None:
        now = time.time()

        if not self._yaw or now - self._yaw[-1][0] > self.interval:
            self._yaw.append((now, yaw))
            self._pitch.append((now, pitch))
            self._gaze.append((now, gaze))

        cutoff = now - self.window
        for buf in (self._yaw, self._pitch, self._gaze):
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        if blinked:
            self.last_blink = now

    def is_fake(self) -> bool:
        yaw_var   = self._variance([v for _, v in self._yaw])
        pitch_var = self._variance([v for _, v in self._pitch])
        gaze_var  = self._variance([v for _, v in self._gaze])

        score = (
            self.weights["yaw"]   * yaw_var +
            self.weights["gaze"]  * gaze_var +
            self.weights["pitch"] * pitch_var
        )
        static   = score < self.min_variance
        no_blink = (time.time() - self.last_blink) > self.blink_timeout
        return static and no_blink




