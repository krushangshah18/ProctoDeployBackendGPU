from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

from config import (
    LIP_DYNAMIC_STD_MIN,
    LIP_HISTORY,
    LIP_MAR_SPEAKING,
    LIP_MAR_YAWN,
    LIP_YAWN_DURATION_S,
)

_TOP = 13
_BOT = 14
_LEFT = 78
_RIGHT = 308

_LIP_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
]

_C_STILL = (150, 150, 150)
_C_SPEAK = (0, 210, 0)
_C_YAWN = (0, 140, 255)
_C_MAR = (0, 230, 230)


@dataclass
class LipState:
    mar: float = 0.0
    is_open: bool = False
    is_speaking: bool = False
    is_yawning: bool = False
    face_detected: bool = False


class LipDetector:
    def __init__(self, own_mesh: bool = True) -> None:
        """
        Args:
            own_mesh: If True (default), create an internal FaceMesh instance.
                      Set False when using a shared FaceMeshProvider — the
                      provider's landmarks are passed via process(..., landmarks=).
        """
        if own_mesh:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._mesh = None
        self._mar_history: deque[float] = deque(maxlen=LIP_HISTORY)
        self._yawn_start: float | None = None
        self._last_landmarks = None

    def process(
        self,
        frame: np.ndarray,
        timestamp: float,
        draw: bool = True,
        landmarks=None,
    ) -> LipState:
        """
        Args:
            frame:     BGR numpy frame.
            timestamp: Monotonic session clock value (for yawn duration tracking).
            draw:      Whether to draw the lip overlay onto frame.
            landmarks: Pre-computed landmark list from FaceMeshProvider.
                       When provided, the internal FaceMesh call is skipped.
                       Pass None to use the internal FaceMesh (own_mesh=True mode).
        """
        h, w = frame.shape[:2]

        if landmarks is None:
            # Standalone path — use own FaceMesh instance
            if self._mesh is None:
                # own_mesh=False but no landmarks supplied → treat as no face
                return LipState(face_detected=False)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self._mesh.process(rgb)
            except Exception as e:
                logger.warning("LipDetector: MediaPipe inference error: %s", e)
                if draw:
                    _f, _s, _t = cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
                    _txt = f"lips: error {str(e)[:20]}"
                    (tw, th), _ = cv2.getTextSize(_txt, _f, _s, _t)
                    cv2.rectangle(frame, (17, 397 - th), (17 + tw + 6, 402), (20, 20, 20), -1)
                    cv2.putText(frame, _txt, (20, 400), _f, _s, (0, 0, 255), _t, cv2.LINE_AA)
                return LipState(face_detected=False)

            if not results.multi_face_landmarks:
                self._last_landmarks = None
                if draw:
                    _f, _s, _t = cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
                    _txt = "lips: no face"
                    (tw, th), _ = cv2.getTextSize(_txt, _f, _s, _t)
                    cv2.rectangle(frame, (17, 397 - th), (17 + tw + 6, 402), (20, 20, 20), -1)
                    cv2.putText(frame, _txt, (20, 400), _f, _s, (80, 80, 180), _t, cv2.LINE_AA)
                return LipState(face_detected=False)

            landmarks = results.multi_face_landmarks[0].landmark

        # landmarks is now the raw landmark list (either from provider or own mesh)
        self._last_landmarks = landmarks

        def px(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

        top = px(_TOP)
        bot = px(_BOT)
        left = px(_LEFT)
        right = px(_RIGHT)

        vertical = float(np.linalg.norm(top - bot))
        horizontal = float(np.linalg.norm(left - right))
        mar = vertical / (horizontal + 1e-6)
        self._mar_history.append(mar)

        is_open = mar > LIP_MAR_SPEAKING
        is_yawning = self._update_yawn(mar, timestamp)
        is_speaking = is_open and self._is_dynamic() and not is_yawning

        state = LipState(
            mar=round(mar, 4),
            is_open=is_open,
            is_speaking=is_speaking,
            is_yawning=is_yawning,
            face_detected=True,
        )
        if draw:
            self._draw_overlay(frame, state)
        return state

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()

    def _draw_overlay(self, frame: np.ndarray, state: LipState) -> None:
        if self._last_landmarks is None:
            return

        h, w = frame.shape[:2]
        lip_color = _C_YAWN if state.is_yawning else (_C_SPEAK if state.is_speaking else _C_STILL)
        points = [
            (int(self._last_landmarks[idx].x * w), int(self._last_landmarks[idx].y * h))
            for idx in _LIP_OUTLINE
        ]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, lip_color, 2)

        pt_top = (
            int(self._last_landmarks[_TOP].x * w),
            int(self._last_landmarks[_TOP].y * h),
        )
        pt_bot = (
            int(self._last_landmarks[_BOT].x * w),
            int(self._last_landmarks[_BOT].y * h),
        )
        cv2.line(frame, pt_top, pt_bot, _C_MAR, 2)
        cv2.circle(frame, pt_top, 3, _C_MAR, -1)
        cv2.circle(frame, pt_bot, 3, _C_MAR, -1)

        label = "YAWN" if state.is_yawning else ("SPEAKING" if state.is_speaking else "STILL")
        _txt = f"lips  MAR:{state.mar:.3f}  {label}"
        _f, _s, _t = cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
        (tw, th), _ = cv2.getTextSize(_txt, _f, _s, _t)
        cv2.rectangle(frame, (17, 397 - th), (17 + tw + 6, 402), (20, 20, 20), -1)
        cv2.putText(frame, _txt, (20, 400), _f, _s, lip_color, _t, cv2.LINE_AA)

    def _update_yawn(self, mar: float, timestamp: float) -> bool:
        if mar > LIP_MAR_YAWN:
            if self._yawn_start is None:
                self._yawn_start = timestamp
            elif timestamp - self._yawn_start >= LIP_YAWN_DURATION_S:
                return True
        else:
            self._yawn_start = None
        return False

    def _is_dynamic(self) -> bool:
        if len(self._mar_history) < 6:
            return False
        return float(np.std(list(self._mar_history)[-12:])) >= LIP_DYNAMIC_STD_MIN
