"""
FaceMeshProvider — single shared MediaPipe FaceMesh instance per session.

Both HeadPoseDetector and LipDetector need face landmarks from the same frame.
Running FaceMesh once and sharing the landmarks halves MediaPipe inference time
and saves ~150 MB RAM per candidate (one FaceMesh model load instead of two).

Usage in ProctorSession:
    self._face_mesh = FaceMeshProvider()
    self.head_detector = HeadPoseDetector(debug=True, own_mesh=False)
    self.lip_detector  = LipDetector(own_mesh=False)

    # in run_mediapipe():
    landmarks = self._face_mesh.process(frame)   # one call
    head_result = self.head_detector.detect(frame, landmarks=landmarks, draw=False)
    lip_result  = self.lip_detector.process(frame, ts, landmarks=landmarks, draw=False)
"""
from __future__ import annotations

import logging

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class FaceMeshProvider:
    """
    Runs MediaPipe FaceMesh once per frame and returns the raw landmark list.

    Parameters match the settings both HeadPoseDetector and LipDetector
    previously used independently:
      - refine_landmarks=True  is required for iris landmarks (indices 468, 473)
        used by HeadPoseDetector for gaze tracking.
    """

    def __init__(
        self,
        max_num_faces:           int   = 1,
        refine_landmarks:        bool  = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray):
        """
        Run FaceMesh on one BGR frame.

        Returns the landmark list (RepeatedCompositeContainer, 478 points when
        refine_landmarks=True) for the first detected face, or None if no face
        was found or an error occurred.

        Both HeadPoseDetector and LipDetector accept this return value directly
        via their ``landmarks`` parameter.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self._mesh.process(rgb)
        except Exception as e:
            logger.warning("FaceMeshProvider: inference error: %s", e)
            return None

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0].landmark

    def close(self) -> None:
        self._mesh.close()
