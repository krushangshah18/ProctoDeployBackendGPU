"""
Per-process FaceMesh singleton for ProcessPoolExecutor.

Each worker process maintains one FaceMesh instance (static_image_mode=True)
that processes individual frames independently — correct for pool workers that
serve frames from multiple sessions without consistent ordering.

Landmarks are serialized as a float32 numpy array for inter-process transfer:
  - 478 landmarks × 3 floats (x, y, z) = 5,736 bytes when refine_landmarks=True
  - Coordinator pre-downscales frames to _FM_H × _FM_W before submission to
    minimise IPC cost (4× less data than 640×480).
  - Normalized landmark coords [0,1] are independent of input resolution, so
    HeadPoseDetector and LipDetector compute correct pixel positions when they
    multiply by the original full-resolution frame dimensions.
"""
from __future__ import annotations

import numpy as np
import cv2

# Per-process singleton — initialised once per worker by the pool initializer.
_face_mesh = None


def init_facemesh_worker(refine_landmarks: bool = True) -> None:
    """
    Pool initializer: create one FaceMesh instance per worker process.
    Called automatically by ProcessPoolExecutor before the first task runs.
    """
    global _face_mesh
    import mediapipe as mp  # imported inside worker so the main process stays lean

    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        # static_image_mode=True: each frame is treated independently — no
        # temporal tracking state carried between sessions or ticks.
        # Required for correctness when multiple sessions share a pool of workers.
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(frame_bytes: bytes, h: int, w: int) -> bytes | None:
    """
    Run FaceMesh on a BGR frame received as raw bytes.

    Returns:
        float32 bytes of shape (N, 3) where N = 478 (refine=True) or 468,
        or None if no face was detected or inference failed.
    """
    global _face_mesh
    if _face_mesh is None:
        # Lazy init fallback if the initializer somehow wasn't called
        init_facemesh_worker()

    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, 3)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = _face_mesh.process(rgb)
    except Exception:
        return None

    if not results.multi_face_landmarks:
        return None

    lms = results.multi_face_landmarks[0].landmark
    n   = len(lms)
    arr = np.empty((n, 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
    return arr.tobytes()
