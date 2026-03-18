"""
utils — shared utilities for alerting, drawing, and proof capture.

    AlertManager    — two-tier warn / alert display manager
    ProofWriter     — evidence capture (JPEG / MP4 / MKV) around alert events
    draw_alerts     — render warn/alert banners onto a CV2 frame
    draw_audio_status — render MIC activity indicator onto a CV2 frame
    draw_detections — render YOLO bounding boxes onto a CV2 frame
"""
from .alerts         import AlertManager
from .draw           import draw_alerts, draw_audio_status, draw_detections
from .logging_config import setup_logging
from .proof_writer   import ProofWriter

__all__ = [
    "AlertManager",
    "ProofWriter",
    "draw_alerts",
    "draw_audio_status",
    "draw_detections",
    "setup_logging",
]
