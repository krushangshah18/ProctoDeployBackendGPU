"""
core — proctoring engine package.

Public surface (importable directly from `core`):
    Coordinators / Sessions
        ProctorCoordinator  — multi-user inference orchestrator
        ProctorSession      — per-candidate state container

    Risk / Alert pipeline
        RiskEngine, ExamState   — scoring state machine
        AlertEngine             — warn vs alert routing

    Detectors / Trackers
        HeadTracker             — duration-gated head/gaze state machine
        LivenessDetector        — static-face / fake-presence detection
        ObjectTemporalTracker   — fps-aware object stability filter

    Audio
        AudioMonitor            — VAD ring buffer (local mic + WebRTC push)
        SpeakerAudioDetector    — speech-without-lips detector
"""

from .alert_engine          import AlertEngine
from .audio_monitor         import AudioMonitor, SpeakerAudioDetector
from .head_tracker          import HeadTracker
from .liveness              import LivenessDetector
from .object_tracker        import ObjectTemporalTracker
from .risk_engine           import RiskEngine, ExamState
from .proctor_session       import ProctorSession
from .proctor_coordinator   import ProctorCoordinator

__all__ = [
    # Coordinator / Session
    "ProctorCoordinator",
    "ProctorSession",
    # Risk / Alert
    "RiskEngine",
    "ExamState",
    "AlertEngine",
    # Detectors / Trackers
    "HeadTracker",
    "LivenessDetector",
    "ObjectTemporalTracker",
    # Audio
    "AudioMonitor",
    "SpeakerAudioDetector",
]
