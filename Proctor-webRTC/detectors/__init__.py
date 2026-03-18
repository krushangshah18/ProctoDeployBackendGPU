"""
detectors — vision detection components.

    FaceMeshProvider    — single shared MediaPipe FaceMesh per session (run once, share landmarks)
    ObjectDetector      — YOLO-based object detection (phone, book, headphone, earbud, person)
    HeadPoseDetector    — MediaPipe head pose + gaze + blink detection
    LipDetector         — MediaPipe lip / MAR speaking / yawn detection
    merge_by_class      — IoU-based duplicate suppression helper
    LipState            — dataclass returned by LipDetector.process()
"""
from .face_mesh_provider import FaceMeshProvider
from .object_detector    import ObjectDetector, merge_by_class
from .head_pose_detector import HeadPoseDetector
from .lip_detector       import LipDetector, LipState

__all__ = [
    "FaceMeshProvider",
    "ObjectDetector",
    "merge_by_class",
    "HeadPoseDetector",
    "LipDetector",
    "LipState",
]
