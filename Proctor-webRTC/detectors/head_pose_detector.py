import cv2
import math
import mediapipe as mp

import numpy as np
from config import (
    LOOK_AWAY_YAW, LOOK_DOWN_PITCH, LOOK_UP_PITCH,
    GAZE_LEFT, GAZE_RIGHT,
    EAR_THRESHOLD, BLINK_FRAMES,
    MIN_FACE_WIDTH, MIN_FACE_HEIGHT,
)


class HeadPoseDetector:
    def __init__(self, debug=False, own_mesh: bool = True):
        """
        Args:
            debug:    Enable draw_debug overlays.
            own_mesh: If True (default), create an internal FaceMesh instance.
                      Set False when using a shared FaceMeshProvider — the
                      provider's landmarks are passed via detect(..., landmarks=).
        """
        if own_mesh:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None
        self.DEBUG = debug

        #IDs of specific face points
        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        self.FOREHEAD = 10
        self.CHIN = 152

        # Left eye
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.LEFT_IRIS = 468

        # Right eye
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        self.RIGHT_IRIS = 473

        # Face Size Constraints — sourced from config.py
        self.MIN_FACE_WIDTH  = MIN_FACE_WIDTH
        self.MIN_FACE_HEIGHT = MIN_FACE_HEIGHT


        # Eye landmarks (MediaPipe)
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

        # Blink config — sourced from config.py
        self.EAR_THRESHOLD = EAR_THRESHOLD
        self.BLINK_FRAMES  = BLINK_FRAMES
        self.blink_counter = 0
        self.total_blinks  = 0

        # Head Pose Thresholds — sourced from config.py
        self.LOOK_AWAY_YAW   = LOOK_AWAY_YAW
        self.LOOK_DOWN_PITCH = LOOK_DOWN_PITCH
        self.LOOK_UP_PITCH   = LOOK_UP_PITCH
        self.GAZE_LEFT       = GAZE_LEFT
        self.GAZE_RIGHT      = GAZE_RIGHT

    

    # Utility Functions
    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _eye_aspect_ratio(self, eye):
        A = self._dist(eye[1], eye[5])
        B = self._dist(eye[2], eye[4])
        C = self._dist(eye[0], eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def draw_debug(
        self,
        frame,
        result_tuple,
        show_gaze: bool = True,
        show_pose: bool = True,
        show_liveness: bool = True,
    ) -> None:
        """
        Draw MediaPipe debug overlays onto *frame* in-place.

        Args:
            frame:        BGR numpy frame to draw on (can be a copy).
            result_tuple: The 12-element tuple returned by detect().
            show_gaze / show_pose / show_liveness: sub-gates.
        """
        (
            looking_away, looking_down, looking_up,
            looking_left, looking_right,
            _partial_face,
            yaw_ratio, pitch_ratio, gaze_ratio,
            ear, _blinked, _total_blinks,
        ) = result_tuple

        # We need the raw landmark pixel positions — recompute from the
        # last processed frame's dimensions stored on self.
        if not hasattr(self, '_last_landmarks') or self._last_landmarks is None:
            return

        landmarks = self._last_landmarks
        h, w = frame.shape[:2]

        def px(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        nose       = px(self.NOSE_TIP)
        left_cheek = px(self.LEFT_CHEEK)
        right_cheek = px(self.RIGHT_CHEEK)
        forehead   = px(self.FOREHEAD)
        chin       = px(self.CHIN)

        face_center_x = (left_cheek[0] + right_cheek[0]) // 2
        face_center_y = (forehead[1]   + chin[1])         // 2

        le_left  = px(self.LEFT_EYE_LEFT)
        le_right = px(self.LEFT_EYE_RIGHT)
        le_iris  = px(self.LEFT_IRIS)
        re_left  = px(self.RIGHT_EYE_LEFT)
        re_right = px(self.RIGHT_EYE_RIGHT)
        re_iris  = px(self.RIGHT_IRIS)

        left_eye  = [px(i) for i in self.LEFT_EYE_POINTS]
        right_eye = [px(i) for i in self.RIGHT_EYE_POINTS]

        _font  = cv2.FONT_HERSHEY_SIMPLEX
        _small = 0.42
        _thick = 1

        def _tag(text, x, y, color):
            (tw, th), _ = cv2.getTextSize(text, _font, _small, _thick)
            cv2.rectangle(frame, (x - 3, y - th - 2), (x + tw + 3, y + 3),
                          (20, 20, 20), -1)
            cv2.putText(frame, text, (x, y), _font, _small, color, _thick, cv2.LINE_AA)

        if show_gaze:
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (180, 0, 180), -1)
            cv2.circle(frame, le_iris, 3, (255, 80, 255), -1)
            cv2.circle(frame, re_iris, 3, (255, 80, 255), -1)

        if show_pose:
            cv2.circle(frame, nose, 4, (0, 220, 220), -1)
            cv2.line(frame, (left_cheek[0], face_center_y),
                     (right_cheek[0], face_center_y), (0, 200, 80), 1)
            cv2.line(frame, (face_center_x, forehead[1]),
                     (face_center_x, chin[1]), (200, 200, 0), 1)

            tag_x = frame.shape[1] - 130
            tag_y_start = 120
            step = 18

            yaw_col   = (80, 80, 255) if looking_away else (100, 220, 100)
            pitch_col = (80, 80, 255) if (looking_down or looking_up) else (100, 220, 100)
            gaze_col  = (80, 80, 255) if (looking_left or looking_right) else (100, 220, 100)

            _tag(f"Yaw  {yaw_ratio:+.2f}",  tag_x, tag_y_start,        yaw_col)
            _tag(f"Ptch {pitch_ratio:+.2f}", tag_x, tag_y_start + step,  pitch_col)
            _tag(f"Gaze {gaze_ratio:+.2f}",  tag_x, tag_y_start + step*2, gaze_col)

        if show_liveness:
            tag_x = frame.shape[1] - 130
            tag_y  = 120 + 18 * 3 if show_pose else 120
            ear_col = (80, 80, 255) if ear < self.EAR_THRESHOLD else (100, 220, 100)
            _tag(f"EAR  {ear:.2f}",          tag_x, tag_y,      ear_col)
            _tag(f"Blnk {self.total_blinks}", tag_x, tag_y + 18, (160, 160, 255))

    def detect(self, frame, draw=True,
               show_gaze=True, show_pose=True, show_liveness=True,
               landmarks=None):
        """
        Args:
            draw          — master draw gate (DEBUG flag)
            show_gaze     — iris dots + eye corner points
            show_pose     — nose dot + H/V lines + yaw/pitch/gaze text
            show_liveness — EAR + blink count text
            landmarks     — pre-computed landmark list from FaceMeshProvider.
                            When provided, the internal FaceMesh call is skipped.
                            Pass None to use the internal FaceMesh (own_mesh=True mode).
        """
        h, w = frame.shape[:2]
        self._last_landmarks = None  # reset each frame

        if landmarks is None:
            # Standalone path — use own FaceMesh instance
            if self.face_mesh is None:
                # own_mesh=False but no landmarks supplied → treat as no face
                return (False, False, False, False, False, False, 0.0, 0.0, 0.0, 0, False, 0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                self.blink_counter = 0
                return (False, False, False, False, False, False, 0.0, 0.0, 0.0, 0, False, 0)
            landmarks = results.multi_face_landmarks[0].landmark
        self._last_landmarks = landmarks  # cache for draw_debug()

        # Convert only required points
        def px(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)
        
        nose = px(self.NOSE_TIP)
        left_cheek = px(self.LEFT_CHEEK)
        right_cheek = px(self.RIGHT_CHEEK)
        forehead = px(self.FOREHEAD)
        chin = px(self.CHIN)

        # Face geometry
        face_width = max(1, right_cheek[0] - left_cheek[0])
        face_height = max(1, chin[1] - forehead[1])
        face_center_x = (left_cheek[0] + right_cheek[0]) // 2
        face_center_y = (forehead[1] + chin[1]) // 2

        partial_face = (
            face_width < self.MIN_FACE_WIDTH or
            face_height < self.MIN_FACE_HEIGHT
        )

        # Head Pose
        """
        raw pixel distance is unreliable. We normalize using face width.
        
        yaw_ratio measures the horizontal displacement of the nose from the face center, 
        normalized by face width, which estimates how much the head is rotated left or right
        
        ≈0  Straight
        +	Looking right
        -	Looking left
        """
        yaw_ratio = (nose[0] - face_center_x) / face_width
        pitch_ratio = (nose[1] - face_center_y) / face_height

        looking_away = abs(yaw_ratio) > self.LOOK_AWAY_YAW
        looking_down = pitch_ratio > self.LOOK_DOWN_PITCH
        looking_up = pitch_ratio < self.LOOK_UP_PITCH

        # Gaze
        le_left = px(self.LEFT_EYE_LEFT)
        le_right = px(self.LEFT_EYE_RIGHT)
        le_iris = px(self.LEFT_IRIS)

        re_left = px(self.RIGHT_EYE_LEFT)
        re_right = px(self.RIGHT_EYE_RIGHT)
        re_iris = px(self.RIGHT_IRIS)

        left_eye_width = max(1, le_right[0] - le_left[0])
        right_eye_width = max(1, re_right[0] - re_left[0])

        left_gaze = (le_iris[0] - (le_left[0] + le_right[0]) // 2) / left_eye_width
        right_gaze = (re_iris[0] - (re_left[0] + re_right[0]) // 2) / right_eye_width
        gaze_ratio = (left_gaze + right_gaze) / 2

        looking_left = gaze_ratio < self.GAZE_LEFT
        looking_right = gaze_ratio > self.GAZE_RIGHT

        # Blink Detection
        left_eye = [px(i) for i in self.LEFT_EYE_POINTS]
        right_eye = [px(i) for i in self.RIGHT_EYE_POINTS]

        ear = (
            self._eye_aspect_ratio(left_eye) +
            self._eye_aspect_ratio(right_eye)
        ) / 2.0

        blinked = False

        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_FRAMES:
                self.total_blinks += 1
                blinked = True

            self.blink_counter = 0

        if draw and self.DEBUG:
            result_tuple = (
                looking_away, looking_down, looking_up,
                looking_left, looking_right,
                partial_face,
                yaw_ratio, pitch_ratio, gaze_ratio,
                ear, blinked, self.total_blinks,
            )
            self.draw_debug(frame, result_tuple, show_gaze=show_gaze,
                            show_pose=show_pose, show_liveness=show_liveness)

        return (
            looking_away,
            looking_down,
            looking_up,
            looking_left,
            looking_right,
            partial_face,
            yaw_ratio,
            pitch_ratio,
            gaze_ratio,
            ear,
            blinked,
            self.total_blinks,
        )