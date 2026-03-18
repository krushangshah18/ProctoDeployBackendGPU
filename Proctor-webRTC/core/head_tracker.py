import time
import cv2

# Per-key y-positions so multiple active timers stack vertically without overlap
_TIMER_Y: dict[str, int] = {
    "looking_away" : 230,
    "looking_down" : 255,
    "looking_up"   : 280,
    "looking_side" : 305,
    "face_hidden"  : 330,
    "partial_face" : 355,
    "fake_presence": 380,
}
_TIMER_Y_DEFAULT_START = 230
_TIMER_Y_STEP          = 25


class HeadTracker:
    """Duration-gated state machine for head-pose and gaze conditions.

    Returns True from process() only once the condition has been
    continuously active for at least `threshold` seconds.  Also keeps
    `states[key]["active"]` in sync so callers (e.g. the partial-face
    banner in ProctorSession) can read current state at any time.
    """

    def __init__(self, states: dict, threshold: float, debug: bool = False):
        self.states    = states
        self.threshold = threshold
        self.DEBUG     = debug

    def process(
        self, frame, key: str, condition: bool, threshold: float | None = None
    ) -> tuple[bool, float]:
        """
        Returns:
            (triggered, duration)
            triggered — True once the condition has been active for >= threshold s
            duration  — seconds the condition has been continuously active (0.0 if not)

        Returning duration avoids a second time.time() call in the caller to
        compute the same value from the same start_time.
        """
        triggered         = False
        duration          = 0.0
        now               = time.time()
        this_state        = self.states[key]
        active_threshold  = threshold if threshold is not None else self.threshold

        if condition:
            if this_state["start_time"] is None:
                this_state["start_time"] = now

            duration = now - this_state["start_time"]

            if duration >= active_threshold:
                triggered            = True
                this_state["active"] = True
        else:
            this_state["start_time"] = None
            this_state["active"]     = False

        if self.DEBUG and this_state["start_time"]:
            label = key.replace("_", " ").title()
            y     = _TIMER_Y.get(key, _TIMER_Y_DEFAULT_START)
            cv2.putText(
                frame,
                f"{label}: {duration:.1f}s",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        return triggered, duration
