import time
from collections import deque

# How far back (seconds) to look for object observations.
# Semantic: "was this object present within the last N seconds?"
OBJECT_WINDOW_S = 1.5

# Per-class vote ratios.
# Fraction of observations in the window that must show the object present
# before it is considered "stably detected".
# Higher = stricter (slower to trigger, fewer false positives).
VOTE_RATIOS: dict[str, float] = {
    "book":       0.65,
    "phone":      0.55,
    "headphones": 0.55,
    "earbud":     0.50,
    "default":    0.55,   # fallback for any unlisted key
}

# Never require fewer than this many votes regardless of how low fps drops.
# Prevents confirming a detection from just 1-2 observations.
MIN_VOTES_FLOOR = 3


class ObjectTemporalTracker:
    """
    Time-based object temporal tracker.

    Replaces the original frame-count window (deque maxlen=N) with a
    wall-clock window so behaviour is consistent at any frame rate.

    Per-class min_votes is computed dynamically from the observed fps at
    each update call, so detection sensitivity stays proportional regardless
    of whether a candidate is streaming at 5fps or 30fps.

    Call site (unchanged return value — still a bool):
        stable = tracker.update(key, present, fps=observed_fps)
    """

    def __init__(
        self,
        window_s: float = OBJECT_WINDOW_S,
        vote_ratios: dict[str, float] | None = None,
        min_votes_floor: int = MIN_VOTES_FLOOR,
    ):
        self.window_s         = window_s
        self._ratios          = vote_ratios if vote_ratios is not None else VOTE_RATIOS
        self._min_votes_floor = min_votes_floor

        # key → deque of (timestamp: float, present: bool)
        self._windows: dict[str, deque] = {}
        # key → number of True entries currently in the window (O(1) lookup)
        self._vote_counts: dict[str, int] = {}

    def update(self, key: str, present: bool, fps: float = 15.0) -> bool:
        """
        Record one observation for `key` and return whether it is stably detected.

        Args:
            key:     Object class name ("phone", "book", etc.)
            present: Whether the object was detected in this frame.
            fps:     Current observed fps for this candidate stream.
                     Used to scale min_votes so sensitivity is fps-independent.
        """
        now = time.time()

        if key not in self._windows:
            self._windows[key]     = deque()
            self._vote_counts[key] = 0

        win = self._windows[key]

        # Append new observation and update running vote count in O(1)
        win.append((now, present))
        if present:
            self._vote_counts[key] += 1

        # Evict observations older than the time window, decrementing count for
        # any True entries that are dropped — still O(k) where k = expired items,
        # but amortised O(1) per call since each entry is added and removed once.
        cutoff = now - self.window_s
        while win and win[0][0] < cutoff:
            _, was_present = win.popleft()
            if was_present:
                self._vote_counts[key] -= 1

        votes = self._vote_counts[key]

        # Dynamic min_votes: scale from fps so 60% presence at 30fps
        # requires the same confidence as 60% presence at 5fps
        ratio     = self._ratios.get(key, self._ratios["default"])
        min_votes = max(self._min_votes_floor, int(fps * self.window_s * ratio))

        return votes >= min_votes

    def reset(self, key: str | None = None) -> None:
        """Clear history for one key or all keys."""
        if key is None:
            self._windows.clear()
            self._vote_counts.clear()
        else:
            self._windows.pop(key, None)
            self._vote_counts.pop(key, None)

    def debug_state(self, key: str, fps: float = 15.0) -> dict:
        """Return internal state for a key — useful for logging/testing."""
        win       = self._windows.get(key, deque())
        votes     = self._vote_counts.get(key, 0)
        ratio     = self._ratios.get(key, self._ratios["default"])
        min_votes = max(self._min_votes_floor, int(fps * self.window_s * ratio))
        return {
            "key":        key,
            "window_s":   self.window_s,
            "samples":    len(win),
            "votes":      votes,
            "min_votes":  min_votes,
            "fps":        fps,
            "stable":     votes >= min_votes,
        }
