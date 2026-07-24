from collections import deque
import time
import numpy as np
from config import *

class SafetySupervisor:
    def __init__(self):
        self.history = deque(maxlen=STABLE_FRAMES)
        self.stop_latched = False
        self.last_frame_time = time.monotonic()
        self.last_target_time = 0.0

    def latch_stop(self, reason):
        self.stop_latched = True
        self.history.clear()
        return False, f"STOP LATCHED: {reason}"

    def observe_frame(self):
        self.last_frame_time = time.monotonic()

    def observe_candidates(self, candidates):
        now = time.monotonic()
        if len(candidates) != 1:
            self.history.clear()
            return False, "requires exactly one candidate"
        target = candidates[0]
        self.history.append(target)
        self.last_target_time = now
        return self.is_stable()

    def is_stable(self):
        if self.stop_latched:
            return False, "stop latched"
        if time.monotonic() - self.last_frame_time > FRAME_TIMEOUT_S:
            return self.latch_stop("camera timeout")
        if time.monotonic() - self.last_target_time > TARGET_TIMEOUT_S:
            self.history.clear()
            return False, "target timeout"
        if len(self.history) < STABLE_FRAMES:
            return False, "building target history"
        data = np.asarray([[t['u'], t['v'], t['z']] for t in self.history], float)
        centre = np.linalg.norm(data[:, :2] - np.median(data[:, :2], axis=0), axis=1)
        if np.max(centre) > MAX_CENTRE_JITTER_PX:
            return False, "pixel jitter"
        if np.ptp(data[:, 2]) > MAX_DEPTH_JITTER_M:
            return False, "depth jitter"
        return True, "stable target"
