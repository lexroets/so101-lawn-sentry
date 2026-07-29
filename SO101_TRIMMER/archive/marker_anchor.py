"""The marker as a permanent ground truth, not just a calibration step.

Everything that has gone wrong with this system has the same root: the ground
plane is *estimated*, and estimates drift. Visual tracking dead-reckons and
slides off the lawn. Encoder geometry ignores the arc the camera swings
through. Clicked corners are only true for the pose they were clicked at.

A marker left lying on the lawn removes the estimation entirely. Its four
corners are a square of known size at a known place, so whenever the camera
sees it the ground plane can be computed outright - four points, one exact
solution, no history, no accumulation. Look away for a minute, come back, and
it is right again immediately.

That is what makes this deployable rather than a one-off setup:

    * install once - place the marker, say where it is, done
    * self-healing - drift is corrected the moment the marker is in frame
    * verifiable - the marker's known size checks the answer independently
    * portable - no arm, no encoders, no depth; a fixed camera is enough

The marker sits at a known spot in lawn coordinates. Place it at a corner and
square it to the edges and there is nothing to measure at all.
"""
# --- put the project's folders on sys.path (see paths.py) --------------------
import sys as _sys, pathlib as _pl
_d = _pl.Path(__file__).resolve().parent
while not (_d / 'paths.py').is_file() and _d != _d.parent:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
import paths as _paths
_paths.bootstrap()
# -----------------------------------------------------------------------------
import math

import cv2
import numpy as np

from marker import MarkerFinder

# Where the marker lives, in lawn metres. Anchoring at a corner means the only
# numbers a fitter has to get right are the lawn's own dimensions.
PLACEMENTS = {
    'near_left':  lambda W, L, s: (s / 2, s / 2),
    'near_right': lambda W, L, s: (W - s / 2, s / 2),
    'far_left':   lambda W, L, s: (s / 2, L - s / 2),
    'far_right':  lambda W, L, s: (W - s / 2, L - s / 2),
    'centre':     lambda W, L, s: (W / 2, L / 2),
}


def marker_lawn_corners(cx, cy, size, yaw_deg=0.0):
    """The marker's four corners in lawn metres.

    Order matches what the detector returns: top-left, top-right,
    bottom-right, bottom-left as printed. With the marker square to the lawn,
    "up" on the print points away from the camera.
    """
    h = size / 2.0
    local = np.array([[-h, +h], [+h, +h], [+h, -h], [-h, -h]], float)
    if abs(yaw_deg) > 1e-6:
        a = math.radians(yaw_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = local @ R.T
    return local + np.array([cx, cy], float)


def homography_from_marker(corners_px, lawn_corners):
    """Ground plane from one marker sighting. Exact, stateless, no drift."""
    src = np.asarray(corners_px, np.float32).reshape(4, 2)
    dst = np.asarray(lawn_corners, np.float32).reshape(4, 2)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return H if np.all(np.isfinite(H)) else None


class MarkerAnchor:
    """Watches for the marker and hands back an exact ground plane.

    Two jobs, and it is worth being clear which is which:

    `solve()` produces a homography from a single sighting - that is the
    calibration, and it is complete on its own.

    `check()` runs during a scan and reports how far the current estimate has
    slipped, so tracking can be corrected against something real instead of
    being trusted indefinitely.
    """

    def __init__(self, size_m, cx, cy, yaw_deg=0.0, marker_id=None):
        self.size = float(size_m)
        self.lawn_corners = marker_lawn_corners(cx, cy, self.size, yaw_deg)
        self.finder = MarkerFinder() if marker_id is None else MarkerFinder(marker_id)
        self.last_seen = None
        self.seen_count = 0
        self.miss_count = 0

    def find(self, frame):
        c = self.finder.find(frame)
        if c is None:
            self.miss_count += 1
            return None
        self.miss_count = 0
        self.seen_count += 1
        self.last_seen = c
        return c

    def solve(self, frame):
        """-> (H, corners_px) or (None, None) if the marker is not visible."""
        c = self.find(frame)
        if c is None:
            return None, None
        H = homography_from_marker(c, self.lawn_corners)
        return H, c

    def check(self, frame, H_current):
        """How far the current plane has slipped, in metres.

        Returns (error_m, H_correct, corners) - error_m is None when the marker
        is not in view, which is not a failure, just no information.
        """
        c = self.find(frame)
        if c is None:
            return None, None, None
        H_true = homography_from_marker(c, self.lawn_corners)
        if H_true is None or H_current is None:
            return None, H_true, c
        pts = np.asarray(c, np.float32).reshape(1, 4, 2)
        now = cv2.perspectiveTransform(pts, np.asarray(H_current, np.float64)).reshape(4, 2)
        err = float(np.mean(np.linalg.norm(now - self.lawn_corners, axis=1)))
        return err, H_true, c

    def measure(self, H, corners):
        """The marker's size as this plane sees it - an independent check.

        The size never enters the solve, so if a 15 cm square measures 15 cm
        the plane is right. Nothing else in this system can say that about
        itself.
        """
        if H is None or corners is None:
            return None, None
        w = cv2.perspectiveTransform(
            np.asarray(corners, np.float32).reshape(1, 4, 2),
            np.asarray(H, np.float64)).reshape(4, 2)
        sides = [float(np.linalg.norm(w[i] - w[(i + 1) % 4])) for i in range(4)]
        mean = float(np.mean(sides))
        return mean, (mean - self.size) * 100.0
