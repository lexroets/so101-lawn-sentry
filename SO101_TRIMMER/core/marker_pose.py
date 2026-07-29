"""Ground plane from the marker's POSE, not from four points on their own.

The first version fitted a homography straight through the marker's four
corners. That is exact for the marker and hopeless everywhere else, and the
reason is worth writing down because it is not obvious: a homography has eight
free parameters, and four corners pin all eight. Every parameter is therefore
fitted entirely from a small square, so half a pixel of corner noise is free to
express itself as a large perspective term - which is precisely the term that
decides where the far edge of the lawn lands. Measured on this rig, a 25 cm
marker gave 122 cm of error at the far corners, and a 50 cm one still gave 24.

Knowing the camera matrix K removes four of those eight unknowns outright. The
marker only has to say where the camera is and which way the ground faces, and
solvePnP with SOLVEPNP_IPPE_SQUARE does that from one square directly. Same
marker, same noise, same rig:

    marker    4-point homography    pose + K
     25 cm            122 cm          10 cm
     35 cm             50 cm           8 cm
     50 cm             24 cm           6 cm

Corner noise is also independent frame to frame, so averaging the detections
while the marker sits still costs two seconds and buys most of the rest:

    25 cm marker    1 frame 10 cm    10 frames 3.1 cm    40 frames 1.9 cm
    35 cm marker    1 frame  8 cm    10 frames 2.5 cm    40 frames 1.3 cm

That is what makes a small marker on a large lawn workable at all.
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

# A marker smaller than this in the image cannot be trusted whatever the maths:
# below roughly four pixels per code module the corners are guesswork. The
# detector will often still fire, which is the dangerous part - it reports a
# confident answer built on nothing.
MIN_SIDE_PX = 22
GOOD_SIDE_PX = 40


def object_points(size_m):
    """The marker's own corners, in the order SOLVEPNP_IPPE_SQUARE requires.

    IPPE_SQUARE is not tolerant here: it wants top-left, top-right,
    bottom-right, bottom-left with +y up in the marker's plane, and silently
    returns a plausible but wrong pose for any other winding. That cost an
    afternoon, so it lives in one function.
    """
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


def solve_pose(corners_px, size_m, K, dist=None):
    """Marker pose in camera coordinates -> (R, t) or (None, None).

    corners_px is what the ArUco detector returns: TL, TR, BR, BL as printed.
    """
    src = np.asarray(corners_px, np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points(size_m), src, np.asarray(K, np.float64),
        None if dist is None else np.asarray(dist, np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    if not (np.isfinite(R).all() and np.isfinite(t).all()) or t[2] <= 0:
        return None, None
    return R, t


def ground_from_pose(R, t, K, cx, cy, yaw_deg=0.0):
    """Image pixels -> lawn metres, given where the marker is on the lawn.

    The marker defines the ground plane; (cx, cy) says where on the lawn its
    centre sits, so the plane's coordinates become the lawn's coordinates.
    Returned as a homography so nothing downstream has to change.
    """
    Ki = np.linalg.inv(np.asarray(K, np.float64))
    n = R[:, 2].reshape(3)              # ground normal, in camera coordinates
    d = float(n @ t)
    if abs(d) < 1e-9:
        return None
    # A ray through pixel p meets the plane at (d / n.r) r; expressing that
    # point in the marker's frame and shifting to the lawn origin is linear in
    # homogeneous coordinates, so it collapses to a single 3x3.
    M = R[:, :2].T @ (np.eye(3) - np.outer(t, n) / d)
    H = np.vstack([M @ Ki, (n @ Ki) / d])
    if abs(yaw_deg) > 1e-9:
        a = math.radians(yaw_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0],
                       [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        H = Rz @ H
    T = np.array([[1, 0, float(cx)], [0, 1, float(cy)], [0, 0, 1.0]])
    H = T @ H
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def camera_from_H(H, K):
    """Where the camera is, in lawn metres, read back out of the homography.

    Derived from H rather than from (R, t) on purpose. The first version took
    the marker pose and added the marker's lawn position, which quietly ignored
    any rotation applied afterwards - so once the lawn had been turned to line
    up, the readout was reporting a camera 90 degrees away from where it was.
    Since H is what actually gets saved and used, reading the answer out of H
    means the readout and the calibration cannot disagree.
    """
    try:
        M = np.linalg.inv(np.asarray(K, np.float64)) @ np.linalg.inv(
            np.asarray(H, np.float64))
    except np.linalg.LinAlgError:
        return None
    r1, r2, t = M[:, 0].copy(), M[:, 1].copy(), M[:, 2].copy()
    s = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    if not np.isfinite(s) or s < 1e-12:
        return None
    r1, r2, t = r1 / s, r2 / s, t / s
    Rm = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    C = -Rm.T @ t
    if not np.isfinite(C).all():
        return None
    return float(C[0]), float(C[1]), abs(float(C[2]))


def camera_in_lawn(R, t, cx, cy):
    """Where the camera is from the marker pose alone - no lawn rotation.

    Only correct while the lawn frame is unrotated. Prefer camera_from_H.
    """
    C = -R.T @ t
    return float(C[0] + cx), float(C[1] + cy), float(C[2])


class PoseAverager:
    """Average the marker's corners while it sits still.

    Averaging the CORNERS rather than the poses is deliberate: corner noise is
    zero-mean and independent, so it averages away cleanly, whereas poses live
    on a manifold where a plain mean is not even well defined.
    """

    def __init__(self, want=40, move_tol_px=6.0):
        self.want = int(want)
        self.move_tol = float(move_tol_px)
        self.buf = []

    def add(self, corners_px):
        c = np.asarray(corners_px, np.float64).reshape(4, 2)
        if self.buf:
            # a marker that has been nudged invalidates everything before it
            if np.abs(c - self.buf[-1]).max() > self.move_tol:
                self.buf = []
        self.buf.append(c)
        if len(self.buf) > self.want:
            self.buf.pop(0)
        return len(self.buf)

    def reset(self):
        self.buf = []

    @property
    def ready(self):
        return len(self.buf) >= self.want

    @property
    def progress(self):
        return len(self.buf) / float(self.want)

    def mean(self):
        if not self.buf:
            return None
        return np.mean(np.asarray(self.buf), axis=0)

    def jitter_px(self):
        """How much the corners are wobbling - a live quality readout."""
        if len(self.buf) < 3:
            return None
        a = np.asarray(self.buf)
        return float(np.mean(np.std(a, axis=0)))
