"""Know where the camera is pointing from the arm's encoder, not from guesswork.

`GroundTracker` estimates camera motion by comparing consecutive frames. That
works, but every frame's small error is added to the last, so the estimate
drifts: measured on this rig, a fixed ground point is displaced by 8 px after
5 degrees of pan and 537 px after 45. Nothing in the pipeline can see that
happening, because a slow shift distorts nothing.

The SO-101 already knows its base angle. An encoder reading is ABSOLUTE, so the
error after 45 degrees is the same as after 5 - nothing is chained, so nothing
accumulates. For a camera rotating about its mount, the image-to-image
homography is then exact arithmetic:

    G = K R K^-1        (verified to 0.000 px out to 80 degrees of pan)

Two things are needed and both come for free:

* K, the camera matrix - the OAK-D carries a factory calibration.
* the rotation axis in camera coordinates. The base turns about the vertical,
  and on level ground the vertical IS the ground-plane normal - which is
  already encoded in the lawn calibration. So it is recovered from H0 rather
  than measured or guessed.

Drop-in for GroundTracker: same `.G`, `.W`, `.update()`, `.status()`,
`.anchor()`. If the servo cannot be read it falls back to the visual tracker,
so a disconnected arm degrades to yesterday's behaviour instead of failing.
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

from camera_intrinsics import rotation_homography_axis


def ground_normal_from_homography(H, K):
    """The ground-plane normal in camera coordinates, from the calibration.

    A ground homography is H^-1 = K [r1 r2 t] up to scale, where r1, r2 span
    the plane. Their cross product is the plane normal - and on level ground
    that is also the vertical the base turns about, which is exactly the axis
    the rotation needs.
    """
    try:
        M = np.linalg.inv(K) @ np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return None
    s1 = np.linalg.norm(M[:, 0])
    s2 = np.linalg.norm(M[:, 1])
    if not (np.isfinite(s1) and np.isfinite(s2)) or min(s1, s2) < 1e-12:
        return None
    lam = (s1 + s2) / 2.0
    r1 = M[:, 0] / lam
    r2 = M[:, 1] / lam
    n = np.cross(r1, r2)
    ln = np.linalg.norm(n)
    if ln < 1e-9 or not np.isfinite(ln):
        return None
    n = n / ln
    # point it "up" in camera terms (camera y is down, so up is -y)
    if n[1] > 0:
        n = -n
    return n


class EncoderTracker:
    """Camera pose from the base encoder. Drop-in for GroundTracker."""

    # How many degrees of movement to observe before trusting the sign we
    # inferred. Small motions are where visual tracking is reliable, so it is
    # used once, to answer one yes/no question, and then not relied on again.
    SIGN_LEARN_DEG = 2.5

    def __init__(self, H0, K, pan, roi_poly=None, sign=None, visual=None):
        self.H0 = np.asarray(H0, np.float64)
        self.K = np.asarray(K, np.float64)
        self.pan = pan
        self.G = np.eye(3)
        self.W = self.H0.copy()
        self.axis = ground_normal_from_homography(self.H0, self.K)
        self.ref = None
        self.sign = sign                 # None -> work it out from the first pan
        self.visual = visual             # optional GroundTracker, for the sign
        self.angle = None
        self.moved_deg = 0.0
        self.fails = 0
        self.fallback = False
        if self.axis is None:
            self.axis = np.array([0.0, -1.0, 0.0])
            self.axis_ok = False
        else:
            self.axis_ok = True

    # -- helpers -------------------------------------------------------------
    def _G_for(self, delta_deg, sign):
        return rotation_homography_axis(self.K, sign * float(delta_deg), self.axis)

    def _learn_sign(self, frame, delta):
        """Pick the sign whose prediction matches what the image actually did.

        Visual tracking is accurate over a couple of degrees; it is only wrong
        over long sweeps. So it is a perfectly good judge of direction, used
        once, and then set aside.
        """
        if self.visual is None or frame is None:
            return -1.0
        try:
            Gv = np.asarray(self.visual.G, float)
        except Exception:
            return -1.0
        h, w = frame.shape[:2]
        pts = np.array([[[w * .25, h * .6]], [[w * .75, h * .6]],
                        [[w * .5, h * .85]]], np.float32)
        try:
            ref = cv2.perspectiveTransform(pts, Gv.astype(np.float32))
        except cv2.error:
            return -1.0
        best, bd = -1.0, None
        for s in (1.0, -1.0):
            try:
                q = cv2.perspectiveTransform(
                    pts, self._G_for(delta, s).astype(np.float32))
            except cv2.error:
                continue
            d = float(np.linalg.norm(q - ref))
            if not np.isfinite(d):
                continue
            if bd is None or d < bd:
                best, bd = s, d
        return best

    # -- GroundTracker interface --------------------------------------------
    def update(self, frame):
        a = self.pan.angle_deg() if self.pan is not None else None
        if a is None:
            self.fails += 1
            if self.fails > 15 and self.visual is not None:
                self.fallback = True
            if self.fallback and self.visual is not None:
                self.W = self.visual.update(frame)
                self.G = self.visual.G
                return self.W
            return self.W
        self.fails = 0
        self.angle = a
        if self.ref is None:
            self.ref = a
        delta = a - self.ref
        self.moved_deg = abs(delta)

        if self.visual is not None and not self.fallback:
            # keep it running only while we still need it for the sign
            if self.sign is None:
                self.visual.update(frame)

        if self.sign is None:
            if abs(delta) >= self.SIGN_LEARN_DEG:
                self.sign = self._learn_sign(frame, delta)
                print(f'[encoder] pan direction resolved (sign {self.sign:+.0f}); '
                      f'visual tracking no longer used')
            else:
                s = -1.0            # provisional; barely moved, so barely matters
                self.G = self._G_for(delta, s)
                self.W = self.H0 @ self.G
                self.W /= self.W[2, 2]
                return self.W

        self.G = self._G_for(delta, self.sign)
        self.W = self.H0 @ self.G
        self.W /= self.W[2, 2]
        return self.W

    def status(self):
        return {'lost': self.pan is None or self.fails > 5,
                'tracked': 0 if self.angle is None else 999,
                'moved_m': 0.0,
                'source': 'visual (fallback)' if self.fallback else 'encoder',
                'angle_deg': self.angle,
                'moved_deg': self.moved_deg}

    def anchor(self, H_known=None):
        """Treat the current view as the reference again."""
        if H_known is not None:
            self.H0 = np.asarray(H_known, np.float64)
        self.ref = self.pan.angle_deg() if self.pan is not None else None
        self.G = np.eye(3)
        self.W = self.H0.copy()

    def correct(self, *a, **k):
        return False

    def to_world(self, u, v):
        p = cv2.perspectiveTransform(
            np.array([[[float(u), float(v)]]], np.float32),
            self.W.astype(np.float32))[0, 0]
        return float(p[0]), float(p[1])
