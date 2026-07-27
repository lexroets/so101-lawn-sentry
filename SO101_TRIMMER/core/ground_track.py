"""Keep world coordinates correct while the CAMERA MOVES.

The problem
-----------
`lawn_calibration.json` gives one homography H0 that maps image pixels to lawn
metres. It is only true for the pose the camera had during calibration. Rotate
or drive the camera and that same pixel now looks at different grass, so a
stationary poop appears to leap across the map and phantom targets appear.

The fix
-------
The ground is flat, so the mapping between two views of it is EXACTLY a
homography. Track features on the ground from frame to frame, estimate that
homography, and chain it:

    G_t  = G_{t-1} @ inv(H_step)        # image_t  -> image_0
    W_t  = H0 @ G_t                     # image_t  -> world metres

Then every detection is projected with W_t instead of H0, and a poop that
hasn't moved keeps the same world coordinate no matter where the camera goes.

This is the same maths a mower will need: it is pure visual odometry on the
ground plane, so it works with no wheel encoders and no mower API.

Usage
-----
    trk = GroundTracker(H0)
    ...
    W = trk.update(frame)           # current image->world homography
    x, y = trk.to_world(u, v)       # project a detection

Limitations, honestly
---------------------
* Drift accumulates — small errors compound over many frames. Fine for a pass
  across a lawn; for long runs you need re-anchoring (see `anchor()`).
* Needs texture. Uniform turf with no features will fail; `lost` goes True and
  the last good mapping is held rather than producing garbage.
* Assumes the ground is planar. True for a lawn, not for a steep bank.
"""

# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import math

import numpy as np
import cv2

# feature tracking parameters — tuned for grass texture
MAX_FEATURES = 400
FEATURE_QUALITY = 0.01
MIN_FEATURE_DIST = 12
MIN_TRACKED = 25          # below this we cannot trust a homography
RANSAC_PX = 3.0
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def estimate_step(prev_gray, gray, mask=None):
    """Homography mapping points in prev_gray -> gray, or None if unreliable."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, MAX_FEATURES, FEATURE_QUALITY,
                                 MIN_FEATURE_DIST, mask=mask)
    if p0 is None or len(p0) < MIN_TRACKED:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
    if p1 is None:
        return None, 0
    st = st.reshape(-1).astype(bool)
    a, b = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
    if len(a) < MIN_TRACKED:
        return None, len(a)
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return None, len(a)
    n_in = int(inl.sum()) if inl is not None else 0
    if n_in < MIN_TRACKED:
        return None, n_in
    return H, n_in


class GroundTracker:
    """Maintains image->world homography as the camera moves."""

    def __init__(self, H0, roi_poly=None):
        self.H0 = np.asarray(H0, np.float64)      # image_0 -> world
        self.G = np.eye(3, dtype=np.float64)      # image_t -> image_0
        self.W = self.H0.copy()                   # image_t -> world
        self.prev = None
        self.lost = False
        self.frames = 0
        self.tracked = 0
        self.lost_frames = 0
        self._mask_poly = np.asarray(roi_poly, np.int32) if roi_poly is not None else None

    # -- internals ----------------------------------------------------------
    def _mask_for(self, gray):
        """Track features on the GROUND only.

        Background clutter — shelves, walls, machines — is NOT on the ground
        plane, so its motion between frames does not follow the ground
        homography. Including it corrupts the estimate, and the error grows the
        more the camera rotates.

        CRUCIALLY the mask must FOLLOW THE GROUND as the camera turns. A mask
        fixed in image coordinates drifts off the grass during rotation and
        starts feeding background features into the fit — which is exactly how
        a survey ends up with a 3.4 m far edge and a 6.8 m near edge.
        """
        if self._mask_poly is None:
            return None
        h, w = gray.shape[:2]
        poly = self._mask_poly.astype(np.float32).reshape(1, -1, 2)
        try:
            # G maps image_t -> image_0, so inv(G) brings the original ground
            # region into the CURRENT view.
            poly = cv2.perspectiveTransform(poly, np.linalg.inv(self.G))
        except np.linalg.LinAlgError:
            pass
        pts = poly.reshape(-1, 2)
        if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5:
            return None
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32).reshape(-1, 1, 2)], 255)
        # grow a little: the useful ground extends past the calibrated quad
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
        # if the ground region has largely left the frame, tracking on whatever
        # is left is worse than nothing — fall back to the lower half, which is
        # almost always ground for a forward-looking camera
        if m.mean() < 12:
            m[:] = 0
            m[int(h * 0.35):, :] = 255
        return m

    # -- public -------------------------------------------------------------
    def update(self, frame):
        """Feed a new frame; returns the current image->world homography."""
        gray = _gray(frame)
        self.frames += 1
        if self.prev is None:
            self.prev = gray
            return self.W
        H_step, n = estimate_step(self.prev, gray, self._mask_for(gray))
        self.tracked = n
        if H_step is None:
            # hold the last good mapping rather than emit nonsense
            self.lost = True
            self.lost_frames += 1
        else:
            try:
                self.G = self.G @ np.linalg.inv(H_step)
                self.G /= self.G[2, 2]
                self.W = self.H0 @ self.G
                self.W /= self.W[2, 2]
                self.lost = False
            except np.linalg.LinAlgError:
                self.lost = True
                self.lost_frames += 1
        self.prev = gray
        return self.W

    def to_world(self, u, v):
        """Project an image point to lawn metres using the CURRENT pose."""
        p = cv2.perspectiveTransform(
            np.array([[[float(u), float(v)]]], np.float32), self.W)[0, 0]
        return float(p[0]), float(p[1])

    def motion_since_start(self):
        """Rough camera displacement in metres since the first frame.

        Where the world point currently under the image centre-bottom was at
        the start — a cheap, honest proxy for 'how far have we moved'."""
        try:
            a = cv2.perspectiveTransform(
                np.array([[[320.0, 470.0]]], np.float32), self.H0)[0, 0]
            b = cv2.perspectiveTransform(
                np.array([[[320.0, 470.0]]], np.float32), self.W)[0, 0]
            return float(np.hypot(b[0] - a[0], b[1] - a[1]))
        except Exception:
            return 0.0

    def anchor(self, H_known=None):
        """Re-anchor to kill accumulated drift.

        Call when you know the true mapping again (e.g. the mower is back at a
        known spot, or you re-run a calibration). With no argument it simply
        resets the chain to the original calibration.
        """
        self.H0 = np.asarray(H_known, np.float64) if H_known is not None else self.H0
        self.G = np.eye(3, dtype=np.float64)
        self.W = self.H0.copy()
        self.lost = False

    def correct(self, dx, dy):
        """Nudge the world mapping by (dx, dy) metres.

        Used by LandmarkAnchor to cancel accumulated drift: everything the
        tracker reports shifts by this much, which is what you want when a
        known landmark says 'you are off by 12 cm'.
        """
        T = np.array([[1.0, 0.0, float(dx)],
                      [0.0, 1.0, float(dy)],
                      [0.0, 0.0, 1.0]], np.float64)
        self.W = T @ self.W
        self.W /= self.W[2, 2]
        # keep G consistent so a later anchor() still behaves
        try:
            self.G = np.linalg.inv(self.H0) @ self.W
            self.G /= self.G[2, 2]
        except np.linalg.LinAlgError:
            pass

    def status(self):
        return {'frames': self.frames, 'tracked': self.tracked,
                'lost': self.lost, 'lost_frames': self.lost_frames,
                'moved_m': round(self.motion_since_start(), 3)}


def camera_pose(W, img_w=640, img_h=480, standoff_m=0.25):
    """Where is the CAMERA, in lawn metres, and which way is it looking?

    The lawn is a fixed thing in the world; the camera is what moves. Given the
    current image->world homography we can recover the camera's own ground
    position and heading, which is what the map should show.

    Projects the near and far points of the image centre-line onto the ground:
    the line between them is the viewing direction, and the camera sits just
    behind the nearest visible ground point.

    Returns (x_m, y_m, heading_deg) or None.
    """
    try:
        cx = img_w / 2.0
        pts = np.array([[[cx, img_h * 0.98]], [[cx, img_h * 0.45]]], np.float32)
        pr = cv2.perspectiveTransform(pts, np.asarray(W, np.float64)).reshape(-1, 2)
        near, far = pr[0], pr[1]
        d = far - near
        n = float(np.hypot(d[0], d[1]))
        if not np.isfinite(n) or n < 1e-6:
            return None
        fwd = d / n
        pos = near - fwd * standoff_m
        if not (np.isfinite(pos).all() and abs(pos[0]) < 50 and abs(pos[1]) < 50):
            return None
        heading = math.degrees(math.atan2(fwd[1], fwd[0]))
        return float(pos[0]), float(pos[1]), float(heading)
    except Exception:
        return None


class LandmarkAnchor:
    """Cancel tracking drift using things whose world position we already know.

    Two sources, both free:

    * CONFIRMED POOPS — each one is a surveyed point. When the detector sees a
      poop again, the gap between where the tracker thinks it is and where we
      logged it IS the accumulated drift. Correct by that amount.
    * LAWN BORDER — the outline's centroid is a stable coarse landmark, used
      when no poop is currently in view.

    Corrections are damped and capped, so a mis-association nudges rather than
    teleports the map. Designed for slow motion, which is your case.
    """

    def __init__(self, match_radius_m=0.45, damping=0.35,
                 max_correction_m=0.80, min_interval_frames=3):
        self.match_radius = match_radius_m
        self.damping = damping
        self.max_correction = max_correction_m
        self.min_interval = min_interval_frames
        self.last_frame = -999
        self.corrections = 0
        self.total_correction_m = 0.0
        self.last_residual_m = 0.0
        self.last_source = None
        self.ref_lawn_centroid = None

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _centroid(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def set_lawn_reference(self, poly):
        """Record the lawn outline measured while we still trusted the pose."""
        if poly and len(poly) >= 3 and self.ref_lawn_centroid is None:
            self.ref_lawn_centroid = self._centroid(poly)

    # -- main ---------------------------------------------------------------
    def update(self, tracker, dets, known_targets, frame_n, lawn_poly=None):
        """Match detections to known targets and correct the tracker.

        dets           : current-frame detections (need 'bbox')
        known_targets  : logged targets, dicts with 'x_m','y_m'
        Returns the residual applied in metres (0.0 if no correction).
        """
        if tracker is None or tracker.lost:
            return 0.0
        if frame_n - self.last_frame < self.min_interval:
            return 0.0

        dx = dy = None
        source = None

        # --- 1. poop landmarks (precise) ---
        if dets and known_targets:
            offs = []
            for d in dets:
                x, y, w, h = d['bbox']
                wx, wy = tracker.to_world(x + w / 2.0, y + h)   # ground contact
                best = None
                bd = self.match_radius
                for t in known_targets:
                    dd = math.hypot(t['x_m'] - wx, t['y_m'] - wy)
                    if dd < bd:
                        bd, best = dd, t
                if best is not None:
                    offs.append((best['x_m'] - wx, best['y_m'] - wy))
            if offs:
                # median is robust to one bad association
                dx = float(np.median([o[0] for o in offs]))
                dy = float(np.median([o[1] for o in offs]))
                source = f'{len(offs)} poop landmark(s)'

        # --- 2. lawn border centroid (coarse fallback) ---
        if dx is None and lawn_poly and self.ref_lawn_centroid:
            cx, cy = self._centroid(lawn_poly)
            rx, ry = self.ref_lawn_centroid
            dx, dy = rx - cx, ry - cy
            source = 'lawn border'

        if dx is None:
            return 0.0

        residual = math.hypot(dx, dy)
        self.last_residual_m = residual
        if residual < 0.02:                    # already accurate; leave it alone
            return 0.0
        if residual > self.max_correction:     # implausible -> likely a bad match
            return 0.0

        tracker.correct(dx * self.damping, dy * self.damping)
        self.last_frame = frame_n
        self.corrections += 1
        self.total_correction_m += residual * self.damping
        self.last_source = source
        return residual

    def status(self):
        return {'corrections': self.corrections,
                'last_residual_cm': round(self.last_residual_m * 100, 1),
                'total_corrected_m': round(self.total_correction_m, 3),
                'source': self.last_source}
