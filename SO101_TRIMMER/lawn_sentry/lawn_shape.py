"""Find the lawn automatically: segment the green area, project it to metres.

Instead of clicking four corners of an assumed rectangle, this looks at what is
actually in front of the camera, decides which of it is grass, and returns that
region as a real-world polygon. The dashboard then draws the true lawn shape
rather than a rectangle.

    mask   = green_mask(bgr)                  # which pixels are grass
    poly   = ground_polygon(mask, H)          # [[x_m, y_m], ...] on the ground
    smooth = LawnShape().update(poly)         # de-jittered across frames

SCALE STILL COMES FROM THE HOMOGRAPHY. You cannot get metres out of a colour
blob alone — something has to fix the scale. So run calibrate_lawn_map.py once
(any known rectangle on the grass) and after that the SHAPE is automatic: the
lawn can be any outline, and it no longer has to match what you clicked.

Pure functions, no camera or depthai imports, so the geometry is unit-testable.
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
import numpy as np
import cv2

# --- what counts as grass (OpenCV HSV: H 0-179, S 0-255, V 0-255) ----------
GRASS_H_LO, GRASS_H_HI = 30, 95     # green through yellow-green
GRASS_S_MIN = 35                    # must have some colour (excludes grey path)
GRASS_V_MIN = 25                    # excludes deep shadow
GRASS_V_MAX = 250                   # excludes blown-out highlights

MIN_AREA_FRAC = 0.02                # ignore green specks < 2% of the frame
SIMPLIFY_PX = 6.0                   # contour simplification tolerance
MAX_RANGE_M = 6.0                   # clip the polygon: beyond this the ground
                                    # plane projection is meaningless (horizon)
MAX_POINTS = 40                     # keep the polygon light for JSON/web


def green_mask(bgr, close_px=9, open_px=5):
    """Boolean mask of the grass region — largest green blob, holes filled."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= GRASS_H_LO) & (h <= GRASS_H_HI) & (s >= GRASS_S_MIN)
         & (v >= GRASS_V_MIN) & (v <= GRASS_V_MAX)).astype(np.uint8) * 255
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             np.ones((open_px, open_px), np.uint8))
    if close_px:
        # close gaps caused by poop, shadows, toys sitting on the grass
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             np.ones((close_px, close_px), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    # biggest component that isn't the background
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < MIN_AREA_FRAC * m.size:
        return np.zeros(m.shape, bool)
    keep = (lbl == idx).astype(np.uint8) * 255
    # fill interior holes so objects ON the lawn don't punch through it
    ff = keep.copy()
    pad = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    return (keep | cv2.bitwise_not(ff)) > 0


def _clip_forward(pts, max_range=MAX_RANGE_M):
    """Drop polygon points that project absurdly far (at or past the horizon)."""
    out = [p for p in pts if abs(p[0]) <= max_range and -0.5 <= p[1] <= max_range]
    return out


def ground_polygon(mask, H, simplify=SIMPLIFY_PX, max_points=MAX_POINTS):
    """Outline of the grass region projected to ground metres via homography H.

    Returns [[x_m, y_m], ...] (same frame as the detections: x right, y away
    from the camera) or None if nothing usable was found.
    """
    if mask is None or not mask.any():
        return None
    m = mask.astype(np.uint8) * 255
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2).astype(np.float32)
    if len(approx) < 3:
        return None
    proj = cv2.perspectiveTransform(approx.reshape(1, -1, 2),
                                    np.asarray(H, np.float64)).reshape(-1, 2)
    pts = _clip_forward([[float(x), float(y)] for x, y in proj])
    if len(pts) < 3:
        return None
    # thin out evenly if still too many points
    if len(pts) > max_points:
        step = len(pts) / max_points
        pts = [pts[int(i * step)] for i in range(max_points)]
    return pts


def polygon_area(poly):
    """Shoelace area in m^2 (always positive)."""
    if not poly or len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def polygon_bounds(poly):
    """(min_x, min_y, max_x, max_y) — what the dashboard needs to fit the view."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(poly, x, y):
    """True if (x, y) lies inside the lawn outline (ray casting)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def fit_rectangle(points, fixed_size=None):
    """Minimum-area ROTATED rectangle through world points -> 4 corners.

    The lawn really is a rectangle, so rather than tracing a wobbly green
    outline we fit the rectangle that best explains what we can see. Pass
    `fixed_size=(width_m, length_m)` to keep the known dimensions and let only
    the POSITION and ORIENTATION follow the camera — which is what you want
    when the lawn size is known and only the viewing angle changes.
    """
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w < 1e-6 or h < 1e-6:
        return None
    if fixed_size:
        fw, fl = float(fixed_size[0]), float(fixed_size[1])
        # keep the long axis of the fitted rect aligned with the long real side,
        # otherwise the lawn would flip 90 degrees
        if (w >= h) != (fw >= fl):
            fw, fl = fl, fw
        w, h = fw, fl
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    return [[float(x), float(y)] for x, y in box]


class RectLawn:
    """A rectangular lawn whose position/orientation follow the camera.

    Feed it whatever green you can currently see; it reports a smoothed
    rectangle of the known size. Because only centre and angle move, the shape
    never wobbles — it just re-orients as the camera turns.
    """

    def __init__(self, size=None, alpha=0.2):
        self.size = size                 # (width_m, length_m) or None to measure
        self.alpha = alpha
        self.cx = self.cy = None
        self.ang = None                  # degrees
        self.updates = 0

    @staticmethod
    def _wrap(a):
        """Normalise a LONG-AXIS angle to (-90, 90].

        Once the angle refers to the rectangle's long side it is meaningful
        modulo 180 deg, not 90. Wrapping to +/-45 (the square convention) would
        fold a 50 deg rotation to -40 deg — i.e. flip the lawn on its side.
        """
        a = (a + 90.0) % 180.0 - 90.0
        return a + 180.0 if a <= -90.0 else a

    def update(self, points):
        pts = np.asarray(points, np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return self.rect()
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        if w < 1e-6 or h < 1e-6:
            return self.rect()
        if self.size is None:
            self.size = (max(w, h), min(w, h))
        # express the angle relative to the rect's long axis so it is stable
        if w < h:
            ang += 90.0
        ang = self._wrap(ang)
        if self.cx is None:
            self.cx, self.cy, self.ang = cx, cy, ang
        else:
            a = self.alpha
            self.cx = (1 - a) * self.cx + a * cx
            self.cy = (1 - a) * self.cy + a * cy
            d = self._wrap(ang - self.ang)          # shortest angular step
            self.ang = self._wrap(self.ang + a * d)
        self.updates += 1
        return self.rect()

    def rect(self):
        """Current 4 corners in world metres, or None before the first update."""
        if self.cx is None or self.size is None:
            return None
        long_side, short_side = max(self.size), min(self.size)
        box = cv2.boxPoints(((self.cx, self.cy), (long_side, short_side), self.ang))
        return [[float(x), float(y)] for x, y in box]

    def area(self):
        return float(self.size[0] * self.size[1]) if self.size else 0.0


class LawnMap:
    """Accumulate the lawn in WORLD space as the camera looks around.

    A single frame only shows the grass currently in view, so the outline would
    shrink and grow as the camera pans. Instead we mark what we have seen into
    a world-space occupancy grid and report the outline of everything observed
    so far. That gives a stable shape that grows to the true lawn, works with a
    moving camera, and handles concave/irregular outlines.

        lm = LawnMap()
        lm.add(green_mask(frame), W)      # W = current image->world homography
        poly = lm.polygon()               # metres, or None
    """

    def __init__(self, cell=0.05, x_range=(-4.0, 4.0), y_range=(-0.5, 7.0)):
        self.cell = cell
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.nx = max(8, int((self.x1 - self.x0) / cell))
        self.ny = max(8, int((self.y1 - self.y0) / cell))
        self.grid = np.zeros((self.ny, self.nx), np.uint8)
        self.frames = 0

    def _to_cell(self, x, y):
        return (int((x - self.x0) / self.cell), int((y - self.y0) / self.cell))

    def _to_world(self, cx, cy):
        return (self.x0 + (cx + 0.5) * self.cell, self.y0 + (cy + 0.5) * self.cell)

    def add(self, mask, H):
        """Project the grass region through H and mark it as seen."""
        if mask is None or not mask.any():
            return
        m = mask.astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return
        c = max(cs, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 4.0, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            return
        proj = cv2.perspectiveTransform(approx.reshape(1, -1, 2),
                                        np.asarray(H, np.float64)).reshape(-1, 2)
        pts = [self._to_cell(x, y) for x, y in proj
               if np.isfinite(x) and np.isfinite(y)
               and self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1]
        if len(pts) < 3:
            return
        cv2.fillPoly(self.grid, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], 255)
        self.frames += 1

    def polygon(self, simplify_cells=1.5, max_points=MAX_POINTS):
        """Outline of everything seen so far, in metres."""
        if not self.grid.any():
            return None
        g = cv2.morphologyEx(self.grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < 4:
            return None
        approx = cv2.approxPolyDP(c, simplify_cells, True).reshape(-1, 2)
        pts = [list(self._to_world(int(p[0]), int(p[1]))) for p in approx]
        if len(pts) < 3:
            return None
        if len(pts) > max_points:
            step = len(pts) / max_points
            pts = [pts[int(i * step)] for i in range(max_points)]
        return pts

    def area(self):
        """Observed lawn area in m^2 (grid cells actually seen)."""
        return float((self.grid > 0).sum()) * self.cell * self.cell


class LawnShape:
    """Temporally smoothed lawn outline.

    A per-frame contour jitters as grass moves and light changes, which would
    make the dashboard shape wobble. We only accept a new outline when it is
    broadly consistent with the running one, and blend the reported area.
    """

    def __init__(self, alpha=0.25, max_area_jump=0.45):
        self.poly = None
        self.area = 0.0
        self.alpha = alpha
        self.max_area_jump = max_area_jump
        self.updates = 0

    def update(self, poly):
        if not poly or len(poly) < 3:
            return self.poly
        a = polygon_area(poly)
        if a <= 0.05:
            return self.poly
        if self.poly is None:
            self.poly, self.area, self.updates = poly, a, 1
            return self.poly
        # reject a wild jump (someone walked through frame, light flickered)
        if abs(a - self.area) / max(self.area, 1e-6) > self.max_area_jump:
            return self.poly
        self.area = (1 - self.alpha) * self.area + self.alpha * a
        self.poly = poly              # outline follows, area is smoothed
        self.updates += 1
        return self.poly
