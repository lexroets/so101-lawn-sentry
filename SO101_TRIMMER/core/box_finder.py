"""Find a box on the table and work out its top face and its vertical edges.

This exists to show the two cuts a strimmer actually has to make. Mowing the
TOP of a bed is one problem; edging the SIDE of it is a different one, and a
machine that can only do the first is half a product. A box is the honest test
object for that: it has a flat top and four vertical faces, and unlike grass it
has exact dimensions you can check the answer against.

The geometry, in order:

    1. FIT THE TABLE.  Everything is measured relative to it, so it is found
       first, from the points that are NOT part of the box. A plane fit rather
       than a single depth reading, because the table recedes across the frame
       and one number would put the box's height out by centimetres.

    2. FIND THE BOX.  Whatever stands proud of that plane, largest blob wins.

    3. FIT A RECTANGLE to the top face. minAreaRect, not a bounding box - the
       box will not be square to the camera and the whole point is to follow
       its real edges.

    4. DROP THE CORNERS.  Each top corner is a 3-D point; the matching base
       corner is that point projected onto the table plane along the plane's
       normal. That gives four genuinely vertical edges rather than lines
       guessed in the image.

Everything comes back in metres, in camera coordinates, so it can be handed to
the arm rather than only drawn.
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
import cv2
import numpy as np

MIN_BOX_PROUD_M = 0.015     # below this it is table noise, not a box
MIN_BOX_AREA_PX = 900
PLANE_TOL_M = 0.006         # how close to the fit a point must be to be "table"


def unproject(u, v, z, K):
    """Pixel + depth -> a 3-D point in camera coordinates, metres."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], float)


def project(P, K):
    """3-D point -> pixel."""
    if P[2] <= 1e-6:
        return None
    return (float(K[0, 0] * P[0] / P[2] + K[0, 2]),
            float(K[1, 1] * P[1] / P[2] + K[1, 2]))


def fit_table(depth, K, z_min, z_max, stride=6):
    """Least-squares plane through the table. Returns (normal, point) or None.

    Fitted twice: once over everything in range, then again over only the
    points that landed close to that first plane. The box is a minority of the
    frame but it is a long way off the table, and a single pass lets it drag
    the fit toward itself.
    """
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth[::stride, ::stride]
    ok = np.isfinite(zs) & (zs > z_min) & (zs < z_max)
    if ok.sum() < 200:
        return None
    P = np.stack([unproject(u, v, z, K) for u, v, z
                  in zip(us[ok].ravel(), vs[ok].ravel(), zs[ok].ravel())])

    def _fit(pts):
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[2]
        return (n / np.linalg.norm(n)), c

    n, c = _fit(P)
    d = np.abs((P - c) @ n)
    keep = d < max(PLANE_TOL_M * 4, np.median(d))
    if keep.sum() > 200:
        n, c = _fit(P[keep])
    if n[2] > 0:                      # point the normal back at the camera
        n = -n
    return n, c


def height_above(P, n, c):
    """How far a point stands off the plane, in metres."""
    return float(abs((P - c) @ n))


def find_box(depth, K, table, z_min, z_max,
             min_proud=MIN_BOX_PROUD_M, min_area=MIN_BOX_AREA_PX):
    """The box's top face and its footprint on the table.

    Returns a dict with everything in metres and pixels, or None.
    """
    n, c = table
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    valid = np.isfinite(depth) & (depth > z_min) & (depth < z_max)
    if valid.sum() < 500:
        return None

    # height of every valid pixel above the table plane
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fy
    hh = np.abs((X - c[0]) * n[0] + (Y - c[1]) * n[1] + (depth - c[2]) * n[2])
    hh[~valid] = 0.0

    mask = (hh > min_proud).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [k for k in cnts if cv2.contourArea(k) >= min_area]
    if not cnts:
        return None
    blob = max(cnts, key=cv2.contourArea)

    # The TOP face only: the upper part of the height range inside the blob.
    # Taking the whole blob would include the visible side wall and the
    # rectangle would come out too long in the direction facing the camera.
    bm = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(bm, [blob], -1, 255, -1)
    inside = hh[(bm > 0) & valid]
    if inside.size < 50:
        return None
    top_h = float(np.percentile(inside, 92))
    # A BAND around the top plane, not everything above a fraction of it.
    # "Above 80% of the height" also catches the visible side wall wherever it
    # is near the top, and on a box viewed at an angle that wall is a wide
    # stripe - it made a 10 cm box measure 14 cm. The top face is flat and
    # parallel to the table, so selecting a slice at that height excludes the
    # walls except for a sliver right at the edge, which is where the edge
    # actually is.
    band = max(0.004, 0.12 * top_h)
    top_mask = (((np.abs(hh - top_h) < band) & (bm > 0) & valid) * 255
                ).astype(np.uint8)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
    tc_, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tc_ = [k for k in tc_ if cv2.contourArea(k) >= min_area * 0.4]
    face = max(tc_, key=cv2.contourArea) if tc_ else blob

    # Fit a QUADRILATERAL, not a rectangle.
    #
    # minAreaRect was the obvious choice and it is wrong here: a rectangle in
    # 3-D seen in perspective is a trapezoid in the image, not a rectangle, so
    # the minimum enclosing rectangle has to overshoot to contain it. Measured
    # against ground truth, three corners landed within a pixel or two and the
    # far one was 45 px out - which inflated the box by about 16%.
    #
    # A four-point polygon can represent the trapezoid exactly. The epsilon is
    # walked up until the approximation collapses to four corners.
    rect = cv2.minAreaRect(face)
    hull = cv2.convexHull(face)
    peri = cv2.arcLength(hull, True)
    corners_px = None
    for eps in np.linspace(0.01, 0.12, 24):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            corners_px = ap.reshape(4, 2).astype(float)
            break
    if corners_px is None:
        corners_px = cv2.boxPoints(rect)      # fall back if it never resolves

    # Put each corner on the TOP PLANE by intersecting its ray with that
    # plane - do not read a depth at the corner.
    #
    # Reading one was the obvious thing and it was wrong: a corner sits exactly
    # on a depth discontinuity, so any patch around it is half box and half
    # table, and the median lands somewhere between. Unprojecting at that
    # too-far depth pushes the corner outward along its ray, and the box
    # measures bigger than it is - 10 cm reading as 14. Snapping the height
    # afterwards does not help, because the error is sideways, not vertical.
    #
    # The top plane is known exactly: parallel to the table, top_h above it.
    # Intersecting the ray with it uses no depth data at the corner at all.
    Ki = np.linalg.inv(np.asarray(K, float))
    plane_pt = c + n * top_h
    top3d, base3d, base_px = [], [], []
    for (u, v) in corners_px:
        ray = Ki @ np.array([float(u), float(v), 1.0])
        den = float(ray @ n)
        if abs(den) < 1e-9:
            return None
        Pt = ray * (float(plane_pt @ n) / den)
        if Pt[2] <= 0:
            return None
        Pb = Pt - n * top_h                       # straight down to the table
        top3d.append(Pt)
        base3d.append(Pb)
        q = project(Pb, K)
        if q is None:
            return None
        base_px.append(q)

    top3d = np.array(top3d)
    sides = [float(np.linalg.norm(top3d[i] - top3d[(i + 1) % 4])) for i in range(4)]
    return {
        'top_px': corners_px, 'base_px': np.array(base_px, float),
        'top_m': top3d, 'base_m': np.array(base3d, float),
        'height_m': top_h,
        'width_m': float((sides[0] + sides[2]) / 2.0),
        'depth_m': float((sides[1] + sides[3]) / 2.0),
        'area_px': float(cv2.contourArea(face)),
        'centre_px': (float(rect[0][0]), float(rect[0][1])),
        'angle_deg': float(rect[2]),
    }


def _depth_near(depth, u, v, valid, win=6):
    """Median depth around a pixel, ignoring holes.

    Corners are the worst place to sample a stereo depth map - they sit on a
    discontinuity - so this looks at a patch and takes the median rather than
    trusting one pixel.
    """
    h, w = depth.shape
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    roi = depth[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    return float(np.median(roi)) if roi.size >= 6 else None
