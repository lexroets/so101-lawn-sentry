"""Object detection: YOLO (on-camera) + classical fallback.

Classical fallback has two modes (config REQUIRE_GREEN_PLANE):
  False (default): DEPTH-BASED — anything standing proud of the desk surface
        counts, whatever colour the desk is. Per image row the desk dominates,
        so pixels closer than the row's median depth by DESK_PROUD_M are
        object pixels.
  True : the old green-plane gating (kept for later).

All candidates are classified by REAL physical size into:
  'target' — graspable object (battery, pen cap, ...)
  'box'    — the drop box

Pass a dict as `stats` to collect per-frame reject counts for debugging.
"""
import warnings

import cv2
import numpy as np

from config import (
    FX_PIXELS,
    BATT_MIN_DIM_M,
    BATT_MAX_DIM_M,
    BOX_WIDTH_M,
    BOX_HEIGHT_M,
    CAP_ASPECT,
    CAP_SOLIDITY_MIN,
    MIN_PLANE_AREA_PX,
    PLANE_EDGE_DILATE_PX,
    REQUIRE_GREEN_PLANE,
    DESK_PROUD_M,
    REACH_NEAR_M,
    REACH_FAR_M,
    REACH_RANGE_MARGIN_M,
    RYOBI_HUE,
    PINK_HUE,
    PINK_HUE2,
    COLOUR_S_MIN,
    PINK_S_MIN,
    PINK_MIN_PX,
    WHITE_S_MAX,
    WHITE_V_MIN,
    WHITE_MIN_FRAC,
    COLOUR_MIN_FRAC,
    DARK_V_MAX,
    DARK_MIN_FRAC,
    NOT_DARK_MAX_FRAC,
    PINK_MIN_FRAC,
    PINK_VS_RYOBI_W,
    LIME_AS_RYOBI,
    DARK_SPLIT_FRAC,
    COLOUR_MIN_EVIDENCE_PX,
    ALLOW_PARTIAL_TARGETS,
    PARTIAL_MIN_AREA_PX,
    COMPUTE_ORIENT_METRICS,
    DETECT_Z_MAX_M,
    YOLO_CONF,
    REACH_NEAR_MARGIN_M,
)

# Reachability window: outside this depth band = unreachable background.
# Hard-capped by DETECT_Z_MAX_M so one far-taught reach point can't open
# the gate to distant background clutter.
_Z_MIN = REACH_NEAR_M - REACH_NEAR_MARGIN_M
_Z_MAX = min(REACH_FAR_M + REACH_RANGE_MARGIN_M, DETECT_Z_MAX_M)

GREEN_LOW = np.array([28, 30, 25], np.uint8)
GREEN_HIGH = np.array([100, 255, 255], np.uint8)


def _bump(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _measure_depth(depth_m, u, v):
    """Median clustered depth (m) in a small window, or None if unreliable."""
    h_img, w_img = depth_m.shape[:2]
    roi = depth_m[
        max(0, v - 7):min(h_img, v + 8),
        max(0, u - 7):min(w_img, u + 8),
    ]
    valid = roi[np.isfinite(roi) & (roi >= 0.22) & (roi <= 1.20)]
    if valid.size < 20:
        return None
    median = float(np.median(valid))
    cluster = valid[np.abs(valid - median) < max(0.025, 0.07 * median)]
    if cluster.size < 12:
        return None
    return float(np.median(cluster))


_BG_STRIDE = 4      # column sampling stride for the desk-depth estimate


def row_background(depth_m):
    """Desk depth for every image row: the median across the whole row.

    Robust because the desk dominates each row — unlike a local patch beside
    the object, which can fall off the table edge and read the floor (that
    bug produced impossible 489mm 'heights').

    Sampled every _BG_STRIDE columns: the desk is flat across a row, so a
    quarter of the pixels give the same median at a quarter of the cost.
    nanmedian over the full frame was one of the most expensive calls in
    the whole loop."""
    d = np.array(depth_m[:, ::_BG_STRIDE], dtype=np.float32)
    d[~(np.isfinite(d) & (d >= 0.22) & (d <= 1.20))] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmedian(d, axis=1)


def top_residual(depth_m, row_bg, x, y, w, h):
    """How far the object's TOP EDGE stands out of the desk, in metres.

    Measured where the signal is strongest (the top quarter of the blob) and
    compared against the desk depth AT THE SAME IMAGE ROWS, so the desk's own
    front-to-back gradient cancels instead of corrupting the reading. That
    gradient is what made every earlier 'proud height' attempt produce
    impossible numbers like 267mm on a 115mm battery.

    Standing battery: its top is far nearer the camera than the desk would be
    at that row  -> large residual (roughly its height, 60-115mm).
    Lying battery:   its top IS the desk plus 23mm of diameter -> small."""
    if row_bg is None:
        return None
    y0 = max(0, y)
    y1 = min(depth_m.shape[0], y + max(2, h // 4))
    x0, x1 = max(0, x), min(depth_m.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return None
    roi = depth_m[y0:y1, x0:x1]
    vals = roi[np.isfinite(roi) & (roi >= 0.12) & (roi <= 1.20)]
    if vals.size < 15:
        return None
    rows = row_bg[y0:y1]
    rows = rows[np.isfinite(rows)]
    if rows.size == 0:
        return None
    resid = float(np.median(rows) - np.median(vals))
    if not -0.02 <= resid <= 0.30:
        return None                     # physically impossible — don't vote
    return resid


def depth_spread(depth_m, x, y, w, h):
    """Spread of depths ACROSS the object itself (p90 - p10), in metres.

    Self-contained orientation cue — no desk reference, so it cannot be
    corrupted by table edges or perspective. An upright battery's top face
    is far nearer the camera than its base (~60-90mm spread); a lying one
    shows a flat top (~10-20mm)."""
    h_img, w_img = depth_m.shape[:2]
    roi = depth_m[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
    vals = roi[np.isfinite(roi) & (roi >= 0.22) & (roi <= 1.20)]
    if vals.size < 25:
        return None
    return float(np.percentile(vals, 90) - np.percentile(vals, 10))


def proud_height(row_bg, v, z):
    """How far the object stands proud of the desk, in metres.

    The camera looks down, so a TALL object's pixels sit at image rows whose
    desk is much farther away than the object itself: upright RB4L30 reads
    ~60-115mm, a lying one only ~20-30mm."""
    if row_bg is None or not (0 <= v < len(row_bg)):
        return None
    bg = row_bg[v]
    if not np.isfinite(bg):
        return None
    proud = float(bg - z)
    if proud < -0.02 or proud > 0.30:
        return None            # nonsense reading — don't let it decide
    return proud


def _classify_size(w_px, h_px, z, partial=False):
    """(kind, width_m, height_m): 'target' = battery-sized in ANY orientation
    (min/max dimension gate), 'box' = drop-box size, None = neither.

    partial=True means the blob is clipped by the frame edge, so its long
    axis is cut short. The LOWER bound on the long dimension is dropped —
    we only insist it isn't too BIG — while the short dimension (the
    battery's thickness, which stays intact when clipped lengthwise) must
    still match. That lets a half-visible battery close to the camera
    register without opening the gate to arbitrary blobs."""
    width_m = w_px * z / FX_PIXELS
    height_m = h_px * z / FX_PIXELS
    small, big = sorted((width_m, height_m))
    big_lo = 0.0 if partial else BATT_MAX_DIM_M[0]
    if (BATT_MIN_DIM_M[0] <= small <= BATT_MIN_DIM_M[1]
            and big_lo <= big <= BATT_MAX_DIM_M[1]):
        return 'target', width_m, height_m
    if (BOX_WIDTH_M[0] <= width_m <= BOX_WIDTH_M[1]
            and BOX_HEIGHT_M[0] <= height_m <= BOX_HEIGHT_M[1]):
        return 'box', width_m, height_m
    return None, width_m, height_m


def classify_colour(image, bbox, with_counts=False):
    """'ryobi' (BLACK body + lime accents), 'pink', or 'unknown'.

    The Ryobi battery is mostly black, so it is classified by DARKNESS
    (fraction of low-brightness pixels) plus any lime accents; the pink
    battery by saturated pink hues. Sampled from the centre of the box
    (20% inset) so desk background can't dominate."""
    x, y, w, h = bbox
    ix, iy = int(w * 0.2), int(h * 0.2)
    roi = image[max(0, y + iy):y + h - iy, max(0, x + ix):x + w - ix]
    if roi.size == 0:
        roi = image[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return ('unknown', 0, 0) if with_counts else 'unknown'
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = float(roi.shape[0] * roi.shape[1])
    # Pastel pink is PALE — its own (low) saturation bar, wide hue bands.
    pink = int(np.count_nonzero(
        (((hue >= PINK_HUE[0]) & (hue <= PINK_HUE[1]))
         | ((hue >= PINK_HUE2[0]) & (hue <= PINK_HUE2[1])))
        & (sat >= PINK_S_MIN) & (val >= 120)))
    # On a GREEN MAT lime is pure background noise, so it is zeroed out and
    # the Ryobi is identified by its black body + white text alone.
    lime = int(np.count_nonzero(
        (hue >= RYOBI_HUE[0]) & (hue <= RYOBI_HUE[1]) & (sat >= COLOUR_S_MIN))
    ) if LIME_AS_RYOBI else 0
    dark = int(np.count_nonzero(val <= DARK_V_MAX))
    # The Ryobi's WHITE text side: bright + unsaturated pixels.
    # WHITE must EXCLUDE pink hues. Pastel pink is pale and weakly saturated,
    # so it was being counted as "white text" = Ryobi evidence, and pink
    # batteries went to the green bucket.
    _pink_hued = (((hue >= PINK_HUE[0]) & (hue <= PINK_HUE[1]))
                  | ((hue >= PINK_HUE2[0]) & (hue <= PINK_HUE2[1])))
    white = int(np.count_nonzero((val >= WHITE_V_MIN) & (sat <= WHITE_S_MAX)
                                 & ~_pink_hued))
    dark_frac = dark / total
    white_frac = white / total
    # NOTHING THERE? Say so. "Not dark" used to mean "pink", so an empty
    # patch of mat returned pink with dark=0 pink=0 white=0 and the robot
    # drove off to grab nothing. A verdict now needs actual evidence.
    if (dark + pink + white + lime) < COLOUR_MIN_EVIDENCE_PX:
        return ('unknown', 0, 0) if with_counts else 'unknown'
    # DARKNESS DECIDES. Measured in situ: Ryobi 539-2215 dark pixels, pink
    # 6-61 — a 10x gap. Hue was useless (both types read hue 80-107, sat
    # 20-50), so it is now only a tie-breaker for the ambiguous middle band.
    if dark_frac >= DARK_SPLIT_FRAC:
        colour = 'ryobi'                    # black shell
    elif dark_frac <= DARK_SPLIT_FRAC * 0.5:
        colour = 'pink'                     # clearly pale
    elif (pink >= PINK_MIN_PX and pink / total >= PINK_MIN_FRAC
            and pink > PINK_VS_RYOBI_W * (lime + white + dark)):
        colour = 'pink'                     # middle band, real pink evidence
    elif white_frac >= WHITE_MIN_FRAC:
        colour = 'ryobi'                    # middle band, white text side
    else:
        colour = 'unknown'
    ryobi_ct = dark + lime + white
    return (colour, ryobi_ct, pink) if with_counts else colour


def candidates_from_yolo(detections, depth_m, frame_w, frame_h,
                         labels=None, stats=None):
    """Convert on-camera YOLO detections into classified candidates.

    The NN sees a 16:9 slice of the sensor; when the display frame is 4:3
    (full sensor), YOLO's normalized y-coords map onto the CENTRED 16:9 band
    of the display — hence the vertical offset below."""
    nn_h = frame_w * 9 // 16
    y_off = max(0, (frame_h - nn_h) // 2)
    row_bg = row_background(depth_m)
    output = []
    for det in detections:
        if det.confidence < YOLO_CONF:
            _bump(stats, 'yolo_conf_rej')
            continue
        x1 = max(0, int(det.xmin * frame_w))
        y1 = max(0, int(det.ymin * nn_h) + y_off)
        x2 = min(frame_w - 1, int(det.xmax * frame_w))
        y2 = min(frame_h - 1, int(det.ymax * nn_h) + y_off)
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 3:
            _bump(stats, 'yolo_tiny_rej')
            continue
        u, v = x1 + w // 2, y1 + h // 2
        z = _measure_depth(depth_m, u, v)
        if z is None:
            _bump(stats, 'yolo_depth_rej')
            continue
        if not _Z_MIN <= z <= _Z_MAX:
            _bump(stats, 'yolo_range_rej')   # unreachable background
            continue
        kind, width_m, height_m = _classify_size(w, h, z)
        if kind is None:
            _bump(stats, 'yolo_size_rej')
            continue
        _bump(stats, 'yolo_accepted')
        label = None
        if labels is not None and 0 <= det.label < len(labels):
            label = labels[det.label]
        output.append({
            'u': u,
            'v': v,
            'z': z,
            'bbox': (x1, y1, w, h),
            'area': float(w * h),
            'width_m': width_m,
            'height_m': height_m,
            'top_m': (top_residual(depth_m, row_bg, x1, y1, w, h)
                      if COMPUTE_ORIENT_METRICS else None),
            'proud_m': None,
            'spread_m': None,
            'kind': kind,
            'source': 'yolo',
            'label': label,
            'conf': float(det.confidence),
        })
    return sorted(output, key=lambda item: item['z'])


def _filter_contours(contours, depth_m, shape, stats, row_bg=None):
    """Shared contour -> candidate filtering for the classical paths."""
    h_img, w_img = shape[:2]
    if stats is not None:
        stats['contours'] = stats.get('contours', 0) + len(contours)
    output = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 20 <= area <= 80000:
            _bump(stats, 'area_rej')
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Clipped by the frame edge? Keep it if it is big enough to be a real
        # object — a battery close to the camera is exactly this case, and
        # rejecting it outright was losing targets in plain view.
        partial = (x < 2 or y < 2 or x + w > w_img - 2 or y + h > h_img - 2)
        if partial and not (ALLOW_PARTIAL_TARGETS
                            and area >= PARTIAL_MIN_AREA_PX):
            _bump(stats, 'edge_rej')
            continue
        aspect = h / float(w)
        if not CAP_ASPECT[0] <= aspect <= CAP_ASPECT[1]:
            _bump(stats, 'aspect_rej')
            continue
        if area / float(w * h) < CAP_SOLIDITY_MIN:
            _bump(stats, 'solidity_rej')
            continue
        u, v = x + w // 2, y + h // 2
        z = _measure_depth(depth_m, u, v)
        if z is None:
            _bump(stats, 'depth_rej')
            continue
        if not _Z_MIN <= z <= _Z_MAX:
            _bump(stats, 'range_rej')        # unreachable background
            continue
        kind, width_m, height_m = _classify_size(w, h, z, partial=partial)
        if kind is None:
            _bump(stats, 'size_rej')
            continue
        if partial:
            _bump(stats, 'partial_ok')
        _bump(stats, 'accepted')
        output.append({
            'u': u,
            'v': v,
            'z': z,
            'bbox': (x, y, w, h),
            'area': float(area),
            'width_m': width_m,
            'height_m': height_m,
            # top_m is the orientation signal. proud/spread are legacy and
            # stay off — they measured the desk, not the object.
            'top_m': (top_residual(depth_m, row_bg, x, y, w, h)
                      if COMPUTE_ORIENT_METRICS else None),
            'proud_m': None,
            'spread_m': None,
            'partial': partial,
            'kind': kind,
            'source': 'hsv' if REQUIRE_GREEN_PLANE else 'depth',
            'label': None,
            'conf': None,
        })
    return sorted(output, key=lambda item: item['z'])


def _mask_depth_objects(depth_m):
    """Objects proud of the desk, from depth alone — works on any desk."""
    d = depth_m.copy()
    valid = np.isfinite(d) & (d >= 0.22) & (d <= 1.20)
    d[~valid] = np.nan
    row_bg = row_background(depth_m)        # the desk dominates each row
    with np.errstate(invalid='ignore'):
        proud = (row_bg[:, None] - d) > DESK_PROUD_M
    mask = (proud & valid).astype(np.uint8) * 255
    # 3x3 open kills speckle noise (2x2 let it through and blobs sprawled),
    # then a generous 13x13 close knits a lying battery's fragments into one.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    return mask


def _mask_green_plane(image, stats):
    """The old path: non-green blobs on the largest green region."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    plane_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    if not plane_contours:
        _bump(stats, 'no_plane')
        return None
    plane = max(plane_contours, key=cv2.contourArea)
    if cv2.contourArea(plane) < MIN_PLANE_AREA_PX:
        _bump(stats, 'no_plane')
        return None
    h_img, w_img = image.shape[:2]
    plane_mask = np.zeros((h_img, w_img), np.uint8)
    cv2.drawContours(plane_mask, [plane], -1, 255, -1)
    k = np.ones((PLANE_EDGE_DILATE_PX, PLANE_EDGE_DILATE_PX), np.uint8)
    plane_mask = cv2.dilate(plane_mask, k)
    objects = cv2.bitwise_and(cv2.bitwise_not(green), plane_mask)
    objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return objects


def detect_candidates(image, depth_m, stats=None):
    """Classical fallback detector (YOLO-independent)."""
    row_bg = row_background(depth_m)
    if REQUIRE_GREEN_PLANE:
        mask = _mask_green_plane(image, stats)
        if mask is None:
            return []
    else:
        mask = _mask_depth_objects(depth_m)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return _filter_contours(contours, depth_m, image.shape, stats, row_bg)
