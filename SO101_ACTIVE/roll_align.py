"""Auto wrist-roll alignment for LYING batteries.

A lying battery's long axis is just the principal axis of its silhouette, so
no learned model is needed — cv2.minAreaRect on the object's depth-proud
pixels gives the heading directly. We convert that image-plane heading to a
wrist-roll servo value via a two-point calibration (calibrate_roll.py) and
override ONLY the roll joint of the taught top-down grab, so the fingers
straddle the battery across its body at any heading.

Everything here is pure computation — it never opens the serial port. The
angle can be displayed and verified on camera before a motor ever moves.
"""
import math
import os
import numpy as np

try:
    from detector import _Z_MIN as _CFG_ZMIN, _Z_MAX as _CFG_ZMAX
except Exception:                       # be defensive if internals move
    _CFG_ZMIN, _CFG_ZMAX = 0.10, 0.60

# Written by calibrate_roll.py; absent until then.
ROLL_REF_RAW = None      # roll servo raw at ROLL_REF_DEG
ROLL_REF_DEG = 0.0       # battery heading (deg) that ROLL_REF_RAW straddles
ROLL_RAW_PER_DEG = None  # servo raw per degree of battery rotation
ROLL_MIN_RAW = 0
ROLL_MAX_RAW = 4095
try:
    from roll_calibration import *      # noqa: F401,F403
except ImportError:
    pass


def battery_axis_deg(depth_m, bbox, pad_frac=0.25):
    """Heading of a lying battery's long axis, in degrees, or None.

    0 deg = horizontal in the image, increasing counter-clockwise, folded to
    [0, 180) because a battery has no head/tail. Returns (deg, elongation);
    elongation < ~2 means the silhouette isn't clearly elongated (probably
    upright or a bad mask) and the heading should not be trusted."""
    h_img, w_img = depth_m.shape[:2]
    x, y, w, h = bbox
    px, py = int(w * pad_frac), int(h * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w_img, x + w + px), min(h_img, y + h + py)
    roi = depth_m[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    valid = np.isfinite(roi) & (roi >= _CFG_ZMIN) & (roi <= _CFG_ZMAX)
    if valid.sum() < 40:
        return None
    # Object = pixels nearer than the ROI's far background (the desk).
    bg = np.nanmedian(np.where(valid, roi, np.nan))
    obj = valid & (roi < bg - 0.006)
    ys, xs = np.nonzero(obj)
    if xs.size < 40:
        return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    major = evecs[:, -1]
    elong = math.sqrt(max(evals[-1], 1e-6) / max(evals[0], 1e-6))
    deg = math.degrees(math.atan2(-major[1], major[0]))  # image y is down
    deg %= 180.0
    return deg, elong


def _wrap_to_range(raw):
    """Pick the representative of `raw` (mod one 180-deg servo turn) that
    lands inside the servo's travel. A battery is 180-deg symmetric, so
    rotating the wrist by a half-turn straddles it identically."""
    if ROLL_RAW_PER_DEG is None:
        return raw
    half_turn = abs(ROLL_RAW_PER_DEG) * 180.0
    while raw < ROLL_MIN_RAW:
        raw += half_turn
    while raw > ROLL_MAX_RAW:
        raw -= half_turn
    return int(round(max(ROLL_MIN_RAW, min(ROLL_MAX_RAW, raw))))


def roll_for_heading(deg):
    """Servo raw that straddles a battery at image heading `deg`, or None if
    roll calibration is missing."""
    if ROLL_REF_RAW is None or ROLL_RAW_PER_DEG is None or deg is None:
        return None
    return _wrap_to_range(ROLL_REF_RAW + ROLL_RAW_PER_DEG * (deg - ROLL_REF_DEG))


def is_calibrated():
    return ROLL_REF_RAW is not None and ROLL_RAW_PER_DEG is not None


def fit(samples):
    """Least-squares fit (raw = a + b*deg) from [(deg, raw), ...].

    Unwraps headings so a battery near 0/180 doesn't split the fit."""
    if len(samples) < 2:
        return None
    degs = np.array([s[0] for s in samples], float)
    raws = np.array([s[1] for s in samples], float)
    base = degs[0]
    degs = base + ((degs - base + 90.0) % 180.0) - 90.0   # unwrap near base
    b, a = np.polyfit(degs, raws, 1)
    return {'ROLL_REF_DEG': float(base),
            'ROLL_REF_RAW': int(round(a + b * base)),
            'ROLL_RAW_PER_DEG': float(b)}
