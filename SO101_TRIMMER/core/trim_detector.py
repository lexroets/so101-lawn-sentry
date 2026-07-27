"""Detect uncut grass patches from the OAK-D depth stream.

The mown deck is flat and level, so uncut grass stands PROUD of it — nearer
to the camera than the deck plane at the same image row. This is the exact
signal the battery detector used for objects standing proud of the desk, so
the approach is proven; only the target changed.

Returns a list of patches, nearest first:
    {'u','v','z','area','bbox'}
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
import warnings
from datetime import timedelta
import cv2
import depthai as dai
import numpy as np

from trimmer_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    GRASS_MIN_PROUD_M, GRASS_MAX_PROUD_M, PATCH_MIN_AREA_PX, PATCH_MAX_AREA_PX,
    DETECT_Z_MIN_M, DETECT_Z_MAX_M,
)

_STRIDE = 4          # column subsampling for the deck-depth estimate


def build_pipeline():
    """RGB + aligned StereoDepth, synchronised. Returns (pipeline, queue)."""
    pipeline = dai.Pipeline()
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rgb_out = rgb.requestOutput(size=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                fps=CAMERA_FPS, enableUndistortion=True)
    left_out = left.requestOutput(size=(640, 400),
                                  type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    right_out = right.requestOutput(size=(640, 400),
                                    type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    rgb_out.link(stereo.inputAlignTo)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=50))
    sync.setSyncAttempts(-1)
    rgb_out.link(sync.inputs['rgb'])
    stereo.depth.link(sync.inputs['depth'])
    return pipeline, sync.out.createOutputQueue(maxSize=1, blocking=False)


def _row_deck(depth_m):
    """Deck depth per image row: median across the row (subsampled).

    Robust because the flat deck dominates each row; grass blades are a
    minority so they don't move the median much."""
    d = np.array(depth_m[:, ::_STRIDE], dtype=np.float32)
    d[~(np.isfinite(d) & (d >= DETECT_Z_MIN_M) & (d <= DETECT_Z_MAX_M + 0.3))] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmedian(d, axis=1)


def _measure_z(depth_m, u, v, win=3):
    y0, y1 = max(0, v - win), min(depth_m.shape[0], v + win + 1)
    x0, x1 = max(0, u - win), min(depth_m.shape[1], u + win + 1)
    roi = depth_m[y0:y1, x0:x1]
    vals = roi[np.isfinite(roi) & (roi >= DETECT_Z_MIN_M) & (roi <= DETECT_Z_MAX_M)]
    if vals.size < 4:
        return None
    return float(np.median(vals))


def detect_patches(image, depth_m, stats=None, min_proud=None, max_proud=None,
                   min_area=None, max_area=None):
    """Find things standing PROUD of the deck (grass patches by default; pass
    the object-* thresholds for a battery on a desk). `image` gives shape."""
    min_proud = GRASS_MIN_PROUD_M if min_proud is None else min_proud
    max_proud = GRASS_MAX_PROUD_M if max_proud is None else max_proud
    min_area = PATCH_MIN_AREA_PX if min_area is None else min_area
    max_area = PATCH_MAX_AREA_PX if max_area is None else max_area
    h_img, w_img = depth_m.shape[:2]
    deck = _row_deck(depth_m)                     # per-row deck depth
    valid = np.isfinite(depth_m) & (depth_m >= DETECT_Z_MIN_M) \
        & (depth_m <= DETECT_Z_MAX_M)
    with np.errstate(invalid='ignore'):
        proud = deck[:, None] - depth_m           # +ve = nearer than deck
        grass = valid & (proud > min_proud) & (proud < max_proud)
    mask = grass.astype(np.uint8) * 255
    # Knit blades/pixels of one object together, drop speckle.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    if stats is not None:
        stats['contours'] = len(contours)
    for c in contours:
        area = cv2.contourArea(c)
        if not min_area <= area <= max_area:
            if stats is not None:
                stats['area_rej'] = stats.get('area_rej', 0) + 1
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        u = int(M['m10'] / M['m00'])
        v = int(M['m01'] / M['m00'])
        z = _measure_z(depth_m, u, v)
        if z is None:
            if stats is not None:
                stats['depth_rej'] = stats.get('depth_rej', 0) + 1
            continue
        x, y, w, h = cv2.boundingRect(c)
        patches.append({'u': u, 'v': v, 'z': z, 'area': float(area),
                        'bbox': (x, y, w, h)})
    patches.sort(key=lambda p: p['z'])            # nearest first
    if stats is not None:
        stats['patches'] = len(patches)
    return patches
