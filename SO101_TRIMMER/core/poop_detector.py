"""Detect dog poop on a grass lawn — heuristic first, CNN-assisted when trained.

Poop on grass is a compact blob that is NOT grass-green: it is brown, dark
and/or desaturated, and usually stands slightly proud of the blades. We fuse
three cheap, robust signals (the same recipe as the grass detector, retuned):

  1. COLOUR  — "non-green blob on green" in HSV (brown / dark / desaturated)
  2. SHAPE   — compact blob of plausible on-ground size (fills its bbox)
  3. DEPTH   — sits slightly above the fitted ground (optional, if depth given)

If poop_cls.onnx is trained and present, its per-blob 'poop' probability is
fused in for sharper discrimination. With NO model, colour+shape(+depth) alone
already flag candidates so you can practise-run and collect data immediately.

detect() returns detections nearest-first:
    {'u','v','z','area','score','fill','bbox'}   (z may be None w/o depth)

Self-contained: imports only numpy, cv2, poop_config, poop_infer — NO depthai,
so it can be unit-tested and previewed without the camera stack.
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
import numpy as np
import cv2

import poop_config as _cfg
from poop_infer import PoopClassifier

_STRIDE = 4          # column subsampling for the ground-depth estimate


def _row_ground(depth_m):
    """Per-image-row ground depth = median across the row (subsampled).

    A roughly flat lawn dominates each row, so its median is the ground plane
    at that row; a poop blob is a small minority and barely shifts it."""
    d = np.array(depth_m[:, ::_STRIDE], dtype=np.float32)
    lo, hi = _cfg.DETECT_Z_MIN_M, _cfg.DETECT_Z_MAX_M + 0.3
    d[~(np.isfinite(d) & (d >= lo) & (d <= hi))] = np.nan
    m = np.full(d.shape[0], np.nan, np.float32)
    for r in range(d.shape[0]):
        row = d[r][np.isfinite(d[r])]
        if row.size:
            m[r] = np.median(row)
    return m


def _measure_z(depth_m, u, v, win=4):
    y0, y1 = max(0, v - win), min(depth_m.shape[0], v + win + 1)
    x0, x1 = max(0, u - win), min(depth_m.shape[1], u + win + 1)
    roi = depth_m[y0:y1, x0:x1]
    vals = roi[np.isfinite(roi) & (roi >= _cfg.DETECT_Z_MIN_M)
               & (roi <= _cfg.DETECT_Z_MAX_M)]
    if vals.size < 4:
        return None
    return float(np.median(vals))


class PoopDetector:
    def __init__(self):
        self.clf = PoopClassifier()
        # optional learned detector (distilled from SAM 3.1). Loaded lazily so
        # the heuristic path has zero extra dependencies.
        self.yolo = None
        mode = getattr(_cfg, 'POOP_USE_YOLO', 'auto')
        want = (mode is True) or (isinstance(mode, str) and mode.lower() == 'auto')
        if want:
            try:
                from poop_yolo import PoopYolo
                y = PoopYolo()
                self.yolo = y if y.ready else None
            except Exception as exc:
                print(f"[poop] YOLO unavailable ({exc}); using colour heuristic.")
        print('[poop] detector =',
              'TRAINED MODEL (poop_yolo.onnx)' if self.yolo else 'colour heuristic')

    def _detect_yolo(self, image, depth_m, roi_poly, max_targets, stats):
        """Learned-model path: YOLO boxes -> the same detection dicts, with the
        lawn ROI, depth and max-targets rules applied exactly as before."""
        ground = _row_ground(depth_m) if depth_m is not None else None
        roi = self._roi_mask(image.shape, roi_poly) if roi_poly is not None else None
        rej = 0
        dets = []
        for b in self.yolo.infer(image):
            u, v = b['u'], b['v']
            # Test the GROUND-CONTACT point (bottom-centre of the box), not the
            # box centre. An object standing on the far edge has its centre
            # above the edge line, and a centre test silently discards it —
            # which is why far-edge targets were boxed on video but never
            # reached the map. _roi_mask is dilated by POOP_ROI_MARGIN_PX too.
            bx, by, bw, bh = b['bbox']
            gu, gv = int(bx + bw / 2), int(by + bh)
            if roi is not None:
                inside = False
                for (tu, tv) in ((gu, gv), (u, v)):     # contact point, then centre
                    if 0 <= tv < roi.shape[0] and 0 <= tu < roi.shape[1] and roi[tv, tu]:
                        inside = True
                        break
                if not inside:
                    rej += 1
                    continue
            z = _measure_z(depth_m, u, v) if depth_m is not None else None
            x, y, w, h = b['bbox']
            dets.append({'u': u, 'v': v, 'z': z, 'area': float(w * h),
                         'score': b['conf'], 'fill': 1.0, 'bbox': (x, y, w, h),
                         'cnn': b['conf']})
        n = _cfg.POOP_MAX_TARGETS if max_targets is None else max_targets
        if n and len(dets) > n:
            dets = sorted(dets, key=lambda d: -d['score'])[:n]
        dets.sort(key=lambda d: (d['z'] if d['z'] is not None else 9.9, -d['score']))
        if stats is not None:
            stats['detections'] = len(dets)
            stats['off_lawn_rejected'] = rej
            stats['source'] = 'yolo'
        return dets

    # -- colour segmentation -------------------------------------------------
    def _brown_masks(self, hsv):
        """Return (brown, warm) boolean maps. `warm` = brown/red hue with some
        saturation; `brown` also requires a dull, non-bright value OR a
        dark-but-still-warm blob. This is what makes it POOP-specific rather
        than 'anything not green'."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        warm_hue = (((h >= _cfg.POOP_BROWN_HUE_LO) & (h <= _cfg.POOP_BROWN_HUE_HI))
                    | (h >= _cfg.POOP_HUE_WRAP))
        warm = warm_hue & (s >= _cfg.POOP_SAT_MIN)
        brown = warm & (v >= _cfg.POOP_VAL_MIN) & (v <= _cfg.POOP_VAL_MAX)
        darkbrown = (warm_hue & (v <= _cfg.POOP_DARK_VAL)
                     & (s >= _cfg.POOP_DARK_SAT_MIN) & (v >= _cfg.POOP_VAL_MIN))
        return (brown | darkbrown), warm

    def _roi_mask(self, shape, roi_poly):
        """Boolean map, True INSIDE the lawn quad (so we ignore everything off
        the grass — the single biggest false-positive killer).

        Grown by POOP_ROI_MARGIN_PX because the quad is a hand-clicked
        approximation: a target sitting right on the boundary would otherwise
        be discarded over a few pixels of calibration error."""
        m = np.zeros(shape[:2], np.uint8)
        pts = np.asarray(roi_poly, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
        margin = int(getattr(_cfg, 'POOP_ROI_MARGIN_PX', 0))
        if margin > 0:
            k = 2 * margin + 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return m > 0

    def _candidate_mask(self, image, roi=None):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s = hsv[..., 0], hsv[..., 1]
        grass = ((h >= _cfg.GRASS_HUE_LO) & (h <= _cfg.GRASS_HUE_HI)
                 & (s >= _cfg.GRASS_SAT_MIN))
        if getattr(_cfg, 'POOP_DEMO_BOX_MODE', False):
            # DEMO: match one specific object by a tight HSV band (the box).
            lo = np.array(_cfg.POOP_BOX_HSV_LO, np.uint8).copy()
            hi = np.array(_cfg.POOP_BOX_HSV_HI, np.uint8)
            # Clamp the floors so a too-loose band can't match desaturated
            # (cream/white plastic) or dark (body/tyres) mower parts.
            lo[1] = max(int(lo[1]), getattr(_cfg, 'POOP_BOX_SAT_FLOOR', 55))
            lo[2] = max(int(lo[2]), getattr(_cfg, 'POOP_BOX_VAL_FLOOR', 90))
            band = cv2.inRange(hsv, lo, hi) > 0
        else:
            band, _warm = self._brown_masks(hsv)
        cand = (~grass) & band
        # NOTE: depth is deliberately NOT required here. A flat, textureless
        # box returns no stereo depth, so requiring valid depth would delete
        # it. We rely on colour + the lawn ROI instead, and use depth only as a
        # scoring bonus below.
        if roi is not None:
            cand &= roi
        mask = (cand.astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask, hsv, band

    def detect(self, image, depth_m=None, stats=None, roi_poly=None,
               max_targets=None):
        """roi_poly: optional [[x,y],...] lawn quad — only blobs INSIDE it are
        considered. max_targets: keep the strongest N (defaults to
        POOP_MAX_TARGETS; 0 = all)."""
        # learned detector takes over when available
        if self.yolo is not None:
            return self._detect_yolo(image, depth_m, roi_poly, max_targets, stats)

        H, W = image.shape[:2]
        ground = None
        if depth_m is not None:
            ground = _row_ground(depth_m)
        roi = self._roi_mask(image.shape, roi_poly) if roi_poly is not None else None
        mask, hsv, band = self._candidate_mask(image, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if stats is not None:
            stats['contours'] = len(contours)

        demo = getattr(_cfg, 'POOP_DEMO_BOX_MODE', False)
        min_area = _cfg.POOP_BOX_MIN_AREA_PX if demo else _cfg.POOP_MIN_AREA_PX
        max_area = _cfg.POOP_BOX_MAX_AREA_PX if demo else _cfg.POOP_MAX_AREA_PX
        min_fill = _cfg.POOP_BOX_MIN_FILL if demo else _cfg.POOP_MIN_FILL

        dets, tiles, tile_idx = [], [], []
        for c in contours:
            area = cv2.contourArea(c)
            if not min_area <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill = area / float(max(1, w * h))
            if fill < min_fill:
                continue
            # rectangularity = how well the blob fills its MINIMUM-area rectangle
            # (rotation-invariant). A clean box ~1.0; the mower's ragged parts low.
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area / rect_area if rect_area > 1 else 0.0
            if demo and rectangularity < getattr(_cfg, 'POOP_BOX_MIN_RECT', 0.62):
                if stats is not None:
                    stats['rect_rej'] = stats.get('rect_rej', 0) + 1
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            u = int(M['m10'] / M['m00'])
            v = int(M['m01'] / M['m00'])

            # blob pixel stats for the colour score. `band` is the discriminating
            # mask (tan box band in demo mode, brown mask otherwise), so the
            # fraction of the blob inside it is our colour-match confidence.
            blob = np.zeros((H, W), np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            bm = blob > 0
            vv = hsv[..., 2][bm]
            brown_frac = float(band[bm].mean())
            # POOP-SPECIFIC GATE: the blob must be genuinely brown, not just
            # 'not green'. Rejects grey stones, pale leaves, bright litter.
            if brown_frac < _cfg.POOP_MIN_BROWN_FRAC:
                if stats is not None:
                    stats['brown_rej'] = stats.get('brown_rej', 0) + 1
                continue
            if demo:
                # favour a bright, uniform, RECTANGULAR tan patch (the box);
                # do NOT reward darkness (that helped the dark mower win before).
                brightness = float(vv.mean()) / 255.0
                colour_score = float(np.clip(
                    0.4 * brown_frac + 0.3 * fill + 0.2 * rectangularity
                    + 0.1 * brightness, 0.0, 1.0))
            else:
                darkness = 1.0 - float(vv.mean()) / 255.0
                colour_score = float(np.clip(
                    0.5 * brown_frac + 0.3 * fill + 0.12 * darkness
                    + 0.08 * min(1.0, area / 4000.0), 0.0, 1.0))

            z = _measure_z(depth_m, u, v) if depth_m is not None else None
            score = colour_score
            if _cfg.POOP_FUSE_DEPTH and z is not None and ground is not None \
                    and np.isfinite(ground[v]):
                proud = ground[v] - z
                if _cfg.POOP_MIN_PROUD_M <= proud <= _cfg.POOP_MAX_PROUD_M:
                    score = min(1.0, score + 0.15)

            det = {'u': u, 'v': v, 'z': z, 'area': float(area),
                   'score': score, 'fill': fill, 'bbox': (x, y, w, h)}
            dets.append(det)
            if self.clf.ready:
                tiles.append(self._crop_tile(image, x, y, w, h))
                tile_idx.append(len(dets) - 1)

        # ---- CNN refine (optional) ----
        if self.clf.ready and tiles:
            pp = self.clf.poop_probs(tiles)
            w = _cfg.POOP_CNN_WEIGHT
            for i, p in zip(tile_idx, pp):
                dets[i]['cnn'] = float(p)
                dets[i]['score'] = float(w * p + (1 - w) * dets[i]['score'])

        dets = [d for d in dets if d['score'] >= _cfg.POOP_SCORE_MIN]
        # keep only the strongest N (e.g. the single brown box) if configured
        n = _cfg.POOP_MAX_TARGETS if max_targets is None else max_targets
        if n and len(dets) > n:
            dets = sorted(dets, key=lambda d: -d['score'])[:n]
        # nearest first if we have depth, else biggest/most-confident first
        dets.sort(key=lambda d: (d['z'] if d['z'] is not None else 9.9,
                                 -d['score']))
        if stats is not None:
            stats['detections'] = len(dets)
        return dets

    def _crop_tile(self, image, x, y, w, h):
        t = _cfg.POOP_TILE_PX
        cx, cy = x + w // 2, y + h // 2
        half = max(t // 2, w // 2, h // 2)
        x0 = max(0, cx - half); y0 = max(0, cy - half)
        x1 = min(image.shape[1], cx + half); y1 = min(image.shape[0], cy + half)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            crop = image[y:y + max(1, h), x:x + max(1, w)]
        return crop
