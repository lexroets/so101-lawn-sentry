"""Detect uncut-grass REGIONS by fusing three signals (per the literature):

  1. learned CNN texture   — MobileNetV2 tile classifier (grass_infer)
  2. classical GLCM texture — coarser texture = uncut (works with NO model)
  3. depth height          — uncut grass stands proud of the mown deck

The frame is tiled; each tile gets an uncut score from whatever signals are
available; scores are smoothed over time, thresholded, and grouped into
regions. Returns targets {u,v,z,area,score,bbox}, nearest first — the same
shape reach_object / main_trimmer already consume, so it drops straight in.
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

import trimmer_config as _cfg      # for live-tunable GRASS_UNCUT_CONF
from trimmer_config import (
    GRASS_TILE_PX, GRASS_STRIDE_PX, GRASS_MIN_REGION_TILES,
    GRASS_FUSE_DEPTH, GRASS_TEMPORAL, GRASS_MIN_PROUD_M, GRASS_MAX_PROUD_M,
    DETECT_Z_MIN_M, DETECT_Z_MAX_M,
)
from trim_detector import _row_deck, _measure_z
from grass_infer import GrassClassifier


def _dir_contrast(q, levels, axis):
    """GLCM contrast along one axis (0=vertical, 1=horizontal), quantised q."""
    if axis == 1:
        a, b = q[:, :-1].ravel(), q[:, 1:].ravel()
    else:
        a, b = q[:-1, :].ravel(), q[1:, :].ravel()
    g = np.zeros((levels, levels), np.float64)
    np.add.at(g, (a, b), 1.0)
    s = g.sum()
    if s == 0:
        return 0.0
    g /= s
    i = np.arange(levels)[:, None]; j = np.arange(levels)[None, :]
    return float(np.sum(g * (i - j) ** 2))


def _glcm_texture(gray, levels=16):
    """Grass-texture score from GLCM contrast. Uncut grass is coarse AND
    directional (tall vertical blades + shadows), so we take the mean contrast
    over horizontal/vertical PLUS a bonus for the H/V asymmetry that blades
    create. Cut grass is smooth and near-isotropic -> low on both."""
    q = np.clip(gray.astype(np.int32) * levels // 256, 0, levels - 1)
    ch = _dir_contrast(q, levels, axis=1)
    cv = _dir_contrast(q, levels, axis=0)
    return 0.5 * (ch + cv) + 0.5 * abs(ch - cv)


class GrassDetector:
    def __init__(self):
        self.clf = GrassClassifier()
        self._ema = None            # temporal score grid

    def detect(self, image, depth_m, stats=None):
        H, W = depth_m.shape[:2]
        t, s = GRASS_TILE_PX, GRASS_STRIDE_PX
        xs = list(range(0, W - t + 1, s))
        ys = list(range(0, H - t + 1, s))
        if not xs or not ys:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        deck = _row_deck(depth_m)

        coords, tiles, glcm, proud = [], [], [], []
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                coords.append((xi, yi, x, y))
                tiles.append(image[y:y + t, x:x + t])
                glcm.append(_glcm_texture(gray[y:y + t, x:x + t]))
                # mean height above the deck over the tile
                sub = depth_m[y:y + t, x:x + t]
                rows = deck[y:y + t][:, None]
                valid = np.isfinite(sub) & (sub >= DETECT_Z_MIN_M) \
                    & (sub <= DETECT_Z_MAX_M)
                pv = (rows - sub)[valid]
                proud.append(float(np.median(pv)) if pv.size else 0.0)

        glcm = np.array(glcm, np.float32)
        # relative texture: high within THIS frame = uncut (adapts to lighting)
        rng = float(np.ptp(glcm))
        if rng > 1e-6:
            glcm_n = (glcm - glcm.min()) / rng
        else:
            glcm_n = np.zeros_like(glcm)
        proud = np.array(proud, np.float32)

        cnn = self.clf.uncut_probs(tiles) if self.clf.ready else None

        # ---- fuse per tile ----
        if cnn is not None:
            score = 0.6 * cnn + 0.4 * glcm_n
        else:
            score = glcm_n.copy()
        if GRASS_FUSE_DEPTH:
            tall = (proud > GRASS_MIN_PROUD_M) & (proud < GRASS_MAX_PROUD_M)
            score = np.where(tall, np.minimum(1.0, score + 0.15), score * 0.5)

        # ---- temporal smoothing on the grid ----
        grid = np.zeros((len(ys), len(xs)), np.float32)
        for (xi, yi, _, _), sc in zip(coords, score):
            grid[yi, xi] = sc
        if self._ema is None or self._ema.shape != grid.shape:
            self._ema = grid
        else:
            self._ema = GRASS_TEMPORAL * self._ema + (1 - GRASS_TEMPORAL) * grid
        grid = self._ema

        # ---- threshold -> regions ----
        mask = (grid >= _cfg.GRASS_UNCUT_CONF).astype(np.uint8)
        n, lbl, st, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if stats is not None:
            stats['tiles'] = len(tiles)
            stats['uncut_tiles'] = int(mask.sum())
            stats['regions'] = max(0, n - 1)

        patches = []
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < GRASS_MIN_REGION_TILES:
                continue
            cxg, cyg = cent[k]
            u = int(xs[min(len(xs) - 1, int(round(cxg)))] + t // 2)
            v = int(ys[min(len(ys) - 1, int(round(cyg)))] + t // 2)
            z = _measure_z(depth_m, u, v, win=t // 3)
            if z is None:
                continue
            # bbox from the tile-grid extent
            x0 = xs[st[k, cv2.CC_STAT_LEFT]]
            y0 = ys[st[k, cv2.CC_STAT_TOP]]
            x1 = xs[min(len(xs) - 1,
                        st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] - 1)] + t
            y1 = ys[min(len(ys) - 1,
                        st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] - 1)] + t
            region_scores = grid[lbl == k]
            patches.append({
                'u': u, 'v': v, 'z': z,
                'area': float(st[k, cv2.CC_STAT_AREA]),
                'score': float(region_scores.mean()),
                'bbox': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            })
        patches.sort(key=lambda p: p['z'])       # nearest first
        return patches
