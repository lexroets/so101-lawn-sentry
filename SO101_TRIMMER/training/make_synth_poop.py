"""Generate a SYNTHETIC poop/grass/other dataset to bootstrap the classifier.

Real dog-poop photos are best (see fetch_poop_dataset.py to pull the public
ScatSpotter set, and collect_poop.py to shoot your own). This procedurally
renders blob-centred tiles so you can train something TODAY:

  poop  : a compact brown/dark lumpy mass on grass (with sheen + shadow)
  grass : plain lawn texture (a green tile must NOT read as poop)
  other : brown-but-not-poop HARD NEGATIVES — leaves, sticks, bare dirt,
          stones, dark shadow on grass. These teach the model the difference.

Tiles are POOP_TILE_PX and match what poop_detector crops around each blob.

    python make_synth_poop.py            # ~500/500/500
    python make_synth_poop.py 900 900 900

CAVEAT: synthetic != real. Bootstrap only; the colour+depth detector already
runs with no model. Add real photos to poop_dataset/ later and retrain.
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
import os
import sys
import numpy as np
import cv2

from poop_config import POOP_TILE_PX, POOP_DATASET_DIR

T = POOP_TILE_PX
rng = np.random.default_rng()


def _lowfreq(scale, amp):
    s = max(2, T // scale)
    n = rng.standard_normal((s, s)).astype(np.float32)
    n = cv2.resize(n, (T, T), interpolation=cv2.INTER_CUBIC)
    n /= (np.abs(n).max() + 1e-6)
    return n * amp


def grass_bg():
    """A lawn background tile (green, bladey)."""
    img = np.zeros((T, T, 3), np.float32)
    img[..., 0] = rng.integers(20, 70)          # B
    img[..., 1] = rng.integers(95, 165)         # G (green dominant)
    img[..., 2] = rng.integers(30, 85)          # R
    for c in range(3):
        img[..., c] += _lowfreq(6, rng.uniform(6, 16))
    # faint vertical blades
    for _ in range(rng.integers(30, 80)):
        x = rng.integers(0, T)
        img[:, x:x + 1, 1] += rng.uniform(-35, 40)
    img += rng.standard_normal((T, T, 3)).astype(np.float32) * rng.uniform(3, 8)
    return img


def _brown():
    palette = [(28, 38, 55), (35, 50, 78), (22, 30, 42), (45, 65, 95),
               (30, 45, 68), (18, 24, 32)]      # BGR dark/mid browns
    return list(palette[rng.integers(0, len(palette))])


def make_poop():
    img = grass_bg()
    cx, cy = T // 2 + rng.integers(-8, 9), T // 2 + rng.integers(-8, 9)
    R = rng.integers(int(T * 0.22), int(T * 0.36))
    # soft shadow: darken the grass under where the mass will sit
    shadow = img.copy()
    cv2.circle(shadow, (cx, cy), int(R * 1.3), (0, 0, 0), -1, cv2.LINE_AA)
    img = cv2.addWeighted(img, 0.75, shadow, 0.25, 0)
    # lumpy mass: several overlapping brown blobs (coiled/segmented look)
    n_lumps = rng.integers(4, 9)
    for _ in range(n_lumps):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, R * 0.6)
        lx = int(cx + rad * np.cos(ang)); ly = int(cy + rad * np.sin(ang))
        lr = rng.integers(int(R * 0.35), int(R * 0.8))
        cv2.circle(img, (lx, ly), lr, _brown(), -1, cv2.LINE_AA)
    # texture + a moist highlight
    m = np.zeros((T, T), np.uint8)
    cv2.circle(m, (cx, cy), R, 255, -1)
    tex = (rng.standard_normal((T, T, 3)).astype(np.float32)
           * rng.uniform(6, 14))
    img[m > 0] += tex[m > 0]
    if rng.random() < 0.7:
        hx = cx + rng.integers(-R // 3, R // 3)
        hy = cy - rng.integers(0, R // 2)
        cv2.circle(img, (hx, hy), rng.integers(2, 5),
                   (200, 210, 210), -1, cv2.LINE_AA)   # specular sheen
    return _finish(img)


def make_other():
    img = grass_bg()
    kind = rng.integers(0, 5)
    if kind == 0:      # dead leaf (tan, flat, thin ellipse w/ vein)
        cx, cy = T // 2 + rng.integers(-6, 7), T // 2 + rng.integers(-6, 7)
        col = (int(60 + rng.integers(0, 40)), int(110 + rng.integers(0, 40)),
               int(150 + rng.integers(0, 50)))
        ax = int(rng.integers(int(T*0.18), int(T*0.34))); ayr = int(rng.integers(6, 14))
        ang = int(rng.integers(0, 180))
        cv2.ellipse(img, (cx, cy), (ax, ayr), ang, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ax, ayr), ang, 0, 360,
                    (40, 70, 90), 1, cv2.LINE_AA)
    elif kind == 1:    # stick / twig (thin brown rectangle)
        p1 = (int(rng.integers(0, T)), int(rng.integers(0, T)))
        p2 = (int(rng.integers(0, T)), int(rng.integers(0, T)))
        cv2.line(img, p1, p2, _brown(), int(rng.integers(2, 5)), cv2.LINE_AA)
    elif kind == 2:    # bare dirt patch (spread brown, NOT a compact blob)
        b = grass_bg()
        b[..., 0] = rng.integers(35, 60); b[..., 1] = rng.integers(60, 95)
        b[..., 2] = rng.integers(85, 125)
        for c in range(3):
            b[..., c] += _lowfreq(4, 18)
        alpha = np.clip(_lowfreq(3, 1.0) + 0.4, 0, 1)[..., None]
        img = img * (1 - alpha) + b * alpha
    elif kind == 3:    # stone (grey circle)
        cx, cy = int(T // 2 + rng.integers(-8, 9)), int(T // 2 + rng.integers(-8, 9))
        g = int(rng.integers(110, 175))
        cv2.circle(img, (cx, cy), int(rng.integers(int(T*0.16), int(T*0.3))),
                   (g, g, g), -1, cv2.LINE_AA)
    else:              # dark shadow on grass (dark GREEN, not brown)
        sh = img.copy()
        cv2.circle(sh, (T//2 + rng.integers(-10, 10), T//2 + rng.integers(-10, 10)),
                   rng.integers(int(T*0.25), int(T*0.45)), (0, 0, 0), -1, cv2.LINE_AA)
        img = cv2.addWeighted(img, 0.5, sh, 0.5, 0)
    return _finish(img)


def make_grass():
    return _finish(grass_bg())


def _finish(img):
    img += rng.uniform(-22, 22)                  # global lighting
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    n_poop = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_grass = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_other = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    plan = [('poop', n_poop, make_poop), ('grass', n_grass, make_grass),
            ('other', n_other, make_other)]
    for name, n, fn in plan:
        d = os.path.join(POOP_DATASET_DIR, name)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            cv2.imwrite(os.path.join(d, f'synth_{name}_{i:05d}.png'), fn())
        print(f'  {name}: {n} tiles')
    print(f'wrote {POOP_DATASET_DIR}. Zip it and train with train_poop_colab.py.')


if __name__ == '__main__':
    main()
