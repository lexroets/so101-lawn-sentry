"""Generate a SYNTHETIC grass-tile dataset (uncut / cut / other).

For when you can't photograph real grass yet. It procedurally renders the
texture cues the classifier keys on — uncut = tall, coarse, directional blades
with inter-blade shadows; cut = short, smooth, uniform (with faint mow
stripes); other = soil / concrete / mulch / dead grass. Heavy per-tile
variation (lighting, hue, density) so the model sees range.

    python make_synth_grass.py            # ~600/600/400 tiles
    python make_synth_grass.py 1200 1200 800

Writes grass_dataset/<label>/. Zip it and train with train_grass_colab.py.

CAVEAT: synthetic != real. This BOOTSTRAPS the CNN; the GLCM+depth signals in
grass_detector.py already work with no model, and you can fine-tune on real
photos later by simply adding them to grass_dataset/ and retraining.
"""
import os
import sys
import numpy as np
import cv2

from trimmer_config import GRASS_TILE_PX

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grass_dataset')
T = GRASS_TILE_PX
rng = np.random.default_rng()


def _lowfreq(scale, amp):
    """Smooth low-frequency field in [-amp, amp] via upscaled noise."""
    s = max(2, T // scale)
    n = rng.standard_normal((s, s)).astype(np.float32)
    n = cv2.resize(n, (T, T), interpolation=cv2.INTER_CUBIC)
    n /= (np.abs(n).max() + 1e-6)
    return n * amp


def _green_base():
    # BGR grass green with hue/brightness variation
    g = rng.integers(90, 170)
    b = rng.integers(20, 70)
    r = rng.integers(30, 90)
    img = np.zeros((T, T, 3), np.float32)
    img[..., 0] = b; img[..., 1] = g; img[..., 2] = r
    # blotchy colour variation
    for c in range(3):
        img[..., c] += _lowfreq(6, rng.uniform(8, 22))
    return img


def make_uncut():
    img = _green_base()
    # tall vertical blades: bright + dark streaks, jittered, varying length
    n_blades = rng.integers(60, 140)
    for _ in range(n_blades):
        x = rng.integers(0, T)
        w = rng.integers(1, 3)
        top = rng.integers(0, T // 2)
        bot = rng.integers(T // 2, T)
        shade = rng.uniform(-60, 70)
        xx = np.clip(x + rng.integers(-1, 2), 0, T - 1)
        img[top:bot, max(0, xx - w):xx + w + 1, 1] += shade    # green channel
        img[top:bot, max(0, xx - w):xx + w + 1, 0] += shade * 0.3
    # inter-blade shadows (dark vertical gaps)
    for _ in range(rng.integers(20, 50)):
        x = rng.integers(0, T)
        img[:, x:x + 1, :] -= rng.uniform(15, 45)
    img += _lowfreq(3, 10)[..., None]            # coarse structure
    img += rng.standard_normal((T, T, 3)).astype(np.float32) * rng.uniform(4, 12)
    return _finish(img)


def make_cut():
    img = _green_base()
    # smooth, uniform, fine short texture only
    img += rng.standard_normal((T, T, 3)).astype(np.float32) * rng.uniform(2, 6)
    img += _lowfreq(10, rng.uniform(3, 8))[..., None]
    # faint mow stripes (horizontal bands)
    if rng.random() < 0.6:
        period = rng.integers(12, 30)
        band = (np.sin(np.arange(T) / period * np.pi) * rng.uniform(4, 10))
        img += band[:, None, None]
    return _finish(img)


def make_other():
    kind = rng.integers(0, 4)
    img = np.zeros((T, T, 3), np.float32)
    if kind == 0:      # soil / dirt (brown, blotchy)
        img[..., 0] = rng.integers(30, 60); img[..., 1] = rng.integers(50, 90)
        img[..., 2] = rng.integers(70, 120)
        for c in range(3):
            img[..., c] += _lowfreq(5, 20)
        img += rng.standard_normal((T, T, 3)).astype(np.float32) * 10
    elif kind == 1:    # concrete / pavement (grey, speckle)
        base = rng.integers(120, 190)
        img[:] = base
        img += rng.standard_normal((T, T, 3)).astype(np.float32) * rng.uniform(6, 16)
        img += _lowfreq(8, 8)[..., None]
    elif kind == 2:    # mulch (dark brown chunky)
        img[..., 0] = rng.integers(20, 45); img[..., 1] = rng.integers(35, 65)
        img[..., 2] = rng.integers(55, 95)
        for _ in range(rng.integers(30, 70)):        # chunks
            x, y = rng.integers(0, T, 2)
            r = rng.integers(3, 8)
            cv2.circle(img, (int(x), int(y)), int(r),
                       [float(rng.integers(20, 90)) for _ in range(3)], -1)
    else:              # dead/dry grass (tan, some streaks)
        img[..., 0] = rng.integers(60, 100); img[..., 1] = rng.integers(120, 170)
        img[..., 2] = rng.integers(150, 200)
        for _ in range(rng.integers(30, 70)):
            x = rng.integers(0, T)
            img[:, x:x + 1, :] += rng.uniform(-30, 30)
        img += _lowfreq(4, 12)[..., None]
    return _finish(img)


def _finish(img):
    img += rng.uniform(-25, 25)                  # global lighting
    if rng.random() < 0.5:
        img = img[:, ::-1]                        # h-flip
    if rng.random() < 0.5:
        img = img[::-1, :]                        # v-flip (grass has no up)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    n_uncut = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n_cut = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    n_other = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    plan = [('uncut', n_uncut, make_uncut), ('cut', n_cut, make_cut),
            ('other', n_other, make_other)]
    for name, n, fn in plan:
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            cv2.imwrite(os.path.join(d, f'synth_{name}_{i:05d}.png'), fn())
        print(f'  {name}: {n} tiles')
    print(f'wrote {OUT}. Zip it and train with train_grass_colab.py.')


if __name__ == '__main__':
    main()
