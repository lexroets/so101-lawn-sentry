"""Collect labelled grass tiles for the uncut/cut/other classifier.

Camera only (arm not powered) — safe. The frame is shown as a grid of tiles;
you label whole views (aim at an all-uncut patch, press 1) or single tiles
(click). Builds grass_dataset/<label>/ for train_grass_colab.py.

WHY TILES: "uncut grass" is a region, not an object, so we classify small
patches and let the uncut ones form the target — the same trick the CNN
lawn-segmentation and GLCM cut/uncut papers use, adapted to your reach camera.

LABELS
  1  uncut   tall / unmown grass (the target)
  2  cut      mown, short, uniform grass
  3  other    NOT grass — path, driveway, soil, mulch, shadow, your arm

KEYS
  1/2/3   set the current label AND bulk-save every tile in this frame as it
  click   save just the clicked tile as the current label
  b       toggle CONTINUOUS capture (saves the whole frame ~2x/sec as the
          current label while you slowly pan — the fast way to bulk-collect)
  u       undo the last save batch
  q       quit

HOW MUCH / HOW VARIED
  Aim for ~600 uncut, ~600 cut, ~400 other. Pan across the whole field,
  vary lighting (sun/shade), grass type, mowing direction, camera height and
  angle. Boundary tiles (half cut / half uncut) — label by what dominates,
  or skip them by clicking only the clear tiles. Variety beats volume.
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
import os
import time
from datetime import timedelta
import cv2
import depthai as dai
import numpy as np

from trimmer_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, GRASS_TILE_PX

from paths import DATA
_HERE = str(DATA)
OUT = str(DATA / 'grass_dataset')
LABELS = {ord('1'): 'uncut', ord('2'): 'cut', ord('3'): 'other'}
COLS = {'uncut': (80, 220, 100), 'cut': (0, 190, 255), 'other': (150, 150, 150)}
CONT_GAP_S = 0.5
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_rgb_pipeline():
    p = dai.Pipeline()
    rgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    out = rgb.requestOutput(size=(CAMERA_WIDTH, CAMERA_HEIGHT), fps=CAMERA_FPS,
                            enableUndistortion=True)
    q = out.createOutputQueue(maxSize=1, blocking=False)
    return p, q


def tiles_of(w, h, t):
    for y in range(0, h - t + 1, t):
        for x in range(0, w - t + 1, t):
            yield x, y


def counts():
    out = {}
    for name in ('uncut', 'cut', 'other'):
        d = os.path.join(OUT, name)
        out[name] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return out


def main():
    for name in ('uncut', 'cut', 'other'):
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
    pipeline, queue = build_rgb_pipeline()
    pipeline.start()
    print(__doc__)
    print('existing:', counts())

    label = 'uncut'
    click = {'pt': None}
    cont = False
    cont_next = 0.0
    last_batch = []
    msg, msg_t = '', 0.0
    t = GRASS_TILE_PX
    window = 'SO-101 grass collector (1/2/3 label · click tile · b bulk · q)'

    def on_mouse(ev, mx, my, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def save_tile(img, x, y, lab):
        crop = img[y:y + t, x:x + t]
        if crop.shape[0] != t or crop.shape[1] != t:
            return None
        path = os.path.join(OUT, lab,
                            f'{lab}_{int(time.time() * 1000)}_{x}_{y}.png')
        cv2.imwrite(path, crop)
        return path

    try:
        while pipeline.isRunning():
            now = time.monotonic()
            group = queue.tryGet()
            if group is not None:
                image = group.getCvFrame()
                clean = image.copy()

                if click['pt'] is not None:
                    mx, my = click['pt']; click['pt'] = None
                    x, y = (mx // t) * t, (my // t) * t
                    p = save_tile(clean, x, y, label)
                    if p:
                        last_batch = [p]
                        msg, msg_t = f'{label} +1 (click)', now + 1.0

                if cont and now >= cont_next:
                    cont_next = now + CONT_GAP_S
                    batch = [save_tile(clean, x, y, label)
                             for x, y in tiles_of(image.shape[1], image.shape[0], t)]
                    last_batch = [b for b in batch if b]
                    msg, msg_t = f'{label} +{len(last_batch)} (bulk)', now + 0.6

                # overlay grid
                for x, y in tiles_of(image.shape[1], image.shape[0], t):
                    cv2.rectangle(image, (x, y), (x + t, y + t),
                                  (60, 60, 60), 1)
                cv2.rectangle(image, (0, 0), (image.shape[1], 30), (18, 18, 18), -1)
                c = counts()
                cv2.putText(image, f"LABEL: {label.upper()}   "
                            f"uncut {c['uncut']}  cut {c['cut']}  other {c['other']}"
                            + ('   [CONT]' if cont else ''),
                            (8, 21), _FONT, 0.55, COLS[label], 2)
                if now < msg_t:
                    cv2.putText(image, msg, (8, image.shape[0] - 12), _FONT,
                                0.6, (0, 255, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key in LABELS:
                label = LABELS[key]
                # bulk-label the whole current frame too
                if group is not None:
                    batch = [save_tile(clean, x, y, label)
                             for x, y in tiles_of(clean.shape[1], clean.shape[0], t)]
                    last_batch = [b for b in batch if b]
                    msg, msg_t = f'{label} +{len(last_batch)} (frame)', now + 0.8
            if key == ord('b'):
                cont = not cont; cont_next = 0.0
            if key == ord('u'):
                n = 0
                for p in last_batch:
                    try:
                        os.remove(p); n += 1
                    except OSError:
                        pass
                last_batch = []
                msg, msg_t = f'undid {n}', now + 1.0
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        c = counts()
        print('\ndataset now:', c)
        thin = [k for k, v in c.items() if v < 300]
        if thin:
            print('  thin classes (<300):', thin, '— collect more variety.')
        print('Next: zip grass_dataset, train with train_grass_colab.py in Colab.')


if __name__ == '__main__':
    main()
