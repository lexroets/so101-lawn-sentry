"""Label poop/grass/other tiles from the OAK-D camera, for training.

Point the camera at grass (with real or stand-in poop) and label tiles:
  - pick the active class with number keys, then CLICK a tile to save it
  - 'f' saves EVERY tile in the frame as the active class (fast for plain
    grass / background frames)

Tiles are POOP_TILE_PX and go to poop_dataset/<class>/, ready for
train_poop_colab.py. Aim for a few hundred of each; lots of varied 'other'
(leaves, dirt, sticks, shadows) makes the model reliable.

Keys:  1 poop · 2 grass · 3 other · click tile = save · f save whole frame ·
       u undo last · q quit
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
import time
import cv2
import numpy as np

import poop_config as cfg
from trim_detector import build_pipeline
from camera_session import CameraSession

_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABELS = list(cfg.POOP_LABELS)          # ('poop','grass','other')
COLORS = {'poop': (0, 90, 230), 'grass': (80, 220, 100), 'other': (200, 160, 0)}
T = cfg.POOP_TILE_PX


class State:
    def __init__(self):
        self.label = 'poop'
        self.saved = {k: 0 for k in LABELS}
        self.history = []
        self.click = None


def _counts_on_disk():
    out = {}
    for k in LABELS:
        d = os.path.join(cfg.POOP_DATASET_DIR, k)
        out[k] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return out


def _save(tile, label, st):
    d = os.path.join(cfg.POOP_DATASET_DIR, label)
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f'cap_{label}_{int(time.time()*1000)}_{st.saved[label]}.png')
    cv2.imwrite(fn, tile)
    st.saved[label] += 1
    st.history.append(fn)


def main():
    cam = CameraSession(build_pipeline)
    st = State()
    for k in LABELS:
        os.makedirs(os.path.join(cfg.POOP_DATASET_DIR, k), exist_ok=True)
    base = _counts_on_disk()
    win = 'SO-101 poop tile collector'
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.click = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            g = cam.get()
            if g is None:
                if cam.crashed():
                    if not cam.restart():
                        break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            frame = g['rgb'].getCvFrame()
            H, W = frame.shape[:2]
            xs = list(range(0, W - T + 1, T))
            ys = list(range(0, H - T + 1, T))

            # handle a click -> save that tile
            if st.click is not None:
                cx, cy = st.click; st.click = None
                x = min(xs, key=lambda v: abs(v + T // 2 - cx))
                y = min(ys, key=lambda v: abs(v + T // 2 - cy))
                _save(frame[y:y + T, x:x + T].copy(), st.label, st)

            disp = frame.copy()
            for x in xs:
                cv2.line(disp, (x, 0), (x, H), (60, 60, 60), 1)
            for y in ys:
                cv2.line(disp, (0, y), (W, y), (60, 60, 60), 1)
            cv2.rectangle(disp, (0, 0), (W, 26), (18, 18, 18), -1)
            tot = {k: base[k] + st.saved[k] for k in LABELS}
            cv2.putText(disp, f"[{st.label}]  " + "  ".join(
                f"{k}:{tot[k]}" for k in LABELS)
                + "   click=save  f=frame  u=undo  q=quit",
                (8, 19), _FONT, 0.5, COLORS[st.label], 1)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                break
            if k == ord('1'):
                st.label = 'poop'
            if k == ord('2'):
                st.label = 'grass'
            if k == ord('3'):
                st.label = 'other'
            if k == ord('f'):
                for y in ys:
                    for x in xs:
                        _save(frame[y:y + T, x:x + T].copy(), st.label, st)
            if k == ord('u') and st.history:
                last = st.history.pop()
                try:
                    os.remove(last)
                    lbl = os.path.basename(os.path.dirname(last))
                    st.saved[lbl] = max(0, st.saved[lbl] - 1)
                except OSError:
                    pass
    finally:
        print("saved this session:", st.saved)
        print("totals on disk:", _counts_on_disk())
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
