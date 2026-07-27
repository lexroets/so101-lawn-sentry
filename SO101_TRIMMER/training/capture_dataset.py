"""Guided capture of YOUR setup — the fix for "only sees big foreground poop".

The public dataset is close-up phone photos, so the model never learned what
poop looks like at 2 m through YOUR camera. This walks you through capturing
that, at the distances and positions the demo will actually use.

    python capture_dataset.py

It prompts you through a shot list (near / mid / far, left / centre / right),
saving to mydata/images/. Then label them with:

    python label_frames.py

Keys:  SPACE save a frame · a auto-capture on/off · n next prompt ·
       b back a prompt   · q quit
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
from pathlib import Path

import cv2

from trim_detector import build_pipeline
from camera_session import CameraSession

from paths import DATA
OUT = DATA / 'mydata' / 'images'
AUTO_INTERVAL_S = 0.6
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Shot list: the variety that actually matters for a detector.
PROMPTS = [
    ('NEAR  centre  — target ~0.5 m in front', 25),
    ('NEAR  left',                              20),
    ('NEAR  right',                             20),
    ('MID   centre  — ~1.0-1.5 m',              30),
    ('MID   left',                              25),
    ('MID   right',                             25),
    ('FAR   centre  — as far as you want it to work', 30),
    ('FAR   left',                              25),
    ('FAR   right',                             25),
    ('TWO+ targets at different distances',     30),
    ('Target part-hidden in longer grass',      20),
    ('EMPTY lawn — no target at all (important!)', 30),
    ('DECOYS only: leaves / sticks / stones, NO poop', 30),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cam = CameraSession(build_pipeline)
    have = len(list(OUT.glob('*.jpg')))
    saved, auto, last, step = 0, False, 0.0, 0
    print(f'saving to {OUT}  (already have {have})')
    print('Work through the prompts. Move the target between shots — '
          'vary distance, angle and lighting.\n')
    win = 'Capture dataset — SPACE save · a auto · n next · q quit'
    try:
        while True:
            g = cam.get()
            if g is None:
                if cam.crashed() and not cam.restart():
                    break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            frame = g['rgb'].getCvFrame()
            now = time.monotonic()
            prompt, want = PROMPTS[step]
            done_this = saved

            disp = frame.copy()
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (w, 54), (18, 18, 18), -1)
            cv2.putText(disp, f'{step+1}/{len(PROMPTS)}  {prompt}', (10, 22),
                        _FONT, 0.52, (33, 224, 212), 1)
            cv2.putText(disp,
                        f'saved {have + saved} total   this run {saved}   '
                        f'auto {"ON" if auto else "off"}   target ~{want}',
                        (10, 44), _FONT, 0.46, (200, 200, 200), 1)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 255
            do_save = False
            if k == ord('q'):
                break
            elif k == ord('a'):
                auto = not auto
            elif k == ord(' '):
                do_save = True
            elif k == ord('n'):
                step = min(len(PROMPTS) - 1, step + 1)
                print(f'-> {PROMPTS[step][0]}')
            elif k == ord('b'):
                step = max(0, step - 1)
                print(f'-> {PROMPTS[step][0]}')
            if auto and now - last >= AUTO_INTERVAL_S:
                do_save = True
            if do_save:
                fn = OUT / f'cap_{int(now*1000)}.jpg'
                cv2.imwrite(str(fn), frame)      # RAW frame, no overlay
                saved += 1
                last = now
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        total = len(list(OUT.glob('*.jpg')))
        print(f'\ncaptured {saved} this run — {total} images in {OUT}')
        if total < 120:
            print('Aim for 150+ across all the prompts for a solid improvement.')
        print('Next:  python label_frames.py')


if __name__ == '__main__':
    main()
