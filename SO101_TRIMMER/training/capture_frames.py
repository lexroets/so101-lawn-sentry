"""Rapidly capture RAW camera frames for training — no labelling needed.

SAM does the labelling later, so here you just collect lots of varied photos of
poop (or your stand-in box) on the grass. Move it around between shots: near/
far, left/right, different lighting and angles. A few hundred varied frames
makes a solid model.

    python capture_frames.py

Auto-saves a frame every ~0.7 s to real_poop/. Keys:
    SPACE  save one now            a  toggle auto-capture on/off
    q      quit
Then:  python gather_images.py  ->  zip sam_images.zip  ->  sam_autolabel.py
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
import time
from pathlib import Path

import cv2

from trim_detector import build_pipeline
from camera_session import CameraSession

from paths import DATA
OUT = DATA / 'real_poop'
SAVE_INTERVAL_S = 0.7
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    OUT.mkdir(exist_ok=True)
    cam = CameraSession(build_pipeline)
    existing = len(list(OUT.glob('*.jpg')))
    saved, auto, last = 0, True, 0.0
    print(f"saving to {OUT}  (already have {existing}). "
          "Move the object around; q to stop.")
    win = 'Capture frames — SPACE save · a auto · q quit'
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
            do_save = False
            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (18, 18, 18), -1)
            cv2.putText(disp, f"saved this run: {saved}   total: {existing + saved}"
                        f"   auto: {'ON' if auto else 'off'}", (8, 20),
                        _FONT, 0.55, (0, 255, 255), 1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                break
            if k == ord('a'):
                auto = not auto
            if k == ord(' '):
                do_save = True
            if auto and now - last >= SAVE_INTERVAL_S:
                do_save = True
            if do_save:
                fn = OUT / f'cap_{int(now * 1000)}.jpg'
                cv2.imwrite(str(fn), frame)   # RAW frame, no overlay
                saved += 1
                last = now
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        print(f"done. captured {saved} frame(s); {existing + saved} total in {OUT}")
        print("next:  python gather_images.py  (then zip + sam_autolabel.py)")


if __name__ == '__main__':
    main()
