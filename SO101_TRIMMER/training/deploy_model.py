"""Install + validate a freshly trained poop_yolo.onnx, then confirm demo-ready.

Run this ONCE after training. It finds the model (even if it's still sitting in
Downloads), checks it really loads and runs, measures speed, tries it on any
sample images you have, and prints a clear GO / NO-GO for the demo.

    python deploy_model.py                 # auto-finds poop_yolo.onnx
    python deploy_model.py ~/Downloads/poop_yolo.onnx

Nothing else needs editing — poop_config.py is set to 'auto', so the robot uses
the trained model as soon as it is in this folder.
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
import sys
import glob
import shutil
import time

import numpy as np

from paths import DATA, ensure as _ensure
_ensure()
HERE = str(DATA)
TARGET = str(DATA / 'poop_yolo.onnx')
OK, BAD = '  [OK]  ', '  [!!]  '


def find_model(argv):
    if len(argv) > 1 and os.path.isfile(argv[1]):
        return argv[1]
    here = TARGET
    if os.path.isfile(here) and os.path.getsize(here) > 10_000:
        return here
    # look in the usual download spots, newest first
    cands = []
    for d in (os.path.expanduser('~/Downloads'), os.path.expanduser('~/Desktop'),
              os.path.expanduser('~')):
        cands += glob.glob(os.path.join(d, 'poop_yolo*.onnx'))
    cands = [c for c in cands if os.path.getsize(c) > 10_000]
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def main():
    print('\n=== DEPLOY POOP MODEL ===\n')

    src = find_model(sys.argv)
    if not src:
        print(BAD + 'No poop_yolo.onnx found.')
        print('        Put the file in this folder, or run:')
        print('        python deploy_model.py ~/Downloads/poop_yolo.onnx')
        return 1
    mb = os.path.getsize(src) / 1e6
    print(f'{OK}found model: {src}  ({mb:.1f} MB)')
    if mb < 3:
        print(BAD + 'That is suspiciously small for YOLOv8n (expect ~10-13 MB).')

    if os.path.abspath(src) != os.path.abspath(TARGET):
        # SAFETY: never clobber a good model with a smaller/suspect one.
        if os.path.isfile(TARGET) and os.path.getsize(TARGET) > 3e6 \
                and os.path.getsize(src) < os.path.getsize(TARGET) * 0.5:
            print(BAD + 'refusing to overwrite: the model already installed is '
                        'much larger than the one you gave me.')
            print(f'        installed {os.path.getsize(TARGET)/1e6:.1f} MB  vs  '
                  f'new {mb:.1f} MB')
            print('        If you really mean it, delete poop_yolo.onnx first.')
            return 1
        if os.path.isfile(TARGET) and os.path.getsize(TARGET) > 3e6:
            # keep timestamped backups so a good model is never lost
            bak = TARGET + '.' + time.strftime('%Y%m%d_%H%M%S') + '.bak'
            shutil.copy2(TARGET, bak)
            print(f'{OK}backed up previous model -> {os.path.basename(bak)}')
        shutil.copy2(src, TARGET)
        print(f'{OK}installed -> {TARGET}')

    # ---- does it load? ----
    try:
        from poop_yolo import PoopYolo
    except Exception as e:
        print(BAD + f'cannot import runner: {e}')
        return 1
    det = PoopYolo()
    if not det.ready:
        print(BAD + 'model did NOT load — see the message above. Re-download it.')
        return 1
    print(f'{OK}model loads via {det.backend}')

    # ---- does it run, and how fast? ----
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 1] = 110
    try:
        det.infer(frame)                       # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            det.infer(frame)
        ms = (time.perf_counter() - t0) / 5 * 1000
    except Exception as e:
        print(BAD + f'inference failed: {e}')
        return 1
    fps = 1000 / max(ms, 1e-6)
    print(f'{OK}inference {ms:.0f} ms/frame ({fps:.0f} fps)')
    if fps < 5:
        print('        slow-ish; consider `pip install onnxruntime` for a speed-up.')

    # ---- wired into the detector? ----
    try:
        import poop_config as cfg
        import poop_detector
        d = poop_detector.PoopDetector()
        if d.yolo is None:
            print(BAD + 'PoopDetector did not pick it up — check POOP_USE_YOLO.')
            return 1
        print(f'{OK}PoopDetector is using the trained model')
    except Exception as e:
        print(BAD + f'detector wiring failed: {e}')
        return 1

    # ---- try it on any real images we have ----
    imgs = []
    for folder in ('real_poop', 'sam_images', 'poop_photos'):
        p = os.path.join(HERE, folder)
        if os.path.isdir(p):
            imgs += [f for f in glob.glob(os.path.join(p, '*'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if imgs:
        import cv2
        hits = 0
        for f in imgs[:25]:
            im = cv2.imread(f)
            if im is not None and det.infer(im):
                hits += 1
        print(f'{OK}sample check: detections on {hits}/{min(len(imgs),25)} of your images')
        print('        (run  python test_poop_model.py  to SEE the boxes)')
    else:
        print('  [--]  no sample images to try — that is fine.')

    # ---- demo prerequisites ----
    print('\n--- demo prerequisites ---')
    cal = os.path.join(HERE, 'lawn_calibration.json')
    print((OK if os.path.isfile(cal) else BAD) +
          ('lawn calibration present' if os.path.isfile(cal)
           else 'NO lawn_calibration.json — run: python calibrate_lawn_map.py'))
    try:
        import streamlit  # noqa: F401
        print(OK + 'streamlit installed (web dashboard will open)')
    except ImportError:
        print(BAD + 'streamlit missing — run: pip install streamlit pandas')
    try:
        import depthai  # noqa: F401
        print(OK + 'depthai installed (OAK-D camera)')
    except ImportError:
        print(BAD + 'depthai missing — run: pip install depthai')

    print('\n=== READY ===')
    print('Start the demo with:   python launch.py')
    print('Tuning: POOP_YOLO_CONF in poop_config.py '
          f'(currently {cfg.POOP_YOLO_CONF}) — raise to cut false hits, '
          'lower to catch more.')
    print('Roll back to the colour demo: set POOP_USE_YOLO = False\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
