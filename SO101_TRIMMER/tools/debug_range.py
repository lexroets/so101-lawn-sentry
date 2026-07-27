"""Side-by-side: what the model sees WITHOUT tiling vs WITH tiling.

Read-only diagnostic — it does not touch your demo settings or the model file.
Use it to decide whether to turn POOP_YOLO_TILES on before the demo.

    python debug_range.py

    CYAN  box  = found normally (POOP_YOLO_TILES = 0 behaviour)
    LIME  box  = found ONLY because of tiling  <- the far-end wins
    Each box shows confidence and estimated range on the lawn.

Keys:  [ / ]  lower / raise the confidence threshold live
       s      save the current annotated frame
       q      quit
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
import json
import math
from pathlib import Path

import cv2
import numpy as np

import poop_config as cfg
from trim_detector import build_pipeline
from camera_session import CameraSession
from poop_yolo import PoopYolo

from paths import DATA, OUTPUT
HERE = DATA
_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Range diagnostic — cyan = normal, lime = only found by tiling'


def load_homography():
    p = HERE / 'lawn_calibration.json'
    if not p.is_file():
        return None, 0, 0
    c = json.loads(p.read_text())
    return (np.asarray(c['homography_image_to_lawn'], np.float64),
            float(c['lawn_width_m']), float(c['lawn_length_m']))


def ground_xy(H, u, v):
    if H is None:
        return None
    p = cv2.perspectiveTransform(np.array([[[float(u), float(v)]]], np.float32), H)[0, 0]
    return float(p[0]), float(p[1])


def same(a, b, tol=45):
    return abs(a['u'] - b['u']) < tol and abs(a['v'] - b['v']) < tol


def main():
    det = PoopYolo()
    if not det.ready:
        print('No trained model loaded — nothing to diagnose.')
        return
    H, LW, LL = load_homography()
    cam = CameraSession(build_pipeline)
    conf0 = cfg.POOP_YOLO_CONF
    n = 0
    gained_total = plain_total = frames = 0
    print(f'threshold {cfg.POOP_YOLO_CONF}  ·  model @ {det.imgsz}px')
    print('Put a target at the FAR end of the lawn and watch the counts.\n')
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

            plain = det.infer(frame, tiles=0)
            tiled = det.infer(frame, tiles=2)
            extra = [t for t in tiled if not any(same(t, p) for p in plain)]

            frames += 1
            plain_total += len(plain)
            gained_total += len(extra)

            disp = frame.copy()
            for d, col, tag in ([(d, (255, 220, 0), '') for d in plain]
                                + [(d, (33, 224, 33), '+') for d in extra]):
                x, y, w, h = d['bbox']
                cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                label = f"{tag}{d['conf']:.2f}"
                xy = ground_xy(H, d['u'], d['v'])
                if xy:
                    rng = math.hypot(xy[0] - LW / 2, xy[1])
                    label += f"  {rng:.1f}m"
                cv2.putText(disp, label, (x, max(14, y - 6)), _FONT, 0.45, col, 1)

            hh, ww = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (ww, 46), (18, 18, 18), -1)
            cv2.putText(disp, f"normal {len(plain)}    tiling adds {len(extra)}"
                        f"    total {len(tiled)}    thr {cfg.POOP_YOLO_CONF:.2f}",
                        (8, 19), _FONT, 0.5, (0, 255, 255), 1)
            avg = gained_total / max(1, frames)
            cv2.putText(disp, f"avg extra per frame from tiling: {avg:.2f}"
                        f"   ([ ] threshold · s save · q quit)",
                        (8, 38), _FONT, 0.45, (33, 224, 33), 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                break
            elif k == ord('['):
                cfg.POOP_YOLO_CONF = max(0.05, cfg.POOP_YOLO_CONF - 0.05)
            elif k == ord(']'):
                cfg.POOP_YOLO_CONF = min(0.95, cfg.POOP_YOLO_CONF + 0.05)
            elif k == ord('s'):
                fn = OUTPUT / f'range_debug_{n:02d}.png'
                n += 1
                cv2.imwrite(str(fn), disp)
                print('saved', fn)
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        cfg.POOP_YOLO_CONF = conf0
        print(f'\nover {frames} frames:')
        print(f'  normal detections     : {plain_total}')
        print(f'  extra found by tiling : {gained_total}')
        if gained_total > plain_total * 0.15:
            print('\n  -> tiling is finding a meaningful number of extra targets.')
            print('     Set POOP_YOLO_TILES = 2 in poop_config.py')
        elif gained_total:
            print('\n  -> tiling helps a little. Worth enabling if the far end matters.')
        else:
            print('\n  -> tiling found nothing extra. The far targets are likely too')
            print('     small even magnified — move the camera closer/lower, or use')
            print('     larger targets for the demo.')


if __name__ == '__main__':
    main()
