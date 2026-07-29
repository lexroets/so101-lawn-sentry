"""Calibrate the ground plane from a SMALL known rectangle — no need to see
the whole lawn.

The problem with calibrate_lawn_map.py
--------------------------------------
It asks you to click the four lawn corners in one view. If the lawn does not
fit in frame you physically cannot do that, and clicking "roughly the lawn"
produces a homography that is subtly wrong. Everything downstream — poop
coordinates, the map outline, the surveyed extent — inherits that error.

What this does instead
----------------------
A ground-plane homography needs ANY rectangle of known size lying flat on the
ground. It does not have to be the lawn. So:

  1. Put a measured rectangle on the grass — a square of tape, a mat, a box,
     two tape measures. 1.0 x 1.0 m is plenty; bigger is better as long as it
     is FULLY VISIBLE and reasonably central.
  2. Type its real width and length.
  3. Click its four corners.
  4. Check the bird's-eye inset: the rectangle must appear square/rectangular
     and fill the yellow box exactly.

That gives a CORRECT ground plane. Afterwards run define_lawn.py to survey the
lawn's real corners by rotating to each one — that now works properly, because
the underlying geometry is right.

    python calibrate_ground.py

    click = corner (in order: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT)
    u undo · r reset · s save · q quit

Accuracy notes
--------------
* The reference MUST lie flat on the same ground plane as the lawn.
* Bigger reference = less error when extrapolating to the lawn edges. Use the
  largest rectangle that still fits comfortably in the frame.
* Measure it properly. A 2 cm error on a 1 m reference is a 2% scale error
  everywhere.
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
import json
from pathlib import Path

import cv2
import numpy as np

from trim_detector import build_pipeline
from camera_session import CameraSession

from paths import DATA, ensure as _ensure
_ensure()
HERE = DATA
OUT = DATA / 'lawn_calibration.json'
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Ground calibration — click the reference rectangle corners'
NAMES = ['1 FAR-LEFT', '2 FAR-RIGHT', '3 NEAR-RIGHT', '4 NEAR-LEFT']
COLS = [(0, 255, 255), (0, 220, 255), (0, 165, 255), (80, 255, 80)]


def ask(prompt):
    while True:
        try:
            v = float(input(prompt).strip())
            if v > 0:
                return v
        except ValueError:
            pass
        print('  enter a positive number in metres, e.g. 1.0')


def homography(points, w, l):
    src = np.asarray(points, np.float32)
    dst = np.asarray([[0, l], [w, l], [w, 0], [0, 0]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def main():
    print('\n' + '=' * 64)
    print('GROUND-PLANE CALIBRATION FROM A KNOWN RECTANGLE')
    print('=' * 64)
    print('\nPut a measured rectangle flat on the grass, fully inside the')
    print('camera view. Tape a square, or use a mat or box of known size.')
    print('Bigger is better, as long as you can see all four corners.\n')
    ref_w = ask('Reference WIDTH  (left-right, metres): ')
    ref_l = ask('Reference LENGTH (near-far,  metres): ')
    print(f'\nreference: {ref_w:.3f} x {ref_l:.3f} m')
    print('Click the corners: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT\n')

    try:
        cam = CameraSession(build_pipeline)
    except Exception as e:
        print('camera failed to start:', e)
        return

    pts = []
    click = {'pt': None}
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    frame = None
    try:
        while True:
            g = cam.get()
            if g is not None:
                frame = g['rgb'].getCvFrame()
            elif cam.crashed() and not cam.restart():
                break
            if frame is None:
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue

            if click['pt'] is not None and len(pts) < 4:
                pts.append([float(click['pt'][0]), float(click['pt'][1])])
                print(f'  {NAMES[len(pts)-1]} at pixel '
                      f'({pts[-1][0]:.0f}, {pts[-1][1]:.0f})')
            click['pt'] = None

            disp = frame.copy()
            h, w = disp.shape[:2]
            for i, p in enumerate(pts):
                cv2.circle(disp, (int(p[0]), int(p[1])), 7, COLS[i], -1, cv2.LINE_AA)
                cv2.circle(disp, (int(p[0]), int(p[1])), 12, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, NAMES[i], (int(p[0]) + 12, int(p[1]) - 8),
                            FONT, 0.5, COLS[i], 2, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.polylines(disp, [np.asarray(pts, np.int32)], len(pts) == 4,
                              (0, 255, 255), 2, cv2.LINE_AA)

            # bird's-eye verification with surrounding margin
            if len(pts) == 4:
                try:
                    H = homography(pts, ref_w, ref_l)
                    ppm = 110.0
                    mg = max(ref_w, ref_l) * 0.5
                    S = np.array([[ppm, 0, mg * ppm],
                                  [0, -ppm, (ref_l + mg) * ppm],
                                  [0, 0, 1]], np.float64)
                    ow = max(60, int((ref_w + 2 * mg) * ppm))
                    oh = max(60, int((ref_l + 2 * mg) * ppm))
                    bird = cv2.warpPerspective(frame, S @ H, (ow, oh))
                    cv2.rectangle(bird, (int(mg * ppm), int(mg * ppm)),
                                  (int((mg + ref_w) * ppm), int((mg + ref_l) * ppm)),
                                  (0, 255, 255), 2)
                    sc = min((w // 3) / ow, (h // 3) / oh, 1.0)
                    bird = cv2.resize(bird, (int(ow * sc), int(oh * sc)))
                    bh, bw = bird.shape[:2]
                    x0, y0 = w - bw - 10, h - bh - 10
                    disp[y0:y0 + bh, x0:x0 + bw] = bird
                    cv2.rectangle(disp, (x0 - 1, y0 - 1), (x0 + bw, y0 + bh),
                                  (0, 255, 255), 2)
                    cv2.putText(disp, 'CHECK: reference must fill the yellow box',
                                (x0 - 60, y0 - 6), FONT, 0.42, (0, 255, 255), 1,
                                cv2.LINE_AA)
                except Exception:
                    pass

            cv2.rectangle(disp, (0, 0), (w, 46), (16, 16, 16), -1)
            nxt = NAMES[len(pts)] if len(pts) < 4 else 'press S to SAVE'
            cv2.putText(disp, f'{len(pts)}/4   next: {nxt}', (8, 19), FONT, 0.52,
                        (0, 255, 255) if len(pts) < 4 else (60, 255, 60), 1)
            cv2.putText(disp, f'reference {ref_w:.2f} x {ref_l:.2f} m   '
                        'u undo · r reset · s save · q quit',
                        (8, 38), FONT, 0.45, (200, 200, 200), 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                print('cancelled — nothing saved'); break
            elif k == ord('u') and pts:
                pts.pop(); print('  undid last corner')
            elif k == ord('r'):
                pts = []; print('  reset')
            elif k == ord('s'):
                if len(pts) != 4:
                    print('  need 4 corners first'); continue
                if abs(cv2.contourArea(np.asarray(pts, np.float32))) < 500:
                    print('  those corners enclose almost no area — re-click')
                    continue
                H = homography(pts, ref_w, ref_l)
                if not np.all(np.isfinite(H)):
                    print('  degenerate corners — press r and try again'); continue
                data = {
                    'version': 3,
                    'calibration_type': 'reference_rectangle',
                    'reference_width_m': ref_w,
                    'reference_length_m': ref_l,
                    # kept for compatibility: the "lawn" starts as the reference
                    # until define_lawn.py surveys the real extent
                    'lawn_width_m': ref_w,
                    'lawn_length_m': ref_l,
                    'image_points_tl_tr_br_bl': [[float(a), float(b)] for a, b in pts],
                    'homography_image_to_lawn': H.astype(float).tolist(),
                    'camera_width_px': int(frame.shape[1]),
                    'camera_height_px': int(frame.shape[0]),
                }
                OUT.write_text(json.dumps(data, indent=2))
                cv2.imwrite(str(HERE / 'lawn_calibration_preview.jpg'), disp)
                print(f'\nSAVED {OUT.name}')
                print(f'  ground plane fixed from a {ref_w:.2f} x {ref_l:.2f} m reference')
                print('\nNEXT: python define_lawn.py')
                print('  rotate to each real lawn corner and click it — the world')
                print('  frame is now correct, so those corners will be accurate.')
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
