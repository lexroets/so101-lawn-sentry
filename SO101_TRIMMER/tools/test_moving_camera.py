"""Watch ground tracking hold a world position while you move the camera.

Read-only test — changes nothing. Point the camera at the lawn, click a spot,
then pan / tilt / walk the camera around. The marker should STAY on that patch
of grass. The printed world coordinate should stay roughly constant; with the
old fixed-homography maths it would wander by tens of centimetres.

    python test_moving_camera.py

    click        anchor a world point
    r            reset the tracker (kills accumulated drift)
    c            clear the anchor
    q            quit
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
import math
from pathlib import Path

import cv2
import numpy as np

from trim_detector import build_pipeline
from camera_session import CameraSession
from ground_track import GroundTracker

from paths import DATA
HERE = DATA
_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Ground tracking test — click a spot, then move the camera'


def main():
    cal_p = HERE / 'lawn_calibration.json'
    if not cal_p.is_file():
        print('Run calibrate_lawn_map.py first.')
        return
    cal = json.loads(cal_p.read_text())
    H0 = np.asarray(cal['homography_image_to_lawn'], np.float64)
    quad = np.asarray(cal['image_points_tl_tr_br_bl'], np.float32)

    cam = CameraSession(build_pipeline)
    trk = GroundTracker(H0, roi_poly=quad)
    anchor_world = None          # the world point we locked on to
    click = {'pt': None}
    hist = []

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    print('Click a distinctive spot on the grass, then move the camera around.')
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
            W = trk.update(frame)
            disp = frame.copy()

            if click['pt'] is not None:
                anchor_world = trk.to_world(*click['pt'])
                hist = []
                print(f'anchored at world ({anchor_world[0]:.3f}, '
                      f'{anchor_world[1]:.3f}) m')
                click['pt'] = None

            if anchor_world is not None:
                # project the fixed WORLD point back into the current image
                try:
                    Winv = np.linalg.inv(W)
                    p = cv2.perspectiveTransform(
                        np.array([[[anchor_world[0], anchor_world[1]]]], np.float32),
                        Winv)[0, 0]
                    u, v = int(p[0]), int(p[1])
                    if -200 < u < 840 and -200 < v < 680:
                        cv2.drawMarker(disp, (u, v), (33, 224, 212),
                                       cv2.MARKER_CROSS, 34, 2)
                        cv2.circle(disp, (u, v), 20, (33, 224, 212), 1)
                        cv2.putText(disp, 'ANCHOR', (u + 24, v),
                                    _FONT, 0.45, (33, 224, 212), 1)
                    # where the OLD maths would have put it, for comparison
                    p0 = cv2.perspectiveTransform(
                        np.array([[[anchor_world[0], anchor_world[1]]]], np.float32),
                        np.linalg.inv(H0))[0, 0]
                    cv2.drawMarker(disp, (int(p0[0]), int(p0[1])), (0, 120, 255),
                                   cv2.MARKER_TILTED_CROSS, 24, 2)
                    cv2.putText(disp, 'fixed-H', (int(p0[0]) + 16, int(p0[1]) + 16),
                                _FONT, 0.4, (0, 120, 255), 1)
                    err = math.hypot(p[0] - p0[0], p[1] - p0[1])
                    hist.append(err)
                except np.linalg.LinAlgError:
                    pass

            s = trk.status()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (16, 16, 16), -1)
            col = (0, 220, 90) if not s['lost'] else (0, 120, 255)
            cv2.putText(disp, f"tracking {'LOST' if s['lost'] else 'OK'}   "
                        f"features {s['tracked']}   moved {s['moved_m']:.2f} m",
                        (8, 20), _FONT, 0.5, col, 1)
            if hist:
                cv2.putText(disp, f"cyan = tracked (correct) | orange = old fixed maths"
                            f"   divergence {hist[-1]:.0f} px",
                            (8, 42), _FONT, 0.44, (200, 200, 200), 1)
            else:
                cv2.putText(disp, 'click a spot on the grass to anchor it',
                            (8, 42), _FONT, 0.44, (200, 200, 200), 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                break
            elif k == ord('r'):
                trk.anchor()
                print('tracker reset (drift cleared)')
            elif k == ord('c'):
                anchor_world = None
                hist = []
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        if hist:
            print(f'\nmax divergence between tracked and fixed-homography: '
                  f'{max(hist):.0f} px')
            print('That gap is the error the old maths introduces when the '
                  'camera moves.')
        print('status:', trk.status())


if __name__ == '__main__':
    main()
