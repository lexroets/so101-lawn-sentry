"""Define the TRUE lawn edges — even when the camera can't see them all at once.

Why this exists
---------------
calibrate_lawn_map.py asks you to click four corners in ONE view. If the camera
sits in front of the lawn, or the lawn is bigger than the field of view, those
four clicks are not the real lawn corners — they are just the bit you could
see. Everything then works in a frame that does not match the actual lawn, so
the outline never sits on the real edges.

How this fixes it
-----------------
Ground tracking keeps a single consistent world frame while the camera moves.
So you can rotate to each corner IN TURN and click it as it comes into view.
The four clicks land in the same world frame even though they were never
visible together.

It then:
  * fits a clean rectangle through the four corners,
  * builds a transform that makes the lawn axis-aligned, 0,0 at the near-left,
  * saves lawn_extent.json, which scan_lawn.py picks up automatically.

    python define_lawn.py

    click       capture the corner under the cursor
    u           undo the last corner
    r           reset tracking (do this if it says LOST)
    s           save (needs 4 corners)
    q           quit without saving

Tip: go round the lawn in order (either direction) and keep the motion slow
and steady so tracking stays locked.
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

from paths import DATA, ensure as _ensure
_ensure()
HERE = DATA
CAL = HERE / 'lawn_calibration.json'
OUT = HERE / 'lawn_extent.json'
_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Define lawn — rotate to each corner and click it'
NAMES = ['CORNER 1', 'CORNER 2', 'CORNER 3', 'CORNER 4']


# A click is only trustworthy near the middle of the frame. Out at the edges
# the ground homography is EXTRAPOLATING far beyond the calibrated region and
# world coordinates blow up (a 3.4 m far edge coming out as a 6.8 m near edge).
CAPTURE_BOX = (0.22, 0.30, 0.78, 0.92)      # l, t, r, b as fractions of frame


def click_quality(u, v, w, h):
    """'good' | 'edge' | 'bad' for a click at (u,v) in a w x h frame."""
    l, t, r, b = CAPTURE_BOX
    if l * w <= u <= r * w and t * h <= v <= b * h:
        return 'good'
    if 0.10 * w <= u <= 0.90 * w and 0.20 * h <= v <= 0.98 * h:
        return 'edge'
    return 'bad'


def validate_rect(corners, tol=0.25):
    """Sanity-check 4 surveyed corners. Returns (ok, message, sides)."""
    p = [np.asarray(c, float) for c in order_corners(corners)]
    sides = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    a, b, c, d = sides                       # opposite pairs are (a,c) and (b,d)
    def mism(x, y):
        return abs(x - y) / max(x, y, 1e-6)
    m1, m2 = mism(a, c), mism(b, d)
    diag1 = float(np.linalg.norm(p[2] - p[0]))
    diag2 = float(np.linalg.norm(p[3] - p[1]))
    m3 = mism(diag1, diag2)
    worst = max(m1, m2, m3)
    if worst > tol:
        return (False,
                f'opposite sides/diagonals disagree by {worst*100:.0f}% — '
                f'sides {a:.2f}/{c:.2f} and {b:.2f}/{d:.2f} m. '
                f'Re-capture with each corner near the CENTRE of the frame.',
                sides)
    return True, f'consistent (worst mismatch {worst*100:.0f}%)', sides


def order_corners(pts):
    """Put 4 world points into a consistent order (near-left first, CCW)."""
    p = np.asarray(pts, np.float64)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    # start from the corner closest to the origin side (smallest y, then x)
    start = int(np.argmin(p[:, 1] * 1000 + p[:, 0]))
    return np.roll(p, -start, axis=0)


def fit_rect(pts):
    """Best-fit rectangle through 4 world corners -> (corners, W, L, angle)."""
    p = np.asarray(pts, np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(p)
    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    return box.astype(np.float64), float(w), float(h), float(ang)


def build_transform(box, W, L):
    """Rigid transform mapping the fitted rectangle onto (0,0)-(W,L).

    After this, lawn coordinates are simply metres from the near-left corner,
    which is what the dashboard and the mower both want.
    """
    src = order_corners(box).astype(np.float32)
    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32)
    # full perspective is overkill for two rectangles; affine is exact here
    T = cv2.getPerspectiveTransform(src, dst)
    return T


def main():
    if not CAL.is_file():
        print('Run calibrate_lawn_map.py first (it fixes the SCALE; this tool '
              'fixes the EXTENT).')
        return
    cal = json.loads(CAL.read_text())
    H0 = np.asarray(cal['homography_image_to_lawn'], np.float64)
    quad = np.asarray(cal['image_points_tl_tr_br_bl'], np.float32)

    cam = CameraSession(build_pipeline)
    trk = GroundTracker(H0, roi_poly=quad)
    corners = []          # world coords
    shots = []            # image coords at capture time, for on-screen marks
    click = {'pt': None}

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    print('Rotate to each lawn corner and click it. Slow, steady motion.')
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
            W_ = trk.update(frame)
            disp = frame.copy()

            fh, fw = frame.shape[:2]
            if click['pt'] is not None and len(corners) < 4:
                u, v = click['pt']
                q = click_quality(u, v, fw, fh)
                if q == 'bad':
                    print('  IGNORED — too close to the frame edge. Rotate so '
                          'the corner sits inside the box, then click.')
                else:
                    wx, wy = trk.to_world(u, v)
                    corners.append([wx, wy])
                    shots.append((u, v, len(corners)))
                    warn = '   <-- near the edge, may be imprecise' if q == 'edge' else ''
                    print(f'  {NAMES[len(corners)-1]} at world '
                          f'({wx:.3f}, {wy:.3f}) m{warn}')
            click['pt'] = None

            # re-project captured corners into the CURRENT view so you can see
            # them stay put on the grass as you rotate
            try:
                Winv = np.linalg.inv(W_)
                for i, (wx, wy) in enumerate(corners):
                    p = cv2.perspectiveTransform(
                        np.array([[[wx, wy]]], np.float32), Winv)[0, 0]
                    x, y = int(p[0]), int(p[1])
                    if -400 < x < 1040 and -400 < y < 880:
                        cv2.drawMarker(disp, (x, y), (33, 224, 212),
                                       cv2.MARKER_CROSS, 26, 2)
                        cv2.putText(disp, str(i + 1), (x + 12, y - 8),
                                    _FONT, 0.55, (33, 224, 212), 2)
                if len(corners) >= 2:
                    pts = []
                    for wx, wy in corners:
                        p = cv2.perspectiveTransform(
                            np.array([[[wx, wy]]], np.float32), Winv)[0, 0]
                        pts.append([int(p[0]), int(p[1])])
                    cv2.polylines(disp, [np.asarray(pts, np.int32)],
                                  len(corners) == 4, (33, 224, 212), 2)
            except np.linalg.LinAlgError:
                pass

            # draw the trustworthy capture zone — put each corner IN HERE
            l, t, r, b = CAPTURE_BOX
            cv2.rectangle(disp, (int(l*fw), int(t*fh)), (int(r*fw), int(b*fh)),
                          (90, 190, 90), 1, cv2.LINE_AA)
            cv2.putText(disp, 'put the corner in this box before clicking',
                        (int(l*fw)+6, int(t*fh)-8), _FONT, 0.42, (90, 190, 90), 1)

            s = trk.status()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (16, 16, 16), -1)
            nxt = NAMES[len(corners)] if len(corners) < 4 else 'press S to SAVE'
            cv2.putText(disp, f'{len(corners)}/4 corners   next: {nxt}',
                        (8, 20), _FONT, 0.52,
                        (33, 224, 212) if len(corners) < 4 else (60, 255, 60), 1)
            col = (0, 220, 90) if not s['lost'] else (0, 120, 255)
            cv2.putText(disp, f"tracking {'LOST — press r' if s['lost'] else 'OK'}"
                        f"   feats {s['tracked']}   u undo · s save · q quit",
                        (8, 42), _FONT, 0.45, col, 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                print('cancelled'); break
            elif k == ord('u') and corners:
                corners.pop(); shots.pop(); print('  undid last corner')
            elif k == ord('r'):
                trk.anchor(); print('  tracking reset')
            elif k == ord('s'):
                if len(corners) != 4:
                    print('  need 4 corners first'); continue
                ok, msg, sides = validate_rect(corners)
                print(f'\n  geometry check: {msg}')
                print(f'  side lengths  : ' +
                      ', '.join(f'{x:.2f}' for x in sides) + ' m')
                if not ok:
                    print('  NOT SAVED. Press u four times and re-capture, '
                          'keeping each corner inside the green box.\n')
                    continue
                box, w, h, ang = fit_rect(corners)
                Wm, Lm = max(w, h), min(w, h)
                T = build_transform(box, Wm, Lm)
                # residual: how far the clicks sit from the fitted rectangle
                oc = order_corners(box)
                res = [float(np.min(np.linalg.norm(oc - np.array(c), axis=1)))
                       for c in corners]
                data = {
                    'corners_world': [[float(x), float(y)] for x, y in corners],
                    'rect_world': [[float(x), float(y)] for x, y in oc],
                    'lawn_width_m': round(Wm, 3),
                    'lawn_length_m': round(Lm, 3),
                    'angle_deg': round(ang, 2),
                    'to_lawn_frame': T.tolist(),
                    'fit_residual_m': round(max(res), 3),
                }
                OUT.write_text(json.dumps(data, indent=2))
                print(f'\nSAVED {OUT.name}')
                print(f'  lawn measured  : {Wm:.2f} x {Lm:.2f} m')
                print(f'  corner fit err : {max(res)*100:.1f} cm '
                      f'({"good" if max(res)<0.15 else "loose — consider redoing"})')
                print('  scan_lawn.py will use this automatically.')
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
