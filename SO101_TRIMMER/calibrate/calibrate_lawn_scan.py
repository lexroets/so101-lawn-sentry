"""Scan-and-click lawn calibration — the whole lawn need NOT fit in frame.

The idea
--------
You know the lawn is 3.5 x 2.1 m. Those dimensions ARE the scale reference, so
no tape square or extra object is needed. The only difficulty is that all four
corners are never visible at once.

Ground tracking solves that: it estimates the transform G from the CURRENT view
back to the FIRST view. So a corner clicked after rotating 40 degrees can be
expressed in the first view's coordinate frame — even if it now lies outside
the physical image. Collect all four that way and you have four points in one
common frame, which with the known rectangle gives the homography directly:

    p0 = G · p_clicked                    (bring each click into frame 0)
    H  = perspectiveTransform(p0 -> [[0,0],[W,0],[W,L],[0,L]])

Nothing depends on the previous calibration being right — the scale comes from
your measured dimensions.

The RGB camera helps too: clicks snap to the corner of the detected green
region when one is nearby, so you do not have to hit the boundary precisely.

    python calibrate_lawn_scan.py

    click   capture a corner (snaps to the grass edge if close)
    n       toggle snapping on/off
    u       undo · r restart tracking · s save · q quit

Click the corners IN ORDER going around the lawn — either direction, but do
not jump about. Rotate slowly so tracking stays locked.
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

from trim_detector import build_pipeline
from camera_session import CameraSession
from ground_track import GroundTracker
import lawn_shape as ls

from paths import DATA, ensure as _ensure
_ensure()
HERE = DATA
OUT = DATA / 'lawn_calibration.json'
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Lawn calibration — scan round and click the 4 corners'
NAMES = ['CORNER 1', 'CORNER 2', 'CORNER 3', 'CORNER 4']
COLS = [(0, 255, 255), (0, 220, 255), (0, 165, 255), (80, 255, 80)]
SNAP_PX = 55            # snap radius to the grass boundary


def ask(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def grass_corners(frame):
    """Corner-ish points on the boundary of the green region (RGB assist)."""
    try:
        m = ls.green_mask(frame)
        if m is None or not m.any():
            return None, None
        cs, _ = cv2.findContours(m.astype(np.uint8) * 255,
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None, None
        c = max(cs, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 12.0, True).reshape(-1, 2)
        return approx, c.reshape(-1, 2)
    except Exception:
        return None, None


def snap(pt, corners, contour):
    """Pull a click onto the grass boundary: prefer a corner, else the edge."""
    if corners is not None and len(corners):
        d = np.linalg.norm(corners - np.asarray(pt, np.float32), axis=1)
        i = int(np.argmin(d))
        if d[i] < SNAP_PX:
            return (float(corners[i][0]), float(corners[i][1])), 'corner'
    if contour is not None and len(contour):
        d = np.linalg.norm(contour - np.asarray(pt, np.float32), axis=1)
        i = int(np.argmin(d))
        if d[i] < SNAP_PX * 0.6:
            return (float(contour[i][0]), float(contour[i][1])), 'edge'
    return (float(pt[0]), float(pt[1])), 'raw'


def order_for_rect(p0):
    """Order 4 reference-frame points so they match [FL, FR, NR, NL].

    'Near' means lower in the reference image (larger y), which is nearer the
    camera; 'far' is higher up. Within each pair, sort by x.
    """
    p = np.asarray(p0, np.float64)
    idx = np.argsort(p[:, 1])            # small y = far
    far, near = p[idx[:2]], p[idx[2:]]
    far = far[np.argsort(far[:, 0])]     # left, right
    near = near[np.argsort(near[:, 0])]
    return np.array([far[0], far[1], near[1], near[0]], np.float32)


def solve(p0_pts, W, L):
    """Homography from reference-frame points to lawn metres."""
    src = order_for_rect(p0_pts)
    dst = np.array([[0, L], [W, L], [W, 0], [0, 0]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H, src


def check(H, src, W, L):
    """Reprojection error of the four corners, in cm."""
    got = cv2.perspectiveTransform(src.reshape(1, -1, 2), H).reshape(-1, 2)
    want = np.array([[0, L], [W, L], [W, 0], [0, 0]], np.float32)
    return float(np.max(np.linalg.norm(got - want, axis=1))) * 100


def main():
    print('\n' + '=' * 62)
    print('SCAN-AND-CLICK LAWN CALIBRATION')
    print('=' * 62)
    print('\nThe lawn does not need to fit in one view. Rotate to each corner')
    print('and click it; tracking stitches the views together.\n')
    W = ask('Lawn WIDTH  (metres)', 3.5)
    L = ask('Lawn LENGTH (metres)', 2.1)
    print(f'\nlawn: {W} x {L} m — click the 4 corners in order around it.\n')

    cam = CameraSession(build_pipeline)
    trk = GroundTracker(np.eye(3))       # scale comes from W,L — H0 irrelevant
    pts0 = []                            # corners in the REFERENCE frame
    labels = []
    click = {'pt': None}
    use_snap = True
    frame = None

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
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

            trk.update(frame)
            gcor, gcon = grass_corners(frame) if use_snap else (None, None)

            if click['pt'] is not None and len(pts0) < 4:
                p, how = snap(click['pt'], gcor, gcon) if use_snap \
                    else ((float(click['pt'][0]), float(click['pt'][1])), 'raw')
                # bring the click into the FIRST view's frame
                q = cv2.perspectiveTransform(
                    np.array([[[p[0], p[1]]]], np.float32), trk.G)[0, 0]
                pts0.append([float(q[0]), float(q[1])])
                labels.append(how)
                print(f'  {NAMES[len(pts0)-1]}  image({p[0]:.0f},{p[1]:.0f}) '
                      f'[{how}]  ->  reference frame ({q[0]:.0f},{q[1]:.0f})')
            click['pt'] = None

            disp = frame.copy()
            h, w = disp.shape[:2]
            # show the detected grass boundary we can snap to
            if gcon is not None and len(gcon) > 2:
                cv2.polylines(disp, [gcon.reshape(-1, 1, 2).astype(np.int32)],
                              True, (60, 180, 60), 1, cv2.LINE_AA)
            if gcor is not None:
                for c in gcor:
                    cv2.circle(disp, (int(c[0]), int(c[1])), 5, (60, 220, 60), 1)

            # re-project captured corners into THIS view so you can see them
            try:
                Ginv = np.linalg.inv(trk.G)
                for i, q in enumerate(pts0):
                    p = cv2.perspectiveTransform(
                        np.array([[[q[0], q[1]]]], np.float32), Ginv)[0, 0]
                    x, y = int(p[0]), int(p[1])
                    if -600 < x < w + 600 and -600 < y < h + 600:
                        cv2.drawMarker(disp, (x, y), COLS[i], cv2.MARKER_CROSS, 28, 2)
                        cv2.putText(disp, str(i + 1), (x + 12, y - 10),
                                    FONT, 0.6, COLS[i], 2)
            except np.linalg.LinAlgError:
                pass

            s = trk.status()
            cv2.rectangle(disp, (0, 0), (w, 50), (16, 16, 16), -1)
            nxt = NAMES[len(pts0)] if len(pts0) < 4 else 'press S to SAVE'
            cv2.putText(disp, f'{len(pts0)}/4   next: {nxt}', (8, 19), FONT, 0.52,
                        (0, 255, 255) if len(pts0) < 4 else (60, 255, 60), 1)
            col = (0, 220, 90) if not s['lost'] else (0, 120, 255)
            cv2.putText(disp, f"tracking {'LOST — press r' if s['lost'] else 'OK'}"
                        f"  feats {s['tracked']}  snap {'ON' if use_snap else 'off'}"
                        f"   u undo · n snap · r restart · s save · q quit",
                        (8, 38), FONT, 0.44, col, 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                print('cancelled — nothing saved'); break
            elif k == ord('u') and pts0:
                pts0.pop(); labels.pop(); print('  undid last corner')
            elif k == ord('n'):
                use_snap = not use_snap
                print(f'  snapping {"on" if use_snap else "off"}')
            elif k == ord('r'):
                trk.anchor(); pts0.clear(); labels.clear()
                print('  tracking restarted — corners cleared')
            elif k == ord('s'):
                if len(pts0) != 4:
                    print('  need 4 corners first'); continue
                try:
                    H, src = solve(pts0, W, L)
                except cv2.error:
                    print('  those 4 points are degenerate — press r and redo')
                    continue
                if not np.all(np.isfinite(H)):
                    print('  bad geometry — press r and redo'); continue
                err = check(H, src, W, L)
                print(f'\n  corner reprojection error: {err:.1f} cm')
                # sanity: the reference-frame quad must be convex and sizeable
                area = abs(cv2.contourArea(src))
                if area < 2000:
                    print('  the 4 corners are nearly collinear — redo'); continue
                data = {
                    'version': 4,
                    'calibration_type': 'scan_and_click',
                    'lawn_width_m': W,
                    'lawn_length_m': L,
                    'image_points_tl_tr_br_bl': [[float(a), float(b)] for a, b in src],
                    'homography_image_to_lawn': H.astype(float).tolist(),
                    'reference_frame_points': pts0,
                    'snap_modes': labels,
                    'corner_error_cm': round(err, 2),
                    'camera_width_px': int(frame.shape[1]),
                    'camera_height_px': int(frame.shape[0]),
                }
                OUT.write_text(json.dumps(data, indent=2))
                print(f'  SAVED {OUT.name}  ({W} x {L} m)')
                print('\n  IMPORTANT: this calibration is tied to the camera pose')
                print('  at the START of this scan. Do not move the camera before')
                print('  running launch.py.')
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
