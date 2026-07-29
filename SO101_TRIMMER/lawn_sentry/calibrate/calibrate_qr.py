"""One-shot calibration from a printed marker. Place it, drag the lawn onto the
lawn, take the marker away.

The first version of this asked you to put the marker at a stated spot and then
believed you. When the camera could not see it there you moved it forward, and
everything downstream was quietly wrong - by roughly however far you moved it,
plus the perspective error from fitting eight homography parameters through one
small square.

Both of those are fixed here, and it is worth being clear about the split,
because it is what makes this repeatable:

    the MARKER supplies what you cannot eyeball
        the tilt of the ground and the metric scale - solved from its pose
        against the camera's known intrinsics, averaged over ~40 frames

    YOU supply what you can eyeball
        where the lawn actually is - by nudging the projected rectangle with
        the arrow keys until it sits on the real one

Neither is asked to guess at the thing it is bad at. Moving the marker is now
harmless: put it wherever the camera sees it clearly, big and sharp, and slide
the rectangle afterwards.

    python lawn_sentry/calibrate/calibrate_qr.py --print --paper a3     get the marker
    python lawn_sentry/calibrate/calibrate_qr.py --marker 0.25          calibrate

Keys
    arrows / WASD   slide the lawn 5 cm        - =   fine, 1 cm
    [ ]             rotate the lawn            f     flip it end for end
    r               re-average the marker      q     quit
    SPACE or ENTER  save
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
import argparse
import json
import math
import time
from datetime import datetime

import cv2
import numpy as np

import cal_store
from paths import DATA, ensure
from frame_source import FrameSource
from marker import printable_sheet, max_size_for, DEFAULT_SIZE_M, MarkerFinder
from marker_pose import (solve_pose, ground_from_pose, camera_from_H,
                         PoseAverager, MIN_SIDE_PX, GOOD_SIDE_PX)
from camera_intrinsics import get as get_K

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'QR calibration - place the marker, slide the lawn onto the lawn, save'

AVG_FRAMES = 40           # 40 halves the error again vs 10, and costs 2 seconds
NUDGE_M = 0.05
FINE_M = 0.01
ROT_DEG = 1.0


def ask(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    try:
        v = float(raw.replace(',', '.'))
        return v if v > 0 else default
    except ValueError:
        return default


def project(H, pts):
    """Lawn metres -> image pixels, or None if it lands somewhere absurd."""
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    p = cv2.perspectiveTransform(
        np.asarray(pts, np.float32).reshape(1, -1, 2), Hi.astype(np.float32))
    p = p.reshape(-1, 2)
    if not np.isfinite(p).all() or np.abs(p).max() > 1e5:
        return None
    return p


def draw_lawn(img, H, W, L):
    """The lawn as the calibration currently believes it, drawn over the view."""
    outline = project(H, [(0, 0), (W, 0), (W, L), (0, L)])
    if outline is None:
        return False
    for x in np.arange(0.5, W, 0.5):
        g = project(H, [(x, 0), (x, L)])
        if g is not None:
            thick = 2 if abs(x - round(x)) < 1e-6 else 1
            col = (70, 170, 70) if thick == 2 else (45, 110, 45)
            cv2.line(img, tuple(g[0].astype(int)), tuple(g[1].astype(int)),
                     col, thick, cv2.LINE_AA)
    for y in np.arange(0.5, L, 0.5):
        g = project(H, [(0, y), (W, y)])
        if g is not None:
            thick = 2 if abs(y - round(y)) < 1e-6 else 1
            col = (70, 170, 70) if thick == 2 else (45, 110, 45)
            cv2.line(img, tuple(g[0].astype(int)), tuple(g[1].astype(int)),
                     col, thick, cv2.LINE_AA)
    cv2.polylines(img, [outline.astype(np.int32)], True, (90, 255, 90), 2,
                  cv2.LINE_AA)
    # the near edge marked, so a lawn placed backwards is obvious
    cv2.line(img, tuple(outline[0].astype(int)), tuple(outline[1].astype(int)),
             (255, 220, 90), 3, cv2.LINE_AA)
    for pt, lab in zip(outline, ['0,0', 'W,0', 'W,L', '0,L']):
        cv2.circle(img, tuple(pt.astype(int)), 4, (255, 220, 90), -1)
        cv2.putText(img, lab, tuple((pt + [6, -6]).astype(int)), FONT, 0.4,
                    (255, 220, 90), 1, cv2.LINE_AA)
    return True


def go_to_rest():
    """Put the arm where it will scan from, then let the serial port go."""
    try:
        from trimmer_arm import TrimmerArm
    except Exception as e:
        print(f'[qr] no arm module ({e}) - calibrating from the current view')
        return None
    try:
        saved = json.loads((DATA / 'lawn_calibration.json').read_text()
                           ).get('rest_pose')
    except (OSError, ValueError):
        saved = None
    try:
        arm = TrimmerArm()
    except Exception as e:
        print(f'[qr] arm not available ({e}) - calibrating from the current view')
        return None
    try:
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(400, 20)
        if saved:
            pose = {int(k): int(v) for k, v in saved.items()}
            print(f'\n[qr] returning to the saved rest pose {pose} ...')
            arm.move_pose(pose)
            t0 = time.time()
            while time.time() - t0 < 8.0 and not arm.arrived(pose, tol=25):
                time.sleep(0.15)
            print('[qr] at rest pose - calibrating from here')
            return pose
        pose = arm.read_pose()
        print(f'\n[qr] no rest pose saved yet; adopting the current one: {pose}')
        return {int(k): int(v) for k, v in (pose or {}).items()} or None
    except Exception as e:
        print(f'[qr] could not reach the rest pose: {e}')
        return None
    finally:
        try:
            arm.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description='One-shot marker calibration.')
    ap.add_argument('--print', dest='do_print', action='store_true')
    ap.add_argument('--paper', default='a3',
                    choices=['a4', 'a3', 'letter', 'tabloid'])
    ap.add_argument('--marker', type=float, default=None)
    ap.add_argument('--source', default=None)
    ap.add_argument('--width', type=float, default=None)
    ap.add_argument('--length', type=float, default=None)
    ap.add_argument('--frames', type=int, default=AVG_FRAMES)
    args = ap.parse_args()
    ensure()

    if args.do_print:
        size = args.marker or max_size_for(args.paper)
        sheet, actual = printable_sheet(side_m=size, paper=args.paper, dpi=150)
        out = DATA / 'calibration_marker.png'
        cv2.imwrite(str(out), sheet)
        print(f'\nMarker sheet: {out}')
        print(f'Print at 100% scale (NOT "fit to page") on {args.paper.upper()}.')
        print(f'Measure the printed square - it should be {actual*100:.1f} cm.')
        print(f'Then:  python lawn_sentry/calibrate/calibrate_qr.py --marker {actual:.3f}\n')
        print('Bigger is better, and you can tape sheets together: a marker')
        print('drawn across four A4 sheets is about 40 cm and is worth the')
        print('sticky tape. Below ~22 px in the image the corners are noise.\n')
        return

    print('\n' + '=' * 70)
    print('QR CALIBRATION - one sighting, then take the marker away')
    print('=' * 70)
    W = args.width or ask('\nLawn WIDTH  (metres)', 3.5)
    L = args.length or ask('Lawn LENGTH (metres)', 2.1)
    # Measured off the actual print. The scale of everything this system
    # reports rides on this one number, so it is the measured size that is
    # offered, not the size the sheet was laid out at.
    size = args.marker or ask('Marker SIZE in metres - MEASURE the black square',
                              DEFAULT_SIZE_M)
    print(f'\nlawn {W} x {L} m, marker {size*100:.1f} cm')
    print('\nPut the marker flat on the grass ANYWHERE the camera sees it')
    print('clearly - close and large in the frame is best. You are not being')
    print('asked where it is; you will slide the lawn onto the lawn afterwards.')

    rest = go_to_rest()

    try:
        cam = FrameSource(args.source)
    except RuntimeError as e:
        print(f'camera problem: {e}')
        return

    finder = MarkerFinder()
    avg = PoseAverager(want=max(1, args.frames))
    K = None
    H = None
    cx, cy, yaw = W / 2.0, L / 2.0, 0.0
    locked = None                 # (R, t) once averaging finished
    frame = None
    side_px = 0.0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        while True:
            f = cam.read()
            if f is not None:
                frame = f
            if frame is None:
                if cv2.waitKey(30) & 255 == ord('q'):
                    break
                continue
            if K is None:
                K, ksrc = get_K(frame.shape[1], frame.shape[0])
            disp = frame.copy()
            h, w = disp.shape[:2]

            corners = finder.find(frame)
            if corners is not None:
                side_px = min(float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
                              for i in range(4))
            if corners is not None and locked is None and side_px >= MIN_SIDE_PX:
                avg.add(corners)
                if avg.ready:
                    R, t = solve_pose(avg.mean(), size, K)
                    if R is not None:
                        locked = (R, t)
            if corners is not None:
                col = ((0, 220, 255) if locked is None else (120, 200, 120))
                cv2.polylines(disp, [np.asarray(corners, np.int32)], True,
                              col, 2, cv2.LINE_AA)

            if locked is not None:
                H = ground_from_pose(locked[0], locked[1], K, cx, cy, yaw)
            drawn = draw_lawn(disp, H, W, L) if H is not None else False

            lines = []
            if locked is None:
                if corners is None:
                    lines.append(('MARKER NOT SEEN - more light, or move it '
                                  'closer to the camera', (80, 140, 255)))
                elif side_px < MIN_SIDE_PX:
                    lines.append((f'marker only {side_px:.0f} px across - too '
                                  f'small to trust. Move it CLOSER or print '
                                  f'bigger.', (80, 140, 255)))
                else:
                    pct = int(avg.progress * 100)
                    jit = avg.jitter_px()
                    bar = '#' * (pct // 5) + '.' * (20 - pct // 5)
                    lines.append((f'holding still, averaging [{bar}] {pct}%   '
                                  f'{side_px:.0f} px'
                                  + (f'   wobble {jit:.2f} px' if jit else ''),
                                  (0, 255, 120)))
            else:
                X, Y, Z = camera_from_H(H, K) or (0.0, 0.0, 0.0)
                # The marker's own size check cannot fail - the size is what
                # was solved from. Where the camera ends up CAN fail, and a
                # person can judge it instantly: a bad solve put the camera
                # 8 m in the air in testing while every internal check was
                # happy. This is the honest check.
                sane = 0.2 < Z < 2.5
                lines.append((f'camera sits {Z:.2f} m above the lawn, '
                              f'{-Y:.2f} m back from the near edge   '
                              f'(marker was {side_px:.0f} px)',
                              (0, 255, 120) if sane else (80, 140, 255)))
                if not sane:
                    lines.append(('^ IS THAT WHERE THE CAMERA REALLY IS? If not, '
                                  'the marker size is wrong or it was too small '
                                  '- press r', (80, 140, 255)))
                # Sliding the rectangle onto the lawn settles the geometry but
                # not which end is the NEAR one, and the picture cannot show
                # you that. The dashboard is not neutral about it: it pins the
                # mower to the bottom-centre of the map and swings the view
                # cone from there, which only makes sense with the camera
                # behind the near edge. Saved backwards, every detection lands
                # on the map rotated - and it looked perfect at save time.
                if Y > -0.05:
                    lines.append(('LAWN IS ROUND THE WRONG WAY - the camera is '
                                  'not behind the yellow near edge. Press f to '
                                  'flip it.', (80, 140, 255)))
                lines.append((f'arrows/WASD slide 5cm   - = fine 1cm   '
                              f'[ ] rotate   f flip 180   r re-average    '
                              f'({cx:.2f}, {cy:.2f}) m {yaw:+.0f}d',
                              (190, 190, 190)))
                lines.append(('SLIDE THE GREEN RECTANGLE ONTO YOUR REAL LAWN, '
                              'then SPACE to save   (q quits)', (255, 220, 90)))
            if H is not None and not drawn:
                lines.append(('the lawn projects off-screen - slide it back '
                              'with the arrow keys', (80, 140, 255)))
            cv2.rectangle(disp, (0, 0), (w, 25 * len(lines) + 8), (16, 16, 16), -1)
            for i, (t_, c) in enumerate(lines):
                cv2.putText(disp, t_, (10, 21 + i * 24), FONT, 0.45, c, 1,
                            cv2.LINE_AA)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(20) & 255
            if k == 255:
                continue
            if k == ord('q'):
                print('cancelled - nothing saved')
                break
            if k == ord('r'):
                avg.reset()
                locked = None
                H = None
                print('  re-averaging...')
                continue
            step = FINE_M if k in (ord('-'), ord('=')) else NUDGE_M
            # arrows and WASD both, because which codes OpenCV delivers depends
            # on the build and a control you cannot press is useless
            if k in (81, 2, ord('a')):
                cx -= step
            elif k in (83, 3, ord('d')):
                cx += step
            elif k in (82, 0, ord('w')):
                cy += step
            elif k in (84, 1, ord('s')):
                cy -= step
            elif k == ord('['):
                yaw -= ROT_DEG
            elif k == ord(']'):
                yaw += ROT_DEG
            elif k == ord('f'):
                yaw += 180.0          # swap which end of the lawn is "near"
            elif k in (32, 13, 10):     # SPACE or ENTER saves; 's' is "slide down"
                if locked is None:
                    print('  no marker averaged yet - hold it still in view')
                    continue
                if H is None:
                    print('  the plane did not solve - press r and try again')
                    continue
                X, Y, Z = camera_from_H(H, K) or (0.0, 0.0, 0.0)
                # The scan loop wants the lawn's corners in pixels. This method
                # never clicks any, so derive them from the plane - they are
                # the same information. Leaving them out crashed the scan the
                # moment it started, and the only visible symptom was a video
                # feed that never appeared.
                quad = project(H, [(0, 0), (W, 0), (W, L), (0, L)])
                data = {
                    'version': 9,
                    'calibration_type': 'qr_pose',
                    'created': datetime.now().isoformat(timespec='seconds'),
                    'lawn_width_m': W, 'lawn_length_m': L,
                    'homography_image_to_lawn': H.astype(float).tolist(),
                    'image_points_tl_tr_br_bl': [[float(a), float(b)]
                                                 for a, b in quad],
                    'reference_frame_points': [[float(a), float(b)]
                                               for a, b in quad],
                    'marker': {
                        'size_m': size,
                        'centre_m': [round(cx, 3), round(cy, 3)],
                        'yaw_deg': round(yaw, 2),
                        'side_px': round(float(side_px), 1),
                        'frames_averaged': len(avg.buf),
                        'corner_jitter_px': (round(avg.jitter_px(), 3)
                                             if avg.jitter_px() else None),
                        'solver': 'solvePnP IPPE_SQUARE + known K',
                    },
                    'camera_position_m': [round(X, 3), round(Y, 3), round(Z, 3)],
                    'intrinsics_source': ksrc,
                    'rest_pose': ({str(k2): int(v) for k2, v in rest.items()}
                                  if rest else None),
                    'camera_width_px': int(frame.shape[1]),
                    'camera_height_px': int(frame.shape[0]),
                    'note': 'marker pose + known intrinsics, lawn placed by eye',
                }
                cal_store.save(data, label='qr_pose')
                cv2.imwrite(str(DATA / 'qr_calibration_preview.jpg'), disp)
                print('\n' + '=' * 70)
                print(f'  SAVED  lawn {W} x {L} m')
                print(f'  averaged {len(avg.buf)} frames, marker {side_px:.0f} px '
                      f'across, intrinsics from {ksrc}')
                print(f'  camera {Z:.2f} m above the lawn, {-Y:.2f} m back '
                      f'from the near edge')
                if not (0.2 < Z < 2.5):
                    print(f'  ^ that height looks wrong. If the camera is not '
                          f'really {Z:.2f} m up, the marker size is probably')
                    print(f'    not {size*100:.0f} cm - measure the print and '
                          f'rerun with --marker')
                if rest:
                    print(f'  rest pose {rest} - launch.py returns here each run')
                print('\n  TAKE THE MARKER AWAY. It is not needed again unless')
                print('  the camera is moved or remounted.')
                print('\n  Previous calibration kept: python tools/calibrations.py')
                print('  Run:  python launch.py')
                print('=' * 70)
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
