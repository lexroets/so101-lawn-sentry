"""Walk-the-marker ground calibration — for any fixed camera.

The idea
--------
To put detections on a map, the system needs to know how image pixels relate to
metres on the ground. That relationship is a homography, and a homography needs
four points whose real positions we know.

Clicking the four lawn corners by hand is the obvious way and it is the weakest
link: a lawn edge is soft, often blurred, sometimes behind a bush, and a few
pixels of slop at the far corner becomes tens of centimetres on the map.

So instead: print one marker, put it flat on the ground at each corner in turn,
and let the camera find it to sub-pixel accuracy. Type the real width and
length of the area, and calibration is done.

The part that makes this trustworthy: the marker's own printed size is never
used to compute the answer, so it is an independent check ON the answer. After
solving, each marker is measured through the recovered homography — if the tool
says a 20 cm square is 20 cm at every corner, the ground plane is right. That
number is reported in centimetres, so the user knows whether to trust it rather
than having to hope.

Works with anything
-------------------
No camera intrinsics, no depth, no lens model. Just a fixed viewpoint. That
means it runs on a home security camera as readily as on the OAK-D.

    # print the marker first
    python calibrate/calibrate_marker.py --print

    # then calibrate
    python calibrate/calibrate_marker.py                       # OAK-D
    python calibrate/calibrate_marker.py --source 0            # USB webcam
    python calibrate/calibrate_marker.py --source rtsp://...   # security camera

Keys
    SPACE / c   capture the marker at this corner
    a           toggle auto-capture (on by default) — hands-free
    u           undo the last corner
    e           capture an EXTRA check point (anywhere on the lawn)
    s           solve and save
    q           quit without saving
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
from datetime import datetime

import cv2
import numpy as np

from paths import DATA, ensure
from frame_source import FrameSource
from marker import (MarkerFinder, printable_sheet, DEFAULT_SIZE_M,
                    max_size_for)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Ground calibration - walk the marker round the area'
CORNER_NAMES = ['CORNER 1', 'CORNER 2', 'CORNER 3', 'CORNER 4']
COLS = [(0, 255, 255), (0, 210, 255), (0, 160, 255), (90, 255, 90)]

# Auto-capture: the marker must sit still this long, within this many pixels,
# before we take it. Stops a hand or a gust being baked into the calibration.
STABLE_FRAMES = 12
STABLE_PX = 2.5

# A 4x4 ArUco code is 6 cells across including its black border, so it needs
# roughly 5 px per cell to decode reliably. Below this the far corners simply
# will not be found, and the honest answer is "print it bigger".
MIN_MARKER_PX = 32

# --- current setup ----------------------------------------------------------
# The printed marker is 15 cm square and the taped reference box on the grass
# is 60 x 40 cm. Change these two lines if either changes, or override on the
# command line with --marker / --width / --length.
MARKER_SIZE_M = 0.15
AREA_WIDTH_M = 0.60
AREA_LENGTH_M = 0.40

# A homography is exact inside the box you calibrated on and degrades as you
# move away from it. Measured on a 1.2 m-high camera viewing a 3.5 x 2.1 m
# lawn: a 0.6 x 0.4 m box gives ~0.4 cm inside itself, ~0.9 cm within 1.5 m,
# and ~5 cm out at the lawn edges; a 1.0 x 0.7 m box holds ~0.8 cm everywhere.
# So a small box is fine for checking the system works, and worth enlarging if
# you want the far corners of a real lawn to be tight.
SMALL_BOX_M = 1.0


def ask_float(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    if not raw:
        return default
    try:
        v = float(raw.replace(',', '.'))
        return v if v > 0 else default
    except ValueError:
        return default


def to_world(H, pts):
    pts = np.asarray(pts, np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def signed_area(pts):
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def measure_marker(H, corners, true_side_m):
    """Measure a marker through H.

    Returns (mean side m, size error cm, out-of-square fraction).

    Both numbers matter and for different reasons. The SIZE says whether the
    scale is right. The SQUARENESS says whether the corner assignment is right:
    if the four lawn corners are matched to the rectangle in the wrong order,
    the ground plane comes out rotated, a square marker maps to an oblong, and
    the size alone would not reveal it because averaging the four sides hides
    the distortion. That combination is what makes a wrong answer detectable.
    """
    w = to_world(H, corners)
    sides = [float(np.linalg.norm(w[i] - w[(i + 1) % 4])) for i in range(4)]
    mean = float(np.mean(sides))
    # how far from square: adjacent sides should be equal, as should diagonals
    a = (sides[0] + sides[2]) / 2.0
    b = (sides[1] + sides[3]) / 2.0
    aspect = abs(a - b) / max(a, b, 1e-9)
    d1 = np.linalg.norm(w[0] - w[2])
    d2 = np.linalg.norm(w[1] - w[3])
    diag = abs(d1 - d2) / max(d1, d2, 1e-9)
    return mean, (mean - true_side_m) * 100.0, max(aspect, diag)


# kept for callers that only want the size figure
def check_marker(H, corners, true_side_m):
    return measure_marker(H, corners, true_side_m)


def solve_panned(captures, angles, K, W, L, marker_side, tilt_hint=None):
    """Solve when each corner was captured at a different pan angle.

    Views are related by K R K^-1 where R is a rotation of the MEASURED angle
    about the mount's vertical axis. That is exact - verified to 0.000 px out to
    80 degrees of pan - and because each angle is absolute, nothing accumulates.

    The one thing we do not know is how far the camera is tilted down, which
    sets where the vertical axis lies in camera coordinates. Rather than ask
    for it, search: the correct tilt is the one that makes a printed square
    come back square and the right size at every corner. The marker supplies
    its own answer, exactly as it does for the corner ordering.

    Returns (H, src, tilt_deg, score).
    """
    from camera_intrinsics import axis_from_tilt, rotation_homography_axis

    ref = angles[0]
    best = None
    coarse = np.arange(0.0, 85.1, 2.5) if tilt_hint is None else \
        np.arange(max(0.0, tilt_hint - 10), min(85.0, tilt_hint + 10) + .1, 1.0)
    for stage, grid in enumerate((coarse, None)):
        if grid is None:                       # refine around the coarse winner
            t0 = best[1]
            grid = np.arange(max(0.0, t0 - 2.5), min(85.0, t0 + 2.5) + .01, 0.25)
        for tilt in grid:
            axis = axis_from_tilt(float(tilt))
            caps = []
            ok = True
            for c, a in zip(captures, angles):
                Hp = rotation_homography_axis(K, float(a - ref), axis)
                q = cv2.perspectiveTransform(
                    np.asarray(c['corners'], np.float32).reshape(1, -1, 2),
                    Hp.astype(np.float32)).reshape(4, 2)
                if not np.isfinite(q).all() or np.abs(q).max() > 1e6:
                    ok = False
                    break
                caps.append({'centre': q.mean(axis=0), 'corners': q})
            if not ok:
                continue
            try:
                H, src = solve(caps, W, L, marker_side)
            except cv2.error:
                continue
            m = [measure_marker(H, c['corners'], marker_side) for c in caps]
            score = (float(np.mean([e[2] for e in m])) * 10.0
                     + float(np.mean([abs(e[1]) for e in m])) / 100.0)
            if best is None or score < best[0]:
                best = (score, float(tilt), H, src, caps)
    if best is None:
        raise cv2.error('could not solve with the measured pan angles')
    score, tilt, H, src, caps = best
    solve_panned.last = {'tilt_deg': tilt, 'score': score, 'caps': caps}
    return H, src, tilt, score


def solve(captures, W, L, marker_side=None):
    """Homography from image pixels to lawn metres, from captures IN ORDER.

    The user walks the marker round the perimeter, so consecutive captures are
    adjacent corners. That fixes the cyclic order but not where the sequence
    started, nor which way round they walked — eight possibilities in all.

    Rather than guess from image position (which fails as soon as the camera is
    mounted at an angle), every possibility is tried and scored by how square
    the markers come out. Only the assignments with the correct aspect can turn
    a printed square back into a square, so the marker itself picks the answer.

    `captures` may be a list of centres, or of dicts with 'centre' and
    'corners' — the latter enables the squareness scoring.
    """
    if isinstance(captures[0], dict):
        centres = [c['centre'] for c in captures]
        quads = [c['corners'] for c in captures]
    else:
        centres, quads = list(captures), None
    centres = np.asarray(centres, np.float32)

    # A camera above the ground always sees the perimeter wound the same way,
    # whatever angle it is mounted at (verified across yaw -55..+40 degrees).
    # Walking the marker the other way round the lawn just reverses it, so
    # normalising the winding here removes the mirror-image possibilities —
    # which squareness scoring below could never distinguish, since a mirrored
    # square is still a square.
    # Reverse the walk direction if needed, but keep the FIRST capture first:
    # [0,3,2,1] rather than [3,2,1,0]. That way the corner the user placed
    # first is always the map origin, whichever way round they walked.
    order = [0, 1, 2, 3]
    if signed_area(centres) > 0:
        order = [0, 3, 2, 1]

    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32)
    cands = []
    for rot in range(4):
        idx = [order[(rot + k) % 4] for k in range(4)]
        src = centres[idx]
        try:
            H = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            continue
        if not np.all(np.isfinite(H)):
            continue
        if quads is None or marker_side is None:
            sq = 0.0 if rot == 0 else 1.0           # nothing to score with
        else:
            sq = float(np.mean([measure_marker(H, quads[i], marker_side)[2]
                                for i in idx]))
        cands.append({'rot': rot, 'H': H, 'src': src, 'idx': idx, 'sq': sq})

    if not cands:
        raise cv2.error('degenerate marker positions')

    # Rotations by 90 degrees swap the lawn's width and length, so a printed
    # square comes back as an oblong — they score several times worse and are
    # eliminated here. The 0 and 180 degree options describe the SAME ground
    # plane with the origin at opposite corners, so they score alike and
    # measurement noise must not be allowed to choose between them. Among
    # equally-square candidates, take the one that puts the first-placed corner
    # at the origin, which is what the user was told would happen.
    best_sq = min(c['sq'] for c in cands)
    tol = best_sq * 0.6 + 0.04
    viable = [c for c in cands if c['sq'] <= best_sq + tol]
    chosen = min(viable, key=lambda c: c['rot'])

    solve.last = {'rot': chosen['rot'], 'order': chosen['idx'],
                  'squareness': chosen['sq'],
                  'rejected': [(c['rot'], round(c['sq'], 4))
                               for c in cands if c is not chosen],
                  'ambiguous': len(viable) > 2}
    return chosen['H'], chosen['src']


def draw_marker(img, corners, colour, label=None, fill=False):
    p = np.asarray(corners, np.int32)
    if fill:
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, p, colour)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    cv2.polylines(img, [p], True, colour, 2, cv2.LINE_AA)
    c = p.mean(axis=0).astype(int)
    cv2.drawMarker(img, tuple(c), colour, cv2.MARKER_CROSS, 18, 2)
    if label:
        cv2.putText(img, label, (c[0] + 12, c[1] - 10), FONT, 0.6, colour, 2,
                    cv2.LINE_AA)


def banner(img, lines, colour=(0, 255, 255)):
    h, w = img.shape[:2]
    box = 26 * len(lines) + 12
    cv2.rectangle(img, (0, 0), (w, box), (18, 18, 18), -1)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (10, 24 + i * 25), FONT, 0.58, col or colour, 1,
                    cv2.LINE_AA)


def do_print(side_m, out_path, dpi, paper):
    asked = side_m
    sheet, actual = printable_sheet(side_m=side_m, dpi=dpi, paper=paper)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    print(f'\nMarker sheet written to:\n    {out_path}\n')
    if actual < asked - 1e-4:
        print(f'  NOTE: {asked*100:.0f} cm does not fit on {paper.upper()} once')
        print(f'  printer margins and the white border are allowed for, so the')
        print(f'  marker is {actual*100:.1f} cm. For a bigger one print on A3:')
        print(f'      --paper a3 --marker {max_size_for("a3"):.2f}\n')
    print('Print at 100% scale - turn OFF "fit to page" / "shrink to fit".')
    print(f'Then measure it: the black square should be {actual*100:.1f} cm across.')
    print('If your printer scaled it anyway, no problem - measure what you got')
    print(f'and pass that when calibrating:   --marker <metres>\n')
    print(f'When ready:   python calibrate/calibrate_marker.py --marker {actual:.3f}\n')
    return actual


def main():
    ap = argparse.ArgumentParser(
        description='Ground calibration by walking a printed marker round the area.')
    ap.add_argument('--print', dest='do_print', action='store_true',
                    help='write a printable marker sheet and exit')
    ap.add_argument('--source', default=None,
                    help='0 for webcam, an RTSP/HTTP URL for a security camera, '
                         'a file path, or omit for the OAK-D')
    ap.add_argument('--marker', type=float, default=MARKER_SIZE_M,
                    help=f'printed marker side in metres (default {MARKER_SIZE_M})')
    ap.add_argument('--width', type=float, default=None, help='area width in metres')
    ap.add_argument('--length', type=float, default=None, help='area length in metres')
    ap.add_argument('--dpi', type=int, default=150, help='print resolution')
    ap.add_argument('--paper', default='a4', choices=['a4', 'a3', 'letter', 'tabloid'],
                    help='paper size for --print (default a4)')
    ap.add_argument('--out', default=None, help='where to write the calibration')
    ap.add_argument('--lawn', nargs=2, type=float, metavar=('W', 'L'),
                    default=None,
                    help='the REAL lawn size, when the box you walked is only '
                         'part of it. The map then covers the whole lawn.')
    ap.add_argument('--offset', nargs=2, type=float, metavar=('X', 'Y'),
                    default=None,
                    help='where the FIRST corner of your box sits in lawn '
                         'coordinates, measured from the lawn corner nearest '
                         'the camera on the left. Defaults to centring the box '
                         'in X.')
    ap.add_argument('--arm', action='store_true',
                    help='read the pan angle from the SO-101 base servo and use '
                         'it to relate views geometrically. Read-only - it never '
                         'commands the arm. Removes the drift that --pan suffers '
                         'from, because an encoder angle is absolute rather than '
                         'accumulated frame by frame')
    ap.add_argument('--pan', action='store_true',
                    help='LAST RESORT: pan between corners when the area will '
                         'not fit in one view. Tracking drifts, so accuracy '
                         'degrades quickly beyond about 15 degrees of pan - '
                         'prefer backing the camera off and using a larger '
                         'marker instead')
    args = ap.parse_args()

    ensure()
    if args.do_print:
        do_print(args.marker, DATA / 'calibration_marker.png', args.dpi,
                 args.paper)
        return

    out_path = _pl.Path(args.out) if args.out else DATA / 'lawn_calibration.json'

    print('\n' + '=' * 66)
    print('WALK-THE-MARKER GROUND CALIBRATION')
    print('=' * 66)
    print('\nPut the printed marker flat on the ground at each corner of the')
    print('area you want watched, going around the edge. The camera does the')
    print('rest. No clicking, no tape measure beyond the two dimensions.\n')
    print("Haven't printed it yet?   python calibrate/calibrate_marker.py --print\n")

    W = args.width if args.width else ask_float('Area WIDTH  (metres)', AREA_WIDTH_M)
    L = args.length if args.length else ask_float('Area LENGTH (metres)', AREA_LENGTH_M)
    side = args.marker
    print(f'\narea {W} x {L} m,  marker {side*100:.1f} cm\n')

    try:
        cam = FrameSource(args.source)
    except RuntimeError as e:
        print(f'\ncamera problem:\n  {e}\n')
        return
    print(f'camera: {cam.describe()}')

    finder = MarkerFinder()

    # --- encoder-driven view stitching ------------------------------------
    pan = None
    K = None
    if args.arm:
        from pan_source import open_pan
        from camera_intrinsics import get as get_K, rotation_homography
        pan = open_pan('arm')
        if pan.name == 'fixed':
            print('\n[arm] could not read the SO-101 base servo. Is it plugged')
            print('      in and powered? Falling back to a fixed camera - do')
            print('      not pan, or the calibration will be wrong.\n')
            pan = None
        else:
            probe = cam.read_blocking(10.0)
            if probe is None:
                print('[arm] no frame from the camera'); return
            K, ksrc = get_K(probe.shape[1], probe.shape[0])
            print(f'\nARM MODE: pan angle read from the SO-101 base servo.')
            print(f'  intrinsics: {ksrc}')
            print(f'  base now at {pan.angle_deg():+.1f} deg')
            print('  Pan the arm by hand or with your usual controls - this tool')
            print('  only ever READS the servo, it never moves it.')
            print('  Views are related by K R K^-1 from the measured angle, so')
            print('  nothing accumulates and nothing drifts.\n')

    trk = None
    if args.pan and pan is None:
        # A homography is only true for ONE camera pose, so four corners
        # captured from four different views cannot simply be combined. Ground
        # tracking estimates the transform G from the current view back to the
        # first, which lets a corner seen after panning be expressed in the
        # first view's frame - even when it now lies outside the picture.
        from ground_track import GroundTracker
        trk = GroundTracker(np.eye(3))
        print('\n' + '!' * 68)
        print('PAN MODE - USE ONLY IF YOU MUST')
        print('!' * 68)
        print('Tracking dead-reckons from the first view, and that estimate')
        print('DRIFTS. Measured on a simulated version of this setup, a ground')
        print('point comes back displaced by roughly:')
        print('     5 deg of pan  ->   8 px      20 deg ->  68 px')
        print('    10 deg of pan  ->  30 px      45 deg -> 537 px')
        print('At 20 degrees that is already tens of centimetres on the map, and')
        print('the usual accuracy check CANNOT see it - the markers still come')
        print('back the right size and shape because the error is a slow shift,')
        print('not a distortion.')
        print('')
        print('BETTER: move the camera back until the whole area fits in one')
        print('view, print a bigger marker (--paper a3 --marker 0.25) so the far')
        print('corners are still readable, and run without --pan.')
        print('')
        print('If you continue: pan as little as possible, move slowly, keep')
        print('textured ground in view, and treat the result as approximate.')
        print('!' * 68 + '\n')

    captured = []          # [{'corners': 4x2, 'centre': (x,y)}]
    extras = []            # optional check points anywhere on the lawn
    auto = True
    stable = 0
    last_centre = None
    frame = None

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

            if trk is not None:
                trk.update(frame)

            disp = frame.copy()
            h, w = disp.shape[:2]
            corners = finder.find(frame)
            centre = corners.mean(axis=0) if corners is not None else None

            # --- stability tracking for hands-free capture ---
            if centre is not None and last_centre is not None \
                    and np.linalg.norm(centre - last_centre) < STABLE_PX:
                stable += 1
            else:
                stable = 0
            last_centre = centre

            need = len(captured) < 4
            take = False
            big_enough = (corners is not None and
                          np.linalg.norm(corners[0] - corners[1]) >= MIN_MARKER_PX)
            if corners is not None and auto and need and big_enough \
                    and stable >= STABLE_FRAMES:
                # don't re-capture the same spot
                if not captured or np.linalg.norm(
                        centre - np.asarray(captured[-1]['centre'])) > 40:
                    take = True

            # --- draw what we have ---
            back = np.eye(3)
            if trk is not None:
                try:
                    back = np.linalg.inv(trk.G)
                except np.linalg.LinAlgError:
                    back = None
            for i, c in enumerate(captured):
                q = c['corners']
                if back is not None and trk is not None:
                    q = cv2.perspectiveTransform(
                        np.asarray(q, np.float32).reshape(1, -1, 2), back
                    ).reshape(4, 2)
                if back is None or not np.isfinite(q).all():
                    continue
                draw_marker(disp, q, COLS[i], f'{i+1}', fill=True)
            for j, c in enumerate(extras):
                draw_marker(disp, c['corners'], (255, 170, 60), f'check {j+1}')
            if len(captured) >= 3:
                pts = np.asarray([c['centre'] for c in captured], np.int32)
                cv2.polylines(disp, [pts], len(captured) == 4, (80, 255, 80), 2,
                              cv2.LINE_AA)

            if corners is not None:
                prog = min(1.0, stable / STABLE_FRAMES)
                col = (0, 255, 120) if prog >= 1 else (0, 200, 255)
                draw_marker(disp, corners, col)
                if auto and need:
                    cx, cy = centre.astype(int)
                    cv2.ellipse(disp, (cx, cy), (34, 34), -90, 0, 360 * prog,
                                col, 3, cv2.LINE_AA)

            lines = []
            if need:
                lines.append((f'{len(captured)}/4  place the marker at '
                              f'{CORNER_NAMES[len(captured)]}', None))
            else:
                lines.append(('4/4 captured  -  press S to solve and save',
                              (90, 255, 90)))
            if corners is None:
                lines.append(('marker NOT visible - move it into view / better light',
                              (80, 140, 255)))
            else:
                mpx = float(np.linalg.norm(corners[0] - corners[1]))
                if mpx < MIN_MARKER_PX:
                    lines.append((f'marker only {mpx:.0f} px - TOO SMALL to read '
                                  f'reliably (need {MIN_MARKER_PX}+). Print it '
                                  f'bigger or move the camera closer.',
                                  (80, 140, 255)))
                else:
                    lines.append((f'marker seen, {mpx:.0f} px  -  '
                                  f'{"holding still..." if stable < STABLE_FRAMES else "STEADY"}',
                                  (0, 255, 120) if stable >= STABLE_FRAMES else (0, 200, 255)))
            if pan is not None:
                a = pan.angle_deg()
                ref = getattr(pan, 'reference', None)
                if a is None:
                    lines.append(('SERVO NOT READABLE - check the arm connection',
                                  (80, 140, 255)))
                else:
                    rel = '' if ref is None else f'   {a-ref:+.1f} deg from corner 1'
                    lines.append((f'base {a:+.1f} deg{rel}', (200, 220, 120)))
            if trk is not None:
                st = trk.status()
                lines.append((f'tracking {"LOST - press r and restart" if st["lost"] else "OK"}'
                              f'  features {st["tracked"]}',
                              (0, 220, 90) if not st['lost'] else (0, 120, 255)))
            lines.append((f'auto-capture {"ON" if auto else "off"} (a)   '
                          f'SPACE capture   u undo   e extra check   s save   q quit',
                          (170, 170, 170)))
            banner(disp, lines)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                print('\ncancelled - nothing saved')
                break
            if k == ord('r') and trk is not None:
                trk.anchor(); captured.clear(); extras.clear()
                print('  tracking restarted - all corners cleared')
            if k == ord('a'):
                auto = not auto
                print(f'  auto-capture {"on" if auto else "off"}')
            if k in (ord('u'),) and captured:
                captured.pop()
                stable = 0
                print(f'  undid corner {len(captured)+1}')
            if k in (32, ord('c')) and corners is not None and need:
                take = True
            if k == ord('e') and corners is not None:
                keep = corners.copy()
                if pan is not None:
                    a = pan.angle_deg()
                    if a is not None:
                        d = a - getattr(pan, 'reference', a)
                        Hp = rotation_homography(K, -d, axis='y')
                        keep = cv2.perspectiveTransform(
                            keep.reshape(1, -1, 2).astype(np.float32),
                            Hp.astype(np.float32)).reshape(4, 2)
                elif trk is not None:
                    keep = cv2.perspectiveTransform(
                        keep.reshape(1, -1, 2).astype(np.float32), trk.G
                    ).reshape(4, 2)
                extras.append({'corners': keep,
                               'centre': tuple(map(float, keep.mean(axis=0)))})
                print(f'  extra check point {len(extras)} captured')

            if take:
                keep = corners.copy()
                if pan is not None:
                    a = pan.angle_deg()
                    if a is None:
                        print('  lost the servo reading - not capturing')
                        click = None
                        continue
                    if not captured:
                        pan.reference = a          # first corner sets the frame
                    d = a - getattr(pan, 'reference', a)
                    Hp = rotation_homography(K, -d, axis='y')
                    keep = cv2.perspectiveTransform(
                        keep.reshape(1, -1, 2).astype(np.float32),
                        Hp.astype(np.float32)).reshape(4, 2)
                elif trk is not None:
                    # bring this capture into the first view's coordinates so
                    # all four corners live in one common frame
                    keep = cv2.perspectiveTransform(
                        keep.reshape(1, -1, 2).astype(np.float32), trk.G
                    ).reshape(4, 2)
                captured.append({'corners': keep,
                                 'centre': tuple(map(float, keep.mean(axis=0)))})
                stable = 0
                print(f'  {CORNER_NAMES[len(captured)-1]} captured at '
                      f'({centre[0]:.0f}, {centre[1]:.0f}) px')
                if len(captured) == 4:
                    print('\n  all four corners captured - press S to solve\n')

            if k == ord('s'):
                if len(captured) != 4:
                    print(f'  need 4 corners, have {len(captured)}')
                    continue
                centres = [c['centre'] for c in captured]
                if abs(cv2.contourArea(np.asarray(centres, np.float32))) < 500:
                    print('  those four points are nearly in a line - recapture')
                    continue
                try:
                    H, src = solve(captured, W, L, side)
                except cv2.error:
                    print('  degenerate geometry - recapture'); continue
                if not np.all(np.isfinite(H)):
                    print('  bad geometry - recapture'); continue

                # ---- the honest accuracy check ----
                print('\n  ACCURACY CHECK - measuring the marker through the result')
                print('  (the solve never saw the marker size, so this is independent)')
                errs, skews = [], []
                for i, c in enumerate(captured):
                    mean, err_cm, skew = measure_marker(H, c['corners'], side)
                    errs.append(err_cm); skews.append(skew)
                    print(f'    corner {i+1}: marker measures {mean*100:5.1f} cm '
                          f'(true {side*100:.1f})  error {err_cm:+5.1f} cm'
                          f'   out-of-square {skew*100:4.1f}%')
                for j, c in enumerate(extras):
                    mean, err_cm, skew = measure_marker(H, c['corners'], side)
                    errs.append(err_cm); skews.append(skew)
                    print(f'    check {j+1} : marker measures {mean*100:5.1f} cm '
                          f'  error {err_cm:+5.1f} cm')
                if trk is not None:
                    # how far the view has swung, measured as the image centre's
                    # displacement when referred back to the first view
                    try:
                        cc = np.array([[[frame.shape[1] / 2.0,
                                         frame.shape[0] / 2.0]]], np.float32)
                        moved = float(np.linalg.norm(
                            cv2.perspectiveTransform(cc, trk.G)[0, 0] - cc[0, 0]))
                    except cv2.error:
                        moved = 0.0
                    print(f'\n    PAN MODE was used. The view drifted '
                          f'{moved:.0f} px from the first frame.')
                    if moved > 120:
                        print('    That is a LOT. Positions on the map may be out by')
                        print('    tens of centimetres, and the check above will not')
                        print('    show it. Re-do this without panning if you can.')
                    else:
                        print('    That is modest, so drift should be small - but it')
                        print('    is still dead reckoning, so treat it as approximate.')
                info = getattr(solve, 'last', {})
                worst_sq = max(skews)
                if worst_sq > 0.35:
                    print(f'\n    WARNING: the marker comes back {worst_sq*100:.0f}% '
                          f'out of square.')
                    print('    That usually means the four corners were not walked in')
                    print('    order round the edge, or the width and length are')
                    print('    swapped. Check both before trusting this.')
                if info.get('ambiguous'):
                    print('\n    NOTE: the corner order could not be pinned down from')
                    print('    shape alone; the first corner you placed was taken as')
                    print('    the origin. If the map comes out rotated, recalibrate')
                    print('    starting from a different corner.')
                worst = max(abs(e) for e in errs)
                rms = float(np.sqrt(np.mean(np.square(errs))))
                pct = worst / (side * 100) * 100
                print(f'\n    worst error {worst:.1f} cm ({pct:.0f}% of the marker), '
                      f'RMS {rms:.1f} cm')
                if pct < 6:
                    verdict, ok = 'EXCELLENT - trust this calibration', True
                elif pct < 15:
                    verdict, ok = 'GOOD - fine for detection and mapping', True
                elif pct < 30:
                    verdict, ok = ('USABLE, but positions may be off by tens of cm. '
                                   'Check the marker lay flat.'), True
                else:
                    verdict, ok = ('POOR - likely a tilted marker, a wrong printed '
                                   'size, or wrong area dimensions.'), False
                print(f'    {verdict}\n')
                if min(W, L) < SMALL_BOX_M:
                    print(f'    NOTE: a {W:.2f} x {L:.2f} m reference box is small.')
                    print('    Positions are most accurate inside it and drift as you')
                    print('    move away - expect a few cm at 1.5 m, more at the far')
                    print('    edges of a full lawn. Fine for a test; use a box of at')
                    print(f'    least {SMALL_BOX_M:.1f} m across for a lawn-wide setup.\n')
                if not ok:
                    if input('    save anyway? [y/N]: ').strip().lower() != 'y':
                        print('    not saved - recapture and try again\n')
                        continue

                # The box may be a small reference patch inside a bigger lawn.
                # Shifting the origin is just a translation applied after the
                # homography, so the map can describe the whole lawn while the
                # markers stayed close enough to the camera to be read.
                out_W, out_L = W, L
                if args.lawn:
                    out_W, out_L = float(args.lawn[0]), float(args.lawn[1])
                    if args.offset:
                        ox, oy = float(args.offset[0]), float(args.offset[1])
                    else:
                        ox, oy = (out_W - W) / 2.0, (out_L - L) / 2.0
                    Tm = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], float)
                    H = Tm @ H
                    H /= H[2, 2]
                    print(f'\n    Box mapped into the full lawn: origin shifted '
                          f'by ({ox:.2f}, {oy:.2f}) m,')
                    print(f'    map now describes {out_W} x {out_L} m.')
                    frac = (W * L) / max(out_W * out_L, 1e-9)
                    if frac < 0.10:
                        print(f'    NOTE: the box is only {frac*100:.0f}% of the lawn '
                              f'area. Accuracy is best')
                        print('    near the box and degrades further out.')

                data = {
                    'version': 5,
                    'calibration_type': 'aruco_marker_walk',
                    'created': datetime.now().isoformat(timespec='seconds'),
                    'lawn_width_m': out_W,
                    'lawn_length_m': out_L,
                    'reference_box_m': [W, L],
                    'marker_size_m': side,
                    'image_points_tl_tr_br_bl': [[float(a), float(b)] for a, b in src],
                    'homography_image_to_lawn': H.astype(float).tolist(),
                    'accuracy': {
                        'worst_error_cm': round(worst, 2),
                        'rms_error_cm': round(rms, 2),
                        'worst_error_pct_of_marker': round(pct, 1),
                        'per_capture_error_cm': [round(e, 2) for e in errs],
                        'worst_out_of_square_pct': round(max(skews) * 100, 1),
                        'corner_assignment': getattr(solve, 'last', {}).get('rot'),
                        'verdict': verdict,
                    },
                    'panned': bool(trk is not None),
                    'pan_source': pan.name if pan is not None else None,
                    'camera': cam.describe(),
                    'camera_width_px': int(frame.shape[1]),
                    'camera_height_px': int(frame.shape[0]),
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(data, indent=2))
                cv2.imwrite(str(DATA / 'calibration_marker_preview.jpg'), disp)
                print(f'  SAVED {out_path}')
                print(f'  area {W} x {L} m, ground plane fixed from 4 marker positions')
                print('\n  Do not move the camera from here on. If it is knocked,')
                print('  run this again.\n')
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
