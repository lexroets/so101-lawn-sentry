"""Set the whole system up in one run. Marker down, outline on the lawn, done.

    python calibrate.py          then      python launch.py

That is the entire installation, and this file exists so that sentence is
literally true rather than nearly true. It used to take three runs - one to
teach the controls which joint pans, one to solve the lawn plane from the
marker, and sometimes a third to turn the lawn the right way round - and each
one was a separate thing to remember, get wrong, or skip. Skipping the first
left the direction buttons guessing; skipping the last put every detection on
the map a quarter turn out.

What happens, in order:

    1  you frame the lawn         - jog the arm; this becomes the REST POSE,
                                    and the plane is measured from it
    2  the arm probes itself      - a few small nudges while watching the
                                    picture reveal which joint pans and which
                                    tilts. Nobody has to know the mounting.
    3  the marker is read         - one square of known size, averaged over
                                    forty frames, gives the ground plane
    4  you place the lawn         - arrow keys until the green rectangle sits
                                    on the real grass
    5  it saves                   - both config files, orientation corrected
                                    automatically, previous ones archived

Then take the marker away. It is not needed again unless the camera moves.

Keys
    1 2 3 4         choose base / shoulder / elbow / wrist   (step 1)
    A D W S         jog it / slide the lawn                  - =  fine steps
    [ ]             rotate the lawn        f   flip it end for end
    SPACE           accept this step       r   re-read the marker
    q               quit without saving
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
import time
from datetime import datetime

import cv2
import numpy as np

import cal_store
import trimmer_config as tc
from paths import DATA, ensure
from frame_source import FrameSource
from marker import MarkerFinder, printable_sheet, max_size_for, DEFAULT_SIZE_M
from marker_pose import (solve_pose, ground_from_pose, camera_from_H,
                         PoseAverager, MIN_SIDE_PX)
from camera_intrinsics import get as get_K
from fix_orientation import quarter_turn, camera_xy, score

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'SO-101 Lawn Sentry - setup'
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID)]
TICKS_PER_DEG = 4096.0 / 360.0
AVG_FRAMES = 40
PROBE_DEG = 7.0             # how far each joint is nudged while self-probing
# Below this the 'movement' is rounding error, not the camera going anywhere.
MIN_PROBE_PX = 4.0
NUDGE_M, FINE_M, ROT_DEG = 0.05, 0.01, 1.0


def ask(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    try:
        v = float(raw.replace(',', '.'))
        return v if v > 0 else default
    except ValueError:
        return default


LK = dict(winSize=(31, 31), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def shift_px(a, b):
    """How far the CONTENT moved between two frames: (dx, dy) in pixels.

    Note the sign is the content's, not the camera's - they are opposite, and
    conflating the two is how this got backwards before. Panning the camera
    right slides everything left, so a rightward pan reads as NEGATIVE dx.

    Median rather than mean, because a few features latching onto the marker's
    edge or a shadow should not swing the answer. Tracked forwards and then
    back again, keeping only the points that return to where they started -
    grass is repetitive, and a mismatched pair of blades looks exactly like a
    confident measurement otherwise.
    """
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(ga, 300, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, **LK)
    if p1 is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, p1, None, **LK)
    if back is None:
        return None
    fb = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.5)
    if good.sum() < 10:
        return None
    d = (p1[good] - p0[good]).reshape(-1, 2)
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def project(H, pts):
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    p = cv2.perspectiveTransform(
        np.asarray(pts, np.float32).reshape(1, -1, 2), Hi.astype(np.float32)
    ).reshape(-1, 2)
    if not np.isfinite(p).all() or np.abs(p).max() > 1e5:
        return None
    return p


def draw_lawn(img, H, W, L):
    outline = project(H, [(0, 0), (W, 0), (W, L), (0, L)])
    if outline is None:
        return False
    for x in np.arange(0.5, W, 0.5):
        g = project(H, [(x, 0), (x, L)])
        if g is not None:
            cv2.line(img, tuple(g[0].astype(int)), tuple(g[1].astype(int)),
                     (60, 150, 60), 1, cv2.LINE_AA)
    for y in np.arange(0.5, L, 0.5):
        g = project(H, [(0, y), (W, y)])
        if g is not None:
            cv2.line(img, tuple(g[0].astype(int)), tuple(g[1].astype(int)),
                     (60, 150, 60), 1, cv2.LINE_AA)
    cv2.polylines(img, [outline.astype(np.int32)], True, (90, 255, 90), 2, cv2.LINE_AA)
    cv2.line(img, tuple(outline[0].astype(int)), tuple(outline[1].astype(int)),
             (255, 220, 90), 3, cv2.LINE_AA)
    return True


def banner(img, lines):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 25 * len(lines) + 8), (16, 16, 16), -1)
    for i, (t, c) in enumerate(lines):
        cv2.putText(img, t, (10, 21 + i * 24), FONT, 0.47, c, 1, cv2.LINE_AA)


def settle(cam, seconds, state):
    """Wait while STILL reading the camera.

    A plain sleep stops draining the DepthAI queue; it fills and the device
    faults. This cost an afternoon once already.
    """
    t0 = time.time()
    while time.time() - t0 < seconds:
        f = cam.read()
        if f is not None:
            state['frame'] = f
            cv2.imshow(WIN, f)
        cv2.waitKey(1)
    return state['frame']


def probe_joints(arm, cam, state):
    """Find out which joint pans and which tilts, by trying them.

    Nothing is assumed about the mounting and nobody is asked to demonstrate
    anything: each joint is nudged a few degrees and the picture is watched.
    Whichever moves the view sideways most is the pan; whichever moves it
    vertically most is the tilt. Measuring beats reasoning here - the sign got
    talked through wrong twice before this existed.
    """
    results = {}
    for name, sid in JOINTS:
        try:
            start = arm.read_position(sid)
            if start is None:
                continue
        except Exception:
            continue
        before = settle(cam, 0.35, state).copy()
        try:
            arm.move_to(sid, int(start + PROBE_DEG * TICKS_PER_DEG))
        except Exception:
            continue
        after = settle(cam, 1.1, state).copy()
        d = shift_px(before, after)
        try:                                   # always put it back
            arm.move_to(sid, int(start))
        except Exception:
            pass
        settle(cam, 0.9, state)
        if d is None:
            print(f'    {name:9s} no features to track - skipped')
            continue
        results[name] = d
        print(f'    {name:9s} +{PROBE_DEG:.0f} deg moved the view '
              f'{d[0]:+6.1f} px across, {d[1]:+6.1f} px down')
    if not results:
        return None
    # A joint only counts as the pan joint if it moved the view SIDEWAYS, by a
    # real amount, and more sideways than up. Without those conditions the
    # choice is decided by whichever joint happened to produce the largest
    # rounding error: a joint that does not move the camera at all still
    # reports a few hundredths of a pixel, and max() will happily crown it.
    def pick(axis, exclude=()):
        best, score_ = None, 0.0
        for n, (dx, dy) in results.items():
            if n in exclude:
                continue
            mine, other = (dx, dy) if axis == 0 else (dy, dx)
            if abs(mine) < MIN_PROBE_PX or abs(mine) <= abs(other):
                continue
            if abs(mine) > score_:
                best, score_ = n, abs(mine)
        return best

    pan = pick(0)
    tilt = pick(1, exclude=(pan,) if pan else ())
    if pan is None or tilt is None:
        print(f'    could not tell which joint does what - the view moved by '
              f'less than {MIN_PROBE_PX:.0f} px on one of the axes.')
        print(f'    Point the camera at something with more detail, make sure '
              f'the arm is free to move, and try again.')
        return None
    # THE SIGN CONVENTION, once, in one place. shift_px reports how the
    # CONTENT moved, and the camera moves the other way:
    #
    #     camera pans RIGHT  -> everything slides LEFT   -> dx < 0
    #     camera tilts DOWN  -> everything slides UP     -> dy < 0
    #
    # Both are therefore "negative means the camera went the way we want".
    # Getting the pan half of that backwards is precisely the bug that made
    # the direction buttons drive away from where the user was pointing.
    roles = {
        'pan_right':  {'joint': pan,  'sign': 1 if results[pan][0] < 0 else -1},
        'tilt_down':  {'joint': tilt, 'sign': 1 if results[tilt][1] < 0 else -1},
    }
    lift = [n for n in results if n not in (pan, tilt)]
    roles['lift_up'] = {'joint': lift[0] if lift else 'shoulder', 'sign': 1}

    # Now PROVE it. The reasoning above is sound - a camera tilting down makes
    # the scene rise in the frame, so downward tilt shows as negative dy - but
    # that exact argument has been talked through backwards twice on this
    # project, and a wrong sign means the buttons drive the camera away from
    # wherever the user is pointing. So drive each role the way a button would
    # and check the picture agrees. Cheap, and it converts an argument into a
    # measurement.
    # want < 0 for both, per the convention above: driving the role forward
    # should make the CONTENT move the opposite way to the camera.
    for role, axis, want, label in (('pan_right', 0, -1, 'RIGHT'),
                                    ('tilt_down', 1, -1, 'DOWN')):
        r = roles[role]
        sid = dict(JOINTS)[r['joint']]
        try:
            start = arm.read_position(sid)
            before = settle(cam, 0.3, state).copy()
            arm.move_to(sid, int(start + r['sign'] * PROBE_DEG * TICKS_PER_DEG))
            after = settle(cam, 1.0, state).copy()
            arm.move_to(sid, int(start))
            settle(cam, 0.8, state)
        except Exception:
            continue
        d = shift_px(before, after)
        if d is None or abs(d[axis]) < MIN_PROBE_PX:
            continue
        if (d[axis] > 0) != (want > 0):
            r['sign'] = -r['sign']
            print(f'    check: {label} was backwards - flipped {r["joint"]}')
        else:
            print(f'    check: {label} confirmed on {r["joint"]}')
    return roles, results


def main():
    ap = argparse.ArgumentParser(description='One-run setup for the Lawn Sentry.')
    ap.add_argument('--print', dest='do_print', action='store_true',
                    help='just produce the printable marker and stop')
    ap.add_argument('--paper', default='a3',
                    choices=['a4', 'a3', 'letter', 'tabloid'])
    ap.add_argument('--marker', type=float, default=None)
    ap.add_argument('--source', default=None)
    ap.add_argument('--keep-joints', action='store_true',
                    help='reuse the existing joint map instead of probing')
    args = ap.parse_args()
    ensure()

    if args.do_print:
        size = args.marker or max_size_for(args.paper)
        sheet, actual = printable_sheet(side_m=size, paper=args.paper, dpi=150)
        out = DATA / 'calibration_marker.png'
        cv2.imwrite(str(out), sheet)
        print(f'\nMarker sheet: {out}')
        print(f'Print at 100% scale (NOT "fit to page") on {args.paper.upper()},')
        print(f'then MEASURE the black square - it should be {actual*100:.1f} cm.')
        print(f'Run:  python calibrate.py --marker <the size you measured>\n')
        return 0

    print('\n' + '=' * 70)
    print('SO-101 LAWN SENTRY - SETUP')
    print('=' * 70)
    W = ask('\nLawn WIDTH  (metres, across your view)', 3.5)
    L = ask('Lawn LENGTH (metres, away from you)', 2.1)
    size = args.marker or ask('Marker SIZE in metres - MEASURE the black square',
                              DEFAULT_SIZE_M)
    print(f'\nlawn {W} x {L} m, marker {size*100:.1f} cm')

    try:
        arm = TrimmerArmOpen()
    except Exception as e:
        print(f'\ncannot talk to the arm: {e}')
        return 1
    try:
        cam = FrameSource(args.source)
    except Exception as e:
        print(f'\ncamera unavailable: {e}')
        arm.close()
        return 1

    state = {'frame': None}
    sel = 0
    jog = 5.0
    step = 1                      # 1 aim, 2 probe, 3 marker, 4 place
    roles = None
    rest = None
    K = None
    locked = None
    cx, cy, yaw = W / 2.0, L / 2.0, 0.0
    H = None
    side_px = 0.0
    avg = PoseAverager(want=AVG_FRAMES)
    finder = MarkerFinder()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    if args.keep_joints:
        try:
            roles = json.loads((DATA / 'arm_map.json').read_text()).get('roles')
            if roles:
                print('\nreusing the saved joint map')
        except (OSError, ValueError):
            roles = None

    try:
        while True:
            f = cam.read()
            if f is not None:
                state['frame'] = f
            frame = state['frame']
            if frame is None:
                if cv2.waitKey(30) & 255 == ord('q'):
                    break
                continue
            if K is None:
                K, ksrc = get_K(frame.shape[1], frame.shape[0])
            disp = frame.copy()

            # ---- step 2 runs itself -------------------------------------
            if step == 2:
                banner(disp, [('STEP 2/4  working out which joint does what - '
                               'hands off', (0, 255, 255))])
                cv2.imshow(WIN, disp)
                cv2.waitKey(1)
                print('\n  probing the joints...')
                out = probe_joints(arm, cam, state)
                if out is None:
                    print('  could not read any movement - point the camera at '
                          'something with detail and try again')
                    step = 1
                    continue
                roles, _ = out
                print(f"\n  pan  = {roles['pan_right']['joint']} "
                      f"{'+' if roles['pan_right']['sign'] > 0 else '-'}")
                print(f"  tilt = {roles['tilt_down']['joint']} "
                      f"{'+' if roles['tilt_down']['sign'] > 0 else '-'}")
                print('\n  Now put the marker FLAT on the grass, close to the '
                      'camera, fully in view.')
                step = 3
                continue

            # ---- step 3: read the marker --------------------------------
            if step == 3:
                corners = finder.find(frame)
                if corners is not None:
                    side_px = min(float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
                                  for i in range(4))
                    cv2.polylines(disp, [np.asarray(corners, np.int32)], True,
                                  (0, 220, 255), 2, cv2.LINE_AA)
                    if side_px >= MIN_SIDE_PX:
                        avg.add(corners)
                        if avg.ready:
                            R, t = solve_pose(avg.mean(), size, K)
                            if R is not None:
                                locked = (R, t)
                                step = 4
                                continue
                pct = int(avg.progress * 100)
                if corners is None:
                    msg = ('MARKER NOT SEEN - more light, or move it closer',
                           (80, 140, 255))
                elif side_px < MIN_SIDE_PX:
                    msg = (f'marker only {side_px:.0f} px across - too small to '
                           f'trust. Move it CLOSER.', (80, 140, 255))
                else:
                    msg = (f'holding still  [{"#"*(pct//5)}{"."*(20-pct//5)}] '
                           f'{pct}%   {side_px:.0f} px', (0, 255, 120))
                banner(disp, [('STEP 3/4  reading the marker', (0, 255, 255)), msg])

            # ---- step 4: put the lawn on the lawn ------------------------
            if step == 4 and locked is not None:
                H = ground_from_pose(locked[0], locked[1], K, cx, cy, yaw)
                drawn = draw_lawn(disp, H, W, L) if H is not None else False
                pos = camera_from_H(H, K) if H is not None else None
                lines = [('STEP 4/4  slide the GREEN rectangle onto your real lawn',
                          (0, 255, 255))]
                if pos:
                    X, Y, Z = pos
                    sane = 0.2 < Z < 2.5
                    lines.append((f'camera: {Z:.2f} m up, {-Y:.2f} m behind the '
                                  f'near edge', (0, 255, 120) if sane else (80, 140, 255)))
                    if not sane:
                        lines.append(('^ if that is not where the camera really '
                                      'is, the marker size is wrong - press r',
                                      (80, 140, 255)))
                lines.append(('arrows/WASD move   - = fine   [ ] rotate   '
                              'f flip   r re-read', (190, 190, 190)))
                lines.append(('SPACE saves and finishes            q quits',
                              (255, 220, 90)))
                if H is not None and not drawn:
                    lines.append(('the lawn is off-screen - slide it back',
                                  (80, 140, 255)))
                banner(disp, lines)

            # ---- step 1: frame the lawn ---------------------------------
            if step == 1:
                jl = '  '.join(f"{'[' if i == sel else ' '}{i+1}:{n}"
                               f"{']' if i == sel else ' '}"
                               for i, (n, _) in enumerate(JOINTS))
                banner(disp, [
                    ('STEP 1/4  aim the camera at the lawn - this becomes the '
                     'rest pose', (0, 255, 255)),
                    (jl + f'      A/D jog {jog:.0f} deg   b/n step size',
                     (90, 255, 90)),
                    ('frame as much of the lawn as you can, then press SPACE',
                     (255, 220, 90))])

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(20) & 255
            if k == 255:
                continue
            if k == ord('q'):
                print('\ncancelled - nothing saved')
                break

            if step == 1:
                if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    sel = k - ord('1')
                elif k == ord('b'):
                    jog = min(20.0, jog + 2)
                elif k == ord('n'):
                    jog = max(1.0, jog - 2)
                elif k in (ord('a'), ord('d'), 81, 83, 2, 3):
                    d = -jog if k in (ord('a'), 81, 2) else jog
                    name, sid = JOINTS[sel]
                    try:
                        cur = arm.read_position(sid)
                        if cur is not None:
                            arm.move_to(sid, int(cur + d * TICKS_PER_DEG))
                    except Exception as e:
                        print(f'  move failed: {e}')
                elif k == 32:
                    rest = arm.read_pose()
                    print(f'\n  rest pose locked: {rest}')
                    step = 2 if roles is None else 3
                    if roles is not None:
                        print('  Put the marker FLAT on the grass, close to the '
                              'camera, fully in view.')
                continue

            if step == 3 and k == ord('r'):
                avg.reset()
                continue

            if step == 4:
                s = FINE_M if k in (ord('-'), ord('=')) else NUDGE_M
                if k in (81, 2, ord('a')):
                    cx -= s
                elif k in (83, 3, ord('d')):
                    cx += s
                elif k in (82, 0, ord('w')):
                    cy += s
                elif k in (84, 1, ord('s')):
                    cy -= s
                elif k == ord('['):
                    yaw -= ROT_DEG
                elif k == ord(']'):
                    yaw += ROT_DEG
                elif k == ord('f'):
                    yaw += 180.0
                elif k == ord('r'):
                    avg.reset(); locked = None; H = None; step = 3
                elif k in (32, 13, 10):
                    if H is None:
                        print('  the plane has not solved - press r')
                        continue
                    save(H, W, L, size, cx, cy, yaw, side_px, avg, rest, roles,
                         frame, K, ksrc)
                    break
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        try:
            arm.close()
        except Exception:
            pass
    return 0


def save(H, W, L, size, cx, cy, yaw, side_px, avg, rest, roles, frame, K, ksrc):
    """Write both config files, turning the lawn the right way round first."""
    Hn, Wn, Ln = np.asarray(H, float), float(W), float(L)
    wpx, hpx = int(frame.shape[1]), int(frame.shape[0])

    # Which corner is (0,0) is not something the picture can show you, and the
    # dashboard is not neutral about it: it pins the mower to the bottom of the
    # map and swings the view cone from there, which only makes sense with the
    # camera behind the near edge. Saved the wrong way round, every detection
    # lands rotated - and it looked perfect at save time. So it is settled
    # here, automatically, instead of being a separate tool to remember.
    best = None
    for k in range(4):
        Hk, Wk, Lk = quarter_turn(Hn, Wn, Ln, k)
        pos = camera_xy(Hk, wpx, hpx)
        if pos is None:
            continue
        sc = score(pos[0], pos[1], Wk, Lk)
        if best is None or sc < best[0]:
            best = (sc, k, Hk, Wk, Lk, pos)
    turned = 0
    if best and best[1] != 0:
        _, turned, Hn, Wn, Ln, pos = best
        print(f'\n  lawn relabelled {turned*90} degrees so the camera sits '
              f'behind the near edge')
    else:
        pos = camera_xy(Hn, wpx, hpx) or (0, 0, 0)

    quad = project(Hn, [(0, 0), (Wn, 0), (Wn, Ln), (0, Ln)])
    cal = {
        'version': 10,
        'calibration_type': 'qr_pose',
        'created': datetime.now().isoformat(timespec='seconds'),
        'lawn_width_m': Wn, 'lawn_length_m': Ln,
        'homography_image_to_lawn': Hn.tolist(),
        'image_points_tl_tr_br_bl': ([[float(a), float(b)] for a, b in quad]
                                     if quad is not None else None),
        'reference_frame_points': ([[float(a), float(b)] for a, b in quad]
                                   if quad is not None else None),
        'marker': {'size_m': size, 'centre_m': [round(cx, 3), round(cy, 3)],
                   'yaw_deg': round(yaw, 2), 'side_px': round(float(side_px), 1),
                   'frames_averaged': len(avg.buf),
                   'corner_jitter_px': (round(avg.jitter_px(), 3)
                                        if avg.jitter_px() else None),
                   'solver': 'solvePnP IPPE_SQUARE + known K'},
        'camera_position_m': [round(float(v), 3) for v in pos],
        'orientation_fixed_deg': turned * 90,
        'intrinsics_source': ksrc,
        'rest_pose': {str(k): int(v) for k, v in (rest or {}).items()} or None,
        'camera_width_px': wpx, 'camera_height_px': hpx,
        'note': 'one-run setup: marker pose + known intrinsics, lawn placed by eye',
    }
    cal_store.save(cal, label='setup')

    if roles:
        (DATA / 'arm_map.json').write_text(json.dumps(
            {'version': 3, 'roles': roles,
             'map_pan_sign': int(roles['pan_right']['sign']),
             'measured': 'automatic joint probe',
             'start_pose': {str(k): int(v) for k, v in (rest or {}).items()}},
            indent=2))

    X, Y, Z = pos
    print('\n' + '=' * 70)
    print(f'  SAVED  lawn {Wn} x {Ln} m')
    print(f'  camera {Z:.2f} m above the lawn, {-Y:.2f} m behind the near edge')
    print(f'  marker {side_px:.0f} px, averaged {len(avg.buf)} frames, '
          f'intrinsics from {ksrc}')
    if roles:
        print(f"  controls: pan = {roles['pan_right']['joint']}, "
              f"tilt = {roles['tilt_down']['joint']}")
    if rest:
        print(f'  rest pose {rest} - launch.py returns here every run')
    print('\n  TAKE THE MARKER AWAY.')
    print('\n  Run:  python launch.py')
    print('=' * 70)


def TrimmerArmOpen():
    from trimmer_arm import TrimmerArm
    a = TrimmerArm()
    if not a.read_pose():
        raise RuntimeError('the servos did not reply')
    a.clamp_enabled = False
    a.enable()
    a.set_profile(500, 25)
    return a


if __name__ == '__main__':
    raise SystemExit(main())
