"""Set the rest pose AND the lawn plane together, so they can never disagree.

A ground homography is only true for the camera pose it was measured at. When
the camera lives on an arm, "the camera pose" means "the joint angles" - so a
calibration measured at one pose and used at another describes a plane the
camera is no longer looking at. That is why the lawn kept landing in the wrong
place: the two were recorded separately and drifted apart.

So they are recorded together here, in one file, in one sitting:

    1. jog the arm until the camera sees the lawn the way you want
    2. lock that as the REST POSE
    3. click the four lawn corners
    4. type the real width and length

`launch.py` then drives the arm back to that exact pose before scanning, so the
plane is always measured from where it was defined. Pan away during the scan
and the encoder tracks it; come back to rest and it lines up again.

    python calibrate/calibrate_rest_plane.py

Keys
    1 2 3 4     choose base / shoulder / elbow / wrist
    A / D       jog the chosen joint          b / n   bigger / smaller step
    SPACE       lock this as the rest pose
    click       place the 4 corners, then the CENTRE of the lawn
                (pan with A/D if a corner is off-screen)
    u           undo the last corner          r   back to jogging
    s           save                          q   quit
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
import time
from datetime import datetime

import cv2
import numpy as np

import math

import trimmer_config as tc
from paths import DATA, ensure
from frame_source import FrameSource
from trimmer_arm import TrimmerArm
from camera_intrinsics import get as get_K, axis_from_tilt, rotation_homography_axis

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Rest pose + lawn plane'
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID)]
NAMES = ['NEAR-LEFT', 'NEAR-RIGHT', 'FAR-RIGHT', 'FAR-LEFT',
         'CENTRE of the lawn (where the diagonals cross)']
COLS = [(0, 255, 255), (0, 210, 255), (0, 160, 255), (90, 255, 90),
        (255, 170, 60)]

# Four corners and a homography are an exact fit for ANY assumed camera tilt,
# so four points alone cannot say how far the camera is tilted down - and the
# tilt is what turns a pan angle into an image shift. A fifth known point
# breaks that tie.
#
# It has to be the CENTRE, not a point on an edge. An edge midpoint sits
# between two corners that are already pinned, so it moves very little as the
# tilt changes and a badly wrong tilt still satisfies it - measured: it passed
# to 0.1 cm while being 15 degrees out and 36 cm wrong across the lawn. The
# centre is off every edge and is sensitive to exactly the error we are
# solving for.
N_POINTS = 5
TICKS_PER_DEG = 4096.0 / 360.0


def referred(pts, angs, K, tilt, sign=1.0):
    """Bring every click back into the REST view.

    Each was clicked at a known pan angle, and for a camera turning about its
    mount the view-to-view homography is exactly K R K^-1. The rotation axis is
    the mount's vertical expressed in camera coordinates, which is where the
    tilt comes in.
    """
    axis = axis_from_tilt(tilt)
    out = []
    for (x, y), a in zip(pts, angs):
        if abs(a) < 1e-6:
            out.append([float(x), float(y)])
            continue
        Hp = rotation_homography_axis(K, float(sign) * float(a), axis)
        q = cv2.perspectiveTransform(
            np.array([[[float(x), float(y)]]], np.float32),
            Hp.astype(np.float32))[0, 0]
        out.append([float(q[0]), float(q[1])])
    return out


def solve_rest(pts, angs, K, W, L, sign=None):
    """Homography to lawn metres, searching for the camera tilt.

    Four corners fit ANY tilt exactly, so the tilt is found using the fifth
    point: the correct tilt is the one that puts the middle of the near edge
    at (W/2, 0). Returns (H, tilt_deg, midpoint error in metres).
    """
    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32)
    target = np.array([W / 2.0, L / 2.0])
    best = None
    cands = []
    # Which way a rising servo count turns the camera is NOT something one
    # check point can decide - tried it, and a wrong sign still satisfied the
    # centre to within a centimetre while being two metres out. It is already
    # known from calibrate_joints.py, where you demonstrated which direction
    # pans right, so take it from there rather than guessing.
    signs = (1.0, -1.0) if sign is None else (float(sign),)
    grid = np.arange(0.0, 85.1, 2.0)
    for _ in range(2):
        for sign in signs:
          for tilt in grid:
            ref = referred(pts, angs, K, float(tilt), sign)
            src = np.asarray(ref[:4], np.float32)
            if abs(cv2.contourArea(src)) < 500:
                continue
            try:
                H = cv2.getPerspectiveTransform(src, dst)
            except cv2.error:
                continue
            if not np.all(np.isfinite(H)):
                continue
            mid = cv2.perspectiveTransform(
                np.array([[ref[4]]], np.float32), H)[0, 0]
            if not np.isfinite(mid).all():
                continue
            e = float(np.linalg.norm(np.asarray(mid, float) - target))
            cands.append((e, float(tilt), H, float(sign)))
        if not cands:
            return None, None, None
        # More than one tilt can satisfy the centre point - a steeply tilted
        # camera can be made to fit almost as well as the true one. Among
        # solutions that are equally good on the check, take the SHALLOWEST:
        # a camera looking across a lawn is tilted tens of degrees, not
        # seventy, and the steep solutions were measured to be metres wrong
        # while still passing the check.
        emin = min(c[0] for c in cands)
        good = [c for c in cands if c[0] <= emin + 0.02 + emin * 0.5]
        best = min(good, key=lambda c: c[1])
        grid = np.arange(max(0.0, best[1] - 2.0), min(85.0, best[1] + 2.0) + .01, 0.2)
        signs = (best[3],)
        cands = []
    e, tilt, H, sign = best
    solve_rest.sign = sign
    return H, tilt, e


def ask(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    try:
        v = float(raw.replace(',', '.'))
        return v if v > 0 else default
    except ValueError:
        return default


def main():
    ensure()
    print('\n' + '=' * 66)
    print('REST POSE + LAWN PLANE')
    print('=' * 66)
    print('\nThe plane is measured FROM the rest pose, and saved with it, so')
    print('the two can never drift apart again.\n')
    W = ask('Lawn WIDTH  (metres)', 3.5)
    L = ask('Lawn LENGTH (metres)', 2.1)
    print(f'\nlawn {W} x {L} m\n')

    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('no reply from the servos')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(260, 14)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return
    try:
        cam = FrameSource(None)
    except Exception as e:
        print(f'camera unavailable: {e}')
        arm.close()
        return

    stage = 'pose'            # 'pose' -> 'corners'
    sel, step = 0, 3.0
    rest = None
    pts = []                  # image points, already referred back to REST view
    raw_pts = []              # what was clicked, for redrawing
    angs = []                 # pan angle at each click
    K = None
    pan_id = None
    pan_sign = -1.0
    click = {'p': None}
    last_click = {'t': 0.0, 'p': None}
    frame = None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(p=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    try:
        while True:
            f = cam.read()
            if f is not None:
                frame = f
            if frame is None:
                if cv2.waitKey(30) & 255 == ord('q'):
                    break
                continue

            if stage == 'corners' and click['p'] and len(pts) < N_POINTS:
                px, py = float(click['p'][0]), float(click['p'][1])
                # Ignore a double-click. One stray second press shifts every
                # label after it, and you only find out at the end when the
                # count is wrong - which is exactly what happened.
                now_t = time.monotonic()
                lp = last_click['p']
                if (now_t - last_click['t'] < 0.45 and lp is not None
                        and abs(px - lp[0]) < 25 and abs(py - lp[1]) < 25):
                    click['p'] = None
                    print('  (ignored a double-click)')
                    continue
                last_click['t'], last_click['p'] = now_t, (px, py)
                try:
                    a_now = arm.read_position(pan_id)
                except Exception:
                    a_now = None
                d_deg = 0.0 if (a_now is None or rest is None) else \
                    (a_now - rest.get(pan_id, a_now)) * 360.0 / 4096.0
                pts.append([px, py])
                raw_pts.append([px, py])
                angs.append(d_deg)
                print(f'  {NAMES[len(pts)-1]} at {px:.0f},{py:.0f}  '
                      f'(panned {d_deg:+.1f} deg from rest)')
            click['p'] = None

            disp = frame.copy()
            h, w = disp.shape[:2]
            # Show each stored point where it should appear FROM HERE. A dot
            # that stays glued to the same screen pixel while the camera turns
            # tells you nothing; one that stays on its lawn corner shows the
            # angle stitching is working before you commit to it.
            shown = list(pts)
            if stage == 'corners' and K is not None and pts:
                try:
                    a_live = arm.read_position(pan_id)
                    d_live = (a_live - rest.get(pan_id, a_live)) * 360.0 / 4096.0
                    sgn = pan_sign
                    axis = axis_from_tilt(35.0)      # rough; only for the preview
                    shown = []
                    for (px0, py0), a0 in zip(pts, angs):
                        dd = (a0 - d_live) * sgn
                        if abs(dd) < 1e-6:
                            shown.append([px0, py0]); continue
                        Hp = rotation_homography_axis(K, float(dd), axis)
                        q = cv2.perspectiveTransform(
                            np.array([[[px0, py0]]], np.float32),
                            Hp.astype(np.float32))[0, 0]
                        shown.append([float(q[0]), float(q[1])])
                except Exception:
                    shown = list(pts)
            for i, p in enumerate(shown):
                cv2.circle(disp, (int(p[0]), int(p[1])), 7, COLS[i], -1, cv2.LINE_AA)
                cv2.circle(disp, (int(p[0]), int(p[1])), 12, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, NAMES[i], (int(p[0]) + 12, int(p[1]) - 8),
                            FONT, 0.5, COLS[i], 2, cv2.LINE_AA)
            if len(shown) >= 2:
                cv2.polylines(disp, [np.asarray(shown, np.int32)], len(shown) == 4,
                              (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(disp, (0, 0), (w, 82), (16, 16, 16), -1)
            if stage == 'pose':
                cv2.putText(disp, 'STEP 1  aim the camera at the lawn, then press SPACE',
                            (10, 26), FONT, 0.66, (0, 255, 255), 2)
                jl = '  '.join(f"{'[' if i == sel else ' '}{i+1}:{n}"
                               f"{']' if i == sel else ' '}"
                               for i, (n, _) in enumerate(JOINTS))
                try:
                    rawv = arm.read_position(JOINTS[sel][1])
                except Exception:
                    rawv = None
                cv2.putText(disp, f'{jl}   raw={rawv}', (10, 50), FONT, 0.5,
                            (90, 255, 90), 1)
                cv2.putText(disp, f'A/D jog {step:.0f}deg   b/n step   SPACE lock   q quit',
                            (10, 72), FONT, 0.45, (170, 170, 170), 1)
            else:
                nxt = NAMES[len(pts)] if len(pts) < N_POINTS else 'press S to SAVE'
                cv2.putText(disp, f'STEP 2  click {nxt}', (10, 26), FONT, 0.66,
                            (0, 255, 255) if len(pts) < N_POINTS else (90, 255, 90), 2)
                try:
                    a_now = arm.read_position(pan_id)
                    d_now = (a_now - rest.get(pan_id, a_now)) * 360.0 / 4096.0
                except Exception:
                    d_now = 0.0
                cv2.putText(disp,
                            f'PAN with A/D if a corner is off-screen   '
                            f'now {d_now:+.1f} deg from rest',
                            (10, 50), FONT, 0.5,
                            (90, 255, 90) if abs(d_now) < 1 else (0, 210, 255), 1)
                cv2.putText(disp, 'u undo   r restart   s save   q quit',
                            (10, 72), FONT, 0.45, (170, 170, 170), 1)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(20) & 255
            if k == 255:
                continue
            if k == ord('q'):
                print('cancelled - nothing saved')
                break

            if stage == 'pose':
                if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    sel = k - ord('1')
                elif k == ord('b'):
                    step = min(15.0, step + 2)
                elif k == ord('n'):
                    step = max(1.0, step - 2)
                elif k in (ord('a'), ord('d'), ord(','), ord('.'),
                           ord('-'), ord('='), 81, 83, 2, 3):
                    d = -step if k in (ord('a'), ord(','), ord('-'), 81, 2) else step
                    _, sid = JOINTS[sel]
                    try:
                        cur = arm.read_position(sid)
                        if cur is not None:
                            arm.move_to(sid, int(cur + d * TICKS_PER_DEG))
                    except Exception as e:
                        print(f'  move failed: {e}')
                elif k == 32:
                    rest = arm.read_pose()
                    if not rest:
                        print('  could not read the pose - try again')
                        continue
                    K, _ks = get_K(frame.shape[1], frame.shape[0], quiet=True)
                    try:
                        from arm_control import load_map
                        _roles, _mapped = load_map()
                        pan_id = {'base': tc.BASE_ID, 'shoulder': tc.SHOULDER_ID,
                                  'elbow': tc.ELBOW_ID, 'wrist': tc.WRIST_ID}.get(
                            (_roles.get('pan_right') or {}).get('joint', 'base'),
                            tc.BASE_ID)
                    except Exception:
                        pan_id, pan_sign, _mapped = tc.BASE_ID, -1.0, False
                    # +1 means a rising servo count pans the view RIGHT, which
                    # makes the ground appear to move LEFT - hence the negation.
                    pan_sign = -float((_roles.get('pan_right') or {}).get('sign', 1))
                    if not _mapped:
                        print('  WARNING: no arm_map.json, so the pan direction is')
                        print('  a guess. Run calibrate/calibrate_joints.py first,')
                        print('  or the corners will be stitched the wrong way.')
                    print(f'  pan joint = id {pan_id}, pan sign {pan_sign:+.0f}, '
                          f'intrinsics {_ks}')
                    stage = 'corners'
                    print(f'\n  REST POSE locked: {rest}')
                    print('  Now click the four lawn corners.\n')
            else:
                if k in (ord('a'), ord('d'), ord(','), ord('.'),
                         ord('-'), ord('='), 81, 83, 2, 3):
                    # pan to reach a corner that is outside the current view;
                    # the angle is recorded with each click and undone later
                    dd = -step if k in (ord('a'), ord(','), ord('-'), 81, 2) else step
                    try:
                        cur = arm.read_position(pan_id)
                        if cur is not None:
                            arm.move_to(pan_id, int(cur + dd * TICKS_PER_DEG))
                    except Exception as e:
                        print(f'  pan failed: {e}')
                elif k == ord('b'):
                    step = min(15.0, step + 2)
                elif k == ord('n'):
                    step = max(1.0, step - 2)
                elif k == ord('u') and pts:
                    pts.pop(); raw_pts.pop(); angs.pop()
                elif k == ord('r'):
                    stage, pts, raw_pts, angs = 'pose', [], [], []
                    print('  back to aiming - corners cleared')
                elif k == ord('s'):
                    if len(pts) != N_POINTS:
                        print(f'  need {N_POINTS} points, have {len(pts)}')
                        continue
                    H, tilt, err = solve_rest(pts, angs, K, W, L, pan_sign)
                    if H is None:
                        print('  could not solve - press r and redo')
                        continue
                    src = np.asarray(referred(pts, angs, K, tilt), np.float32)
                    print(f'\n  camera tilt {tilt:.1f} deg below horizontal')
                    print(f'  centre check point lands {err*100:.1f} cm from '
                          f'where it should')
                    if err > 0.25:
                        print('  that is a long way out. Likely causes: a corner')
                        print('  clicked in the wrong order, or the centre point')
                        print('  not actually at the middle of the lawn.')
                        if input('  save anyway? [y/N]: ').strip().lower() != 'y':
                            continue
                    data = {
                        'version': 6,
                        'calibration_type': 'rest_pose_plane',
                        'created': datetime.now().isoformat(timespec='seconds'),
                        'lawn_width_m': W,
                        'lawn_length_m': L,
                        'image_points_tl_tr_br_bl': [[float(a), float(b)]
                                                     for a, b in src],
                        'homography_image_to_lawn': H.astype(float).tolist(),
                        # THE POINT OF THIS FILE: the plane and the pose it was
                        # measured at, stored together and used together.
                        'rest_pose': {str(k2): int(v) for k2, v in rest.items()},
                        'pan_joint_id': int(pan_id),
                        'camera_tilt_deg': round(float(tilt), 2),
                        'pan_sign': float(getattr(solve_rest, 'sign', 1.0)),
                        'pan_angles_deg': [round(float(a), 2) for a in angs],
                        'centre_check_error_m': round(float(err), 4),
                        'camera_width_px': int(frame.shape[1]),
                        'camera_height_px': int(frame.shape[0]),
                    }
                    import cal_store
                    out = DATA / 'lawn_calibration.json'
                    cal_store.save(data, label='rest_plane')
                    cv2.imwrite(str(DATA / 'rest_pose_preview.jpg'), disp)
                    print(f'\n  SAVED {out.name}')
                    print(f'  lawn {W} x {L} m, measured at rest pose {rest}')
                    print('\n  launch.py now returns the arm to this pose before')
                    print('  scanning, so the plane always matches the view.\n')
                    break
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        try:
            arm.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
