"""ONE calibration, start to finish. Run this and nothing else.

Things kept drifting out of sync because the pieces were calibrated separately
and stored separately: which joint pans, where the arm rests, where the lawn
corners are, which way the map should turn. Any one of them being stale made
the whole map look wrong, and there was no way to tell which.

So this does all of it in one sitting and writes ONE file. Every step ends by
showing you the result, and nothing is saved until the last step passes.

    python calibrate/calibrate_all.py

    STEP 1  which joint PANS RIGHT                (you demonstrate it)
    STEP 2  which joint TILTS DOWN                (you demonstrate it)
    STEP 3  aim at the lawn - this becomes REST
    STEP 4  which way the map turns               (measured automatically)
    STEP 5  click the four lawn corners, then save

Keys throughout
    1 2 3 4   pick a joint      A / D   jog it      b / n   step size
    SPACE     accept this step  u  undo     r  redo this step     q  quit
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

import trimmer_config as tc
from paths import DATA, ensure
from frame_source import FrameSource
from trimmer_arm import TrimmerArm
from ground_track import GroundTracker
from calibrate_lawn_scan import order_for_rect, grass_corners, snap

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'SO-101 Lawn Sentry - full calibration'
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID)]
SID = dict(JOINTS)
TPD = 4096.0 / 360.0
CORNERS = ['CORNER 1', 'CORNER 2', 'CORNER 3', 'CORNER 4']
# Back to the method that actually worked on this rig: visual ground tracking
# stitches the views as you pan, exactly as calibrate_lawn_scan.py does. The
# encoder version is better in theory - the pan angle is exact - but the camera
# sits on the END of the arm, so a base rotation swings it through an arc
# rather than spinning it in place, and that unmodelled translation wrecked the
# fit. Tracking measures whatever the image actually did, arc included.
COLS = [(0,255,255),(0,210,255),(0,160,255),(90,255,90),(255,170,60)]
JOG_KEYS = (ord('a'), ord('d'), ord(','), ord('.'), ord('-'), ord('='), 81, 83, 2, 3)
NEG_KEYS = (ord('a'), ord(','), ord('-'), 81, 2)


def ask(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    try:
        v = float(raw.replace(',', '.'))
        return v if v > 0 else default
    except ValueError:
        return default


def banner(img, lines):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 24 * len(lines) + 10), (16, 16, 16), -1)
    for i, (txt, col, sc) in enumerate(lines):
        cv2.putText(img, txt, (10, 22 + i * 24), FONT, sc, col, 1, cv2.LINE_AA)


def shift_px(a, b):
    """How far the picture moved between two frames, in pixels (median flow)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(ga, 120, 0.01, 12)
    if p0 is None or len(p0) < 8:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None)
    if p1 is None:
        return None
    ok = st.reshape(-1) == 1
    if ok.sum() < 8:
        return None
    return float(np.median((p1[ok] - p0[ok]).reshape(-1, 2)[:, 0]))


def main():
    ensure()
    print('\n' + '=' * 68)
    print('FULL CALIBRATION - joints, rest pose, lawn, all in one file')
    print('=' * 68)
    W = ask('\nLawn WIDTH  (metres)', 3.5)
    L = ask('Lawn LENGTH (metres)', 2.1)
    print(f'\nlawn {W} x {L} m\n')

    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('servos did not reply')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(700, 30)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return
    try:
        cam = FrameSource(None)
    except Exception as e:
        print(f'camera unavailable: {e}')
        arm.close()
        return

    state = dict(step=1, sel=0, jog=4.0, roles={}, moved={}, rest=None,
                 pan_sign=None, pts=[], angs=[], frame=None, probe=None,
                 msg='', trk=None, pan_id=None, rest_raw=None, snapon=True)
    click = {'p': None}
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(p=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    STEPS = {
        1: ('STEP 1/5  make the camera PAN RIGHT, then press SPACE',
            'jog any joint until the view sweeps RIGHT'),
        2: ('STEP 2/5  make the camera TILT DOWN, then press SPACE',
            'jog until the view tips DOWN toward the ground'),
        3: ('STEP 3/5  aim at the lawn - this becomes the REST POSE',
            'frame the lawn how you want it, then press SPACE'),
        4: ('STEP 4/5  checking which way the map should turn...',
            'the arm will nudge itself - hands off'),
        5: ('STEP 5/5  click the 4 lawn corners  (A/D pans, n toggles snap)',
            'go round the edge - tracking stitches the views as you pan'),
    }

    def jog(d):
        name, sid = JOINTS[state['sel']]
        try:
            cur = arm.read_position(sid)
            if cur is not None:
                arm.move_to(sid, int(cur + d * TPD))
                state['moved'][name] = state['moved'].get(name, 0.0) + d
        except Exception as e:
            state['msg'] = f'move failed: {e}'

    try:
        while True:
            f = cam.read()
            if f is not None:
                state['frame'] = f
            if state['frame'] is None:
                if cv2.waitKey(30) & 255 == ord('q'):
                    break
                continue
            st = state['step']
            if st == 5:
                if state['trk'] is None:
                    state['trk'] = GroundTracker(np.eye(3))
                state['trk'].update(state['frame'])
            disp = state['frame'].copy()

            # ---- step 4 runs itself: nudge and watch which way the image goes
            if st == 4 and state['probe'] is None:
                state['probe'] = 'running'
                before = state['frame'].copy()
                r = state['roles']['pan_right']
                sid = SID[r['joint']]
                home_raw = arm.read_position(sid)

                def settle(seconds):
                    """Wait while STILL PUMPING the camera.

                    A plain time.sleep() stops draining the DepthAI queue; it
                    fills, and the device faults - which is what crashed it.
                    Keep reading frames and the camera stays healthy.
                    """
                    t0 = time.time()
                    last = state['frame']
                    while time.time() - t0 < seconds:
                        g = cam.read()
                        if g is not None:
                            last = g
                            state['frame'] = g
                        cv2.imshow(WIN, g if g is not None else last)
                        cv2.waitKey(1)
                    return last

                arm.move_to(sid, int(home_raw + 8 * TPD * r['sign']))
                after = settle(1.4)
                dx = shift_px(before, after)
                arm.move_to(sid, int(home_raw))     # back to exactly where it was
                settle(1.2)
                # panning RIGHT makes the scene slide LEFT (negative dx)
                if dx is None:
                    # too little texture to measure - say so rather than
                    # silently picking a sign and getting the map backwards
                    state['pan_sign'] = 1
                    state['msg'] = ('could not measure the pan direction - '
                                    'assuming +1; press r to retry')
                    print('  could not measure the pan direction (not enough '
                          'texture in view). Assuming +1.')
                else:
                    state['pan_sign'] = 1 if dx < 0 else -1
                    print(f'  scene moved {dx:+.1f} px when panning right '
                          f'-> map sign {state["pan_sign"]:+d}')
                state['step'] = 5
                continue

            if st == 5 and click['p'] and len(state['pts']) < 4:
                p = (float(click['p'][0]), float(click['p'][1]))
                how = 'raw'
                if state['snapon']:
                    gc, gk = grass_corners(state['frame'])
                    p, how = snap(p, gc, gk)
                q = cv2.perspectiveTransform(
                    np.array([[[p[0], p[1]]]], np.float32),
                    state['trk'].G.astype(np.float32))[0, 0]
                state['pts'].append([float(q[0]), float(q[1])])
                print(f"  {CORNERS[len(state['pts'])-1]}  image({p[0]:.0f},"
                      f"{p[1]:.0f}) [{how}] -> reference({q[0]:.0f},{q[1]:.0f})")
            click['p'] = None

            # Show each stored point WHERE IT SHOULD BE FROM HERE. A dot frozen
            # to a screen pixel tells you nothing; one that stays on its corner
            # while you pan proves the angle stitching works before you commit.
            shown = []
            if st == 5 and state['pts']:
                try:
                    Ginv = np.linalg.inv(state['trk'].G)
                    for q in state['pts']:
                        p = cv2.perspectiveTransform(
                            np.array([[[q[0], q[1]]]], np.float32),
                            Ginv.astype(np.float32))[0, 0]
                        shown.append([float(p[0]), float(p[1])])
                except np.linalg.LinAlgError:
                    shown = []
            for i, p in enumerate(shown):
                if not (np.isfinite(p[0]) and np.isfinite(p[1])):
                    continue
                cv2.circle(disp, (int(p[0]), int(p[1])), 7, COLS[i], -1, cv2.LINE_AA)
                cv2.putText(disp, CORNERS[i], (int(p[0]) + 10, int(p[1]) - 8),
                            FONT, 0.45, COLS[i], 2, cv2.LINE_AA)
            if len(shown) >= 2:
                cv2.polylines(disp, [np.asarray(shown[:4], np.int32)],
                              len(shown) >= 4, (0, 255, 255), 2, cv2.LINE_AA)

            title, hint = STEPS[st]
            jl = '  '.join(f"{'[' if i == state['sel'] else ' '}{i+1}:{n}"
                           f"{']' if i == state['sel'] else ' '}"
                           for i, (n, _) in enumerate(JOINTS))
            lines = [(title, (0, 255, 255), 0.6), (hint, (200, 200, 200), 0.45)]
            if st in (1, 2, 3):
                lines.append((f"{jl}   jog {state['jog']:.0f}deg  (A/D, b/n)",
                              (90, 255, 90), 0.45))
            if st == 5:
                tk = state['trk'].status() if state['trk'] else {}
                lines.append((f"tracking {'LOST - press r' if tk.get('lost') else 'OK'}"
                              f"   feats {tk.get('tracked', 0)}"
                              f"   snap {'on' if state['snapon'] else 'off'}",
                              (0, 220, 90) if not tk.get('lost') else (0, 120, 255),
                              0.45))
                nxt = (CORNERS[len(state['pts'])] if len(state['pts']) < 4
                       else 'press SPACE to save')
                lines.append((f'next: {nxt}', (90, 255, 90), 0.5))
            if state['msg']:
                lines.append((state['msg'], (80, 140, 255), 0.45))
            banner(disp, lines)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(20) & 255
            if k == 255:
                continue
            if k == ord('q'):
                print('cancelled - nothing saved')
                return
            if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                state['sel'] = k - ord('1')
            elif k == ord('b'):
                state['jog'] = min(15.0, state['jog'] + 2)
            elif k == ord('n'):
                state['jog'] = max(1.0, state['jog'] - 2)
            elif k in JOG_KEYS and st in (1, 2, 3):
                jog(-state['jog'] if k in NEG_KEYS else state['jog'])
            elif k in JOG_KEYS and st == 5:
                # pan to reach a corner outside the view; the angle is recorded
                # with each click, so this does not disturb the calibration
                d = -state['jog'] if k in NEG_KEYS else state['jog']
                try:
                    cur = arm.read_position(state['pan_id'])
                    arm.move_to(state['pan_id'], int(cur + d * TPD))
                except Exception as e:
                    state['msg'] = f'pan failed: {e}'
            elif k == ord('u') and st == 5 and state['pts']:
                state['pts'].pop()
            elif k == ord('n') and st == 5:
                state['snapon'] = not state['snapon']
                state['msg'] = f"grass snapping {'on' if state['snapon'] else 'off'}"
            elif k == ord('r'):
                if st == 5:
                    state['pts'] = []
                    if state['trk'] is not None:
                        state['trk'].anchor()
                    state['msg'] = 'tracking restarted, corners cleared'
                    if state['msg'].startswith('could not measure'):
                        state['probe'] = None       # retry the direction probe
                        state['step'] = 4
                state['moved'] = {}
                state['msg'] = 'redo this step'
            elif k == 32:
                state['msg'] = ''
                if st in (1, 2):
                    if not state['moved']:
                        state['msg'] = 'jog a joint first'
                        continue
                    nm = max(state['moved'], key=lambda n: abs(state['moved'][n]))
                    net = state['moved'][nm]
                    if abs(net) < 0.5:
                        state['msg'] = 'that joint is back where it started'
                        continue
                    key = 'pan_right' if st == 1 else 'tilt_down'
                    state['roles'][key] = {'joint': nm,
                                           'sign': 1 if net > 0 else -1}
                    print(f"  {key} = {nm} {'+' if net > 0 else '-'}")
                    state['moved'] = {}
                    state['step'] += 1
                elif st == 3:
                    state['rest'] = arm.read_pose()
                    r = state['roles']['pan_right']
                    state['pan_id'] = SID[r['joint']]
                    state['rest_raw'] = state['rest'][state['pan_id']]
                    print(f"  REST POSE = {state['rest']}")
                    state['step'] = 4
                elif st == 5:
                    if len(state['pts']) != 4:
                        state['msg'] = (f"need 4 corners, have "
                                        f"{len(state['pts'])}")
                        continue
                    # Same solve as calibrate_lawn_scan.py: order the four
                    # reference-frame points into far/near pairs, then map them
                    # onto the known rectangle. order_for_rect sorts them, so
                    # the click order does not have to be perfect.
                    src = order_for_rect(state['pts'])
                    if abs(cv2.contourArea(src)) < 2000:
                        state['msg'] = 'those 4 points are nearly in a line - redo'
                        continue
                    dst = np.array([[0, L], [W, L], [W, 0], [0, 0]], np.float32)
                    try:
                        H = cv2.getPerspectiveTransform(src, dst)
                    except cv2.error:
                        state['msg'] = 'degenerate - press r and redo'
                        continue
                    if not np.all(np.isfinite(H)):
                        state['msg'] = 'bad geometry - press r and redo'
                        continue
                    got = cv2.perspectiveTransform(src.reshape(1, -1, 2),
                                                   H).reshape(-1, 2)
                    err = float(np.max(np.linalg.norm(got - dst, axis=1))) * 100
                    tilt = None
                    print(f'  corner reprojection error: {err:.1f} cm')
                    data = {
                        'version': 7,
                        'calibration_type': 'full_wizard',
                        'created': datetime.now().isoformat(timespec='seconds'),
                        'lawn_width_m': W, 'lawn_length_m': L,
                        'image_points_tl_tr_br_bl': [[float(a), float(b)]
                                                     for a, b in src],
                        'homography_image_to_lawn': H.astype(float).tolist(),
                        'rest_pose': {str(k2): int(v)
                                      for k2, v in state['rest'].items()},
                        'roles': state['roles'],
                        'map_pan_sign': int(state['pan_sign']),
                        'corner_error_cm': round(float(err), 2),
                        'stitching': 'visual ground tracking',
                        'camera_width_px': int(state['frame'].shape[1]),
                        'camera_height_px': int(state['frame'].shape[0]),
                    }
                    import cal_store
                    cal_store.save(data, label='full_wizard')
                    # the arm map lives in its own file for the control pad
                    roles = dict(state['roles'])
                    roles.setdefault('lift_up', {'joint': 'shoulder', 'sign': 1})
                    (DATA / 'arm_map.json').write_text(json.dumps(
                        {'version': 2, 'roles': roles,
                         'map_pan_sign': int(state['pan_sign']),
                         'start_pose': {str(k2): int(v)
                                        for k2, v in state['rest'].items()}},
                        indent=2))
                    cv2.imwrite(str(DATA / 'calibration_preview.jpg'), disp)
                    print('\n' + '=' * 68)
                    print('  SAVED  data/lawn_calibration.json  +  data/arm_map.json')
                    for k2, v in state['roles'].items():
                        print(f"    {k2:10s} = {v['joint']} "
                              f"{'+' if v['sign'] > 0 else '-'}")
                    print(f"    map pan sign = {state['pan_sign']:+d}")
                    print(f"    rest pose    = {state['rest']}")
                    print(f"    corner error = {err:.1f} cm")
                    print('\n  Everything is in step now. Run:  python launch.py')
                    print('=' * 68)
                    return
    finally:
        # camera first, then the arm - by here nothing is reading frames, so a
        # short wait for the arm to settle cannot stall the device queue
        cv2.destroyAllWindows()
        try:
            cam.stop()
        except Exception:
            pass
        try:
            if state.get('rest'):
                arm.move_pose({int(k): int(v)
                               for k, v in state['rest'].items()})
                time.sleep(0.6)
            arm.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
