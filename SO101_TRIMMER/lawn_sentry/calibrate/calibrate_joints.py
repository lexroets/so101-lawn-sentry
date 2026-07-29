"""Teach the system which joint moves the camera which way.

I had assumed base = pan, shoulder = height, wrist = tilt. That was true for the
old mounting and is not a safe assumption for any other, so instead of guessing,
this asks.

You jog each joint while watching the live picture and confirm when the view
moves the way the prompt describes. What gets saved is not "shoulder" but "the
thing that makes the camera look UP", which is what the buttons actually mean.

    python lawn_sentry/calibrate/calibrate_joints.py

Keys
    1 2 3 4     choose base / shoulder / elbow / wrist
    A / D       jog the chosen joint (arrow keys and , . also work)
    ENTER       yes, THAT is the movement being asked for
    s           skip this one
    b           bigger step        n   smaller step
    r           back to where we started
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
import json

import cv2
import numpy as np

import trimmer_config as tc
from paths import DATA, ensure
from frame_source import FrameSource
from trimmer_arm import TrimmerArm

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Joint calibration - which joint moves the camera which way?'
OUT_NAME = 'arm_map.json'

JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID)]

# Each role is described by what the PICTURE does, not by which servo turns.
ROLES = [
    ('pan_right',  'PAN RIGHT',  'the view should sweep to the RIGHT'),
    ('tilt_down',  'TILT DOWN',  'the view should tip DOWN toward the ground'),
    ('lift_up',    'RAISE',      'the camera should get HIGHER / see further'),
]

TICKS_PER_DEG = 4096.0 / 360.0


def main():
    ensure()
    print('\n' + '=' * 68)
    print('JOINT CALIBRATION - teach the controls which way is which')
    print('=' * 68)
    print('\nYou will be asked for three movements. For each one:')
    print('  pick a joint with 1-4, jog it with , and . while watching the')
    print('  picture, and press ENTER when the view moves as described.')
    print('\nNothing is assumed about how the camera is mounted.\n')

    try:
        arm = TrimmerArm()
        start = arm.read_pose()
        if not start:
            raise RuntimeError('no reply from the servos')
        arm.clamp_enabled = False      # full travel while learning the mapping
        arm.enable()
        arm.set_profile(260, 14)
        print('[joints] software limits OFF - the arm can reach its full range.')
        print('         Keep the workspace clear.')
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return
    print(f'start pose: {start}\n')

    try:
        cam = FrameSource(None)
    except Exception as e:
        print(f'camera unavailable ({e}) - continuing without video')
        cam = None

    sel = 0
    step = 3.0
    result = {}
    frame = None
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        for key, label, hint in ROLES:
            print(f'--- {label}: {hint}')
            done = False
            moved = {}                  # joint -> net degrees moved this round
            while not done:
                if cam is not None:
                    f = cam.read()
                    if f is not None:
                        frame = f
                disp = (frame.copy() if frame is not None
                        else np.zeros((360, 640, 3), np.uint8))
                h, w = disp.shape[:2]
                cv2.rectangle(disp, (0, 0), (w, 104), (16, 16, 16), -1)
                cv2.putText(disp, f'MAKE THE CAMERA: {label}', (10, 26),
                            FONT, 0.72, (0, 255, 255), 2)
                cv2.putText(disp, hint, (10, 50), FONT, 0.5, (200, 200, 200), 1)
                jl = '  '.join(
                    f"{'[' if i == sel else ' '}{i+1}:{n}{']' if i == sel else ' '}"
                    for i, (n, _) in enumerate(JOINTS))
                try:
                    _raw = arm.read_position(JOINTS[sel][1])
                except Exception:
                    _raw = None
                cv2.putText(disp, jl + (f'    raw={_raw}' if _raw is not None
                                        else '    raw=?'),
                            (10, 74), FONT, 0.5, (90, 255, 90), 1)
                cv2.putText(disp,
                            f'A/D or arrows: jog {step:.0f}deg   ENTER=this is it   '
                            f's=skip  b/n=step  r=reset  q=quit',
                            (10, 96), FONT, 0.45, (170, 170, 170), 1)
                cv2.imshow(WIN, disp)

                k = cv2.waitKey(30) & 255
                if k == 255:
                    continue
                if k == ord('q'):
                    print('cancelled - nothing saved')
                    return
                if k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    sel = k - ord('1')
                elif k == ord('b'):
                    step = min(15.0, step + 2)
                elif k == ord('n'):
                    step = max(1.0, step - 2)
                elif k in (ord(','), ord('.'), ord('a'), ord('d'),
                           ord('-'), ord('='), ord('+'), 81, 83, 2, 3):
                    # generous: , . / a d / - = / arrow keys all jog, because
                    # which codes OpenCV delivers depends on the build and the
                    # keyboard, and a control you cannot press is useless
                    neg = k in (ord(','), ord('a'), ord('-'), 81, 2)
                    d = -step if neg else step
                    name, sid = JOINTS[sel]
                    try:
                        cur = arm.read_position(sid)
                        if cur is not None:
                            arm.move_to(sid, int(cur + d * TICKS_PER_DEG))
                            moved[name] = moved.get(name, 0.0) + d
                    except Exception as e:
                        print(f'  move failed: {e}')
                elif k == ord('r'):
                    try:
                        arm.move_pose(start)
                        moved.clear()
                        print('  back to the start pose')
                    except Exception:
                        pass
                elif k == ord('s'):
                    print(f'  skipped {label}\n')
                    done = True
                elif k in (13, 10):
                    # whichever joint was moved most, and in which direction
                    if not moved:
                        print('  move a joint first, then press ENTER')
                        continue
                    name = max(moved, key=lambda n: abs(moved[n]))
                    net = moved[name]
                    if abs(net) < 0.5:
                        print('  that joint ended up back where it started - '
                              'jog it one way only')
                        continue
                    result[key] = {'joint': name, 'sign': 1 if net > 0 else -1}
                    print(f'  {label} = {name} {"+" if net > 0 else "-"}  '
                          f'(moved {net:+.0f} deg)\n')
                    done = True

        if not result:
            print('nothing recorded - not saved')
            return

        # tidy up: return to where we found it
        try:
            arm.move_pose(start)
        except Exception:
            pass

        out = DATA / OUT_NAME
        out.write_text(json.dumps({
            'version': 1,
            'note': 'which joint moves the camera which way, for THIS mounting',
            'roles': result,
            'start_pose': {str(k): int(v) for k, v in start.items()},
        }, indent=2))
        print('=' * 68)
        for k, v in result.items():
            print(f'  {k:10s} -> {v["joint"]} {"+" if v["sign"] > 0 else "-"}')
        missing = [k for k, _, _ in ROLES if k not in result]
        if missing:
            print(f'  not set: {", ".join(missing)} (those buttons will do nothing)')
        print(f'\n  saved {out}')
        print('  The web app buttons now use this. Run:  python launch.py')
        print('=' * 68)
    finally:
        cv2.destroyAllWindows()
        if cam is not None:
            cam.stop()
        try:
            arm.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
