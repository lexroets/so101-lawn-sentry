"""Thirty seconds before the demo: is everything actually ready?

Every failure this system has had in front of someone came from a thing that
was checkable beforehand - a calibration measured at a pose the arm was no
longer in, a lawn saved a quarter turn round, a port still held by a previous
run. None of them announced themselves; they looked like the robot being
broken.

So this checks them, in the order they would ruin the demo, and says GO or
says exactly what to fix. It touches nothing and starts nothing.

    python tools/demo_check.py
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
import datetime
import socket
import time

from paths import DATA, SESSIONS

PROBLEMS = []
NOTES = []


def ok(msg):
    print(f'  [ok]   {msg}')


def bad(msg, fix):
    print(f'  [FIX]  {msg}')
    PROBLEMS.append((msg, fix))


def note(msg):
    print(f'  [note] {msg}')
    NOTES.append(msg)


def port_busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(('127.0.0.1', port)) == 0


def main():
    print('\n' + '=' * 68)
    print('PRE-DEMO CHECK')
    print('=' * 68)

    # 1. the calibration -----------------------------------------------------
    print('\n1. Lawn calibration')
    cal = None
    try:
        import cal_store
        cal = cal_store.load()
    except Exception as e:
        bad(f'cannot read the calibration ({e})',
            'python lawn_sentry/calibrate/calibrate_qr.py')
    if cal:
        W, L = cal.get('lawn_width_m'), cal.get('lawn_length_m')
        kind = cal.get('calibration_type', '?')
        age_h = (time.time() - (DATA / 'lawn_calibration.json').stat().st_mtime) / 3600
        ok(f'{kind}, {W} x {L} m, measured {age_h:.1f} h ago')
        missing = [k for k in ('homography_image_to_lawn', 'image_points_tl_tr_br_bl')
                   if not cal.get(k)]
        if missing:
            bad(f'missing {", ".join(missing)} - the scan will crash on startup',
                'python lawn_sentry/calibrate/calibrate_qr.py')
        else:
            ok('has everything the scan reads')

        if cal.get('rest_pose'):
            ok(f'rest pose recorded {cal["rest_pose"]} - the arm returns here')
        else:
            bad('NO rest pose - the plane is only right if the arm happens to '
                'be where this was measured',
                'python lawn_sentry/calibrate/calibrate_lawn_scan.py')

        # 2. is the lawn the right way round? --------------------------------
        print('\n2. Lawn orientation')
        try:
            import numpy as np
            from fix_orientation import quarter_turn, camera_xy, score
            H = np.asarray(cal['homography_image_to_lawn'], float)
            wpx = int(cal.get('camera_width_px', 640))
            hpx = int(cal.get('camera_height_px', 480))
            best, cur = None, None
            for k in range(4):
                Hk, Wk, Lk = quarter_turn(H, float(W), float(L), k)
                pos = camera_xy(Hk, wpx, hpx)
                if pos is None:
                    continue
                sc = score(pos[0], pos[1], Wk, Lk)
                if k == 0:
                    cur = pos
                if best is None or sc < best[0]:
                    best = (sc, k, pos)
            if cur:
                ok(f'camera reads {cur[2]:.2f} m above the lawn, '
                   f'{-cur[1]:.2f} m behind the near edge')
                if not (0.2 < cur[2] < 2.5):
                    bad(f'that height is not believable - the marker size was '
                        f'probably wrong',
                        'python lawn_sentry/calibrate/calibrate_qr.py')
            if best and best[1] != 0:
                bad(f'the lawn is {best[1]*90} degrees round - detections will '
                    f'land on the map rotated',
                    'python tools/fix_orientation.py --apply')
            elif best:
                ok('the right way round - camera is behind the near edge')
        except Exception as e:
            note(f'could not check the orientation ({e})')

    # 3. the detector --------------------------------------------------------
    print('\n3. Detector')
    model = DATA / 'poop_yolo.onnx'
    if model.is_file() and model.stat().st_size > 10_000:
        ok(f'trained model, {model.stat().st_size/1e6:.1f} MB')
    else:
        bad('no trained model - detection falls back to a colour heuristic',
            'python training/deploy_model.py')
    try:
        import re
        import poop_config as c
        # Read scan_lawn's flags out of the FILE rather than importing it:
        # importing pulls in depthai and the whole camera stack, which is slow
        # and can fail for reasons that have nothing to do with these settings.
        src = (_paths.LAWN / 'scan_lawn.py').read_text()
        def flag(name):
            m = re.search(rf'^{name}\s*=\s*(True|False)', src, re.M)
            return m.group(1) if m else '?'
        ok(f'confidence {c.POOP_YOLO_CONF}, {c.POOP_YOLO_TILES}x{c.POOP_YOLO_TILES} '
           f'tiling')
        ok(f'detector off the main loop: {flag("DETECT_ASYNC")}, '
           f'boxes carried forward while panning: {flag("DETECT_CARRY_FORWARD")}')
        if c.POOP_YOLO_CONF > 0.35:
            note(f'confidence {c.POOP_YOLO_CONF} is high for this lawn - '
                 f'0.25 found twice as many on your own frames')
    except Exception as e:
        note(f'could not read the detector settings ({e})')

    # 4. the arm -------------------------------------------------------------
    print('\n4. Arm')
    try:
        import arm_control as a
        ok(f'survey sweep {a.SWEEP_DEG_PER_S} deg/s, plane re-anchors within '
           f'{a.AT_HOME_TOL_DEG} deg of home')
        from trimmer_arm import TrimmerArm
        arm = TrimmerArm()
        pose = arm.read_pose()
        arm.close()
        if pose:
            ok(f'servos answering: {pose}')
            if cal and cal.get('rest_pose'):
                rest = {int(k): int(v) for k, v in cal['rest_pose'].items()}
                off = max(abs(pose.get(k, v) - v) for k, v in rest.items())
                if off > 400:
                    note(f'currently {off} ticks from the rest pose - launch.py '
                         f'will drive it back, give it a moment')
        else:
            bad('the servos did not reply', 'check power and the USB lead')
    except Exception as e:
        bad(f'arm not reachable ({e})', 'check the USB lead, then rerun this')

    # 5. nothing left running ------------------------------------------------
    print('\n5. Ports')
    for port, what in ((8501, 'web app'), (8799, 'video/data endpoint')):
        if port_busy(port):
            note(f'{port} ({what}) is in use - launch.py will clear it')
        else:
            ok(f'{port} ({what}) free')

    # 6. disk ----------------------------------------------------------------
    print('\n6. Session folder')
    try:
        n = len(list(SESSIONS.glob('session_*'))) if SESSIONS.exists() else 0
        ok(f'{n} old session(s) - the scan clears these on start')
    except OSError:
        pass

    # verdict ----------------------------------------------------------------
    print('\n' + '=' * 68)
    if PROBLEMS:
        print(f'  NOT READY - {len(PROBLEMS)} thing(s) to fix:\n')
        for i, (msg, fix) in enumerate(PROBLEMS, 1):
            print(f'   {i}. {msg}')
            print(f'      -> {fix}')
    else:
        print('  GO. Run:  python launch.py')
        if NOTES:
            print('\n  Worth knowing, but nothing that stops the demo:')
            for n_ in NOTES:
                print(f'   - {n_}')
    print('=' * 68 + '\n')
    return 1 if PROBLEMS else 0


if __name__ == '__main__':
    raise SystemExit(main())
