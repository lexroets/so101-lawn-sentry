"""Teach the cutting plane by touching it - with the head held as it will cut.

    python trimmer/level_plane.py

WHAT WENT WRONG THE FIRST TIME, AND WHY IT MATTERS
--------------------------------------------------
The first version let you place the arm entirely by hand and recorded the
shoulder and elbow. The fit looked excellent - the touches sat 1.4 degrees off
the curve - and the result was worse than no correction at all: the pass
started high and dug into the table as it extended.

The fit was not the problem. The model behind it was.

The head's height above the table is

    L1*cos(shoulder) + L2*cos(shoulder+elbow) + L3*cos(shoulder+elbow+wrist)

Three terms, and the third one is the trimmer head itself sticking out past the
wrist. Holding the WRIST fixed makes that third term constant, and only then is
constant height a curve through (shoulder, elbow) alone - which is what the
correction assumed.

But during that calibration the wrist was free. Every touch had a different
wrist angle, so every touch was measuring a DIFFERENT curve, and the fit ran
through a family of them. Meanwhile the pass is played with the wrist driven to
a fixed pitch. The calibration and the playback were describing two different
arms, and the mismatch grew with reach - which is exactly the "starts high,
digs in as it extends" you saw.

THE FIX
-------
Calibrate in the geometry you will cut in. The wrist is held under torque here
and driven continuously to the same locked pitch the pass uses, so as you move
the arm the head keeps the attitude it will actually cut at. Only the base,
shoulder and elbow are free for you to place by hand.

Then every touch lies on one curve, because the third term really is constant.

The raw touches are saved alongside the fit, so a bad calibration can be
inspected afterwards rather than guessed at.

DRAGGING IT INSTEAD OF TOUCHING IT
----------------------------------
Pressing SPACE at eight spots describes the plane with eight numbers. Dragging
the head along the surface describes it with hundreds, spread continuously
across the whole reach rather than clustered wherever a hand happened to stop.
The fit has two unknowns; more evidence over a wider span pins them harder,
and gaps between touch points are where a fitted curve is free to wander.

The safety is unchanged and matters more here, not less. The wrist is still
held at the cutting pitch throughout, and any sample where the head is pushed
off that angle is DISCARDED rather than recorded - which during a drag is the
likely failure, because leaning on the surface is exactly what pushes the head
off its angle. So press lightly: the head should rest on the surface, not
against it.

Keys
    SPACE   record one point - the head is resting ON the surface right now
    g       DRAG record on/off - hold the head on the surface and sweep it
            in and out; it samples continuously as the arm moves
    u       undo the last one
    f       fit, check it, and save
    q       quit
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
import time

import cv2
import numpy as np

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError
from keys import Keys

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'teach the cutting plane - the wrist is held, you move the rest'
OUT = 'height_plane.py'
MIN_SPAN_RAW = 300
MIN_POINTS = 4
# How far the wrist may sit from its commanded pitch before a touch is treated
# as untrustworthy. If the wrist is being pushed off its target - by the head
# resting hard on the surface, say - the third term is not constant after all
# and that touch belongs to a different curve.
PITCH_TOL_RAW = 40
# While dragging, take a sample once the arm has moved this far. Sampling on
# MOVEMENT rather than on a timer means pausing to reposition your hands does
# not fill the record with a hundred copies of one pose.
DRAG_STEP_RAW = 12
MAX_STORED = 120        # thin before writing; the fit uses everything


def pitch_target():
    """The head pitch the pass will be played at - see pathplay.level_pitch."""
    from box_paths import BOX_PATHS
    for name in getattr(tc, 'LEVEL_PITCH_PATHS', ('top',)):
        wps = BOX_PATHS.get(name) or []
        if wps:
            s = sorted(w[tc.SHOULDER_ID] + w[tc.ELBOW_ID] + w[tc.WRIST_ID]
                       for w in wps)
            fixed = getattr(tc, 'LEVEL_PITCH_TARGET', {}).get(name)
            return (fixed if fixed is not None else s[len(s) // 2]) + \
                getattr(tc, 'LEVEL_PITCH_TRIM', 0)
    return None


def fit(points, target, log):
    """Fit the constant-height curve, and say whether it can be believed."""
    if len(points) < MIN_POINTS:
        log(f'need at least {MIN_POINTS} touches, have {len(points)}')
        return None
    sh = np.array([p[tc.SHOULDER_ID] for p in points], float)
    el = np.array([p[tc.ELBOW_ID] for p in points], float)
    wr = np.array([p[tc.WRIST_ID] for p in points], float)
    span = sh.max() - sh.min()
    if span < MIN_SPAN_RAW:
        log(f'the touches only span {span:.0f} raw of shoulder - move the arm '
            f'further in and out between them (need {MIN_SPAN_RAW})')
        return None
    # The check the first version did not do: was the head actually at the
    # pitch it will cut at, for every touch?
    off = np.abs(sh + el + wr - target)
    if off.max() > PITCH_TOL_RAW:
        log(f'REFUSED: on {int((off > PITCH_TOL_RAW).sum())} touch(es) the '
            f'wrist was {off.max():.0f} raw off the cutting pitch. The head was '
            f'being pushed off its angle - press less hard and redo those.')
        return None
    deg = 2 if len(points) >= 5 else 1
    coef = np.polyfit(sh, el, deg)
    resid = el - np.polyval(coef, sh)
    log(f'fitted degree {deg} over shoulder {sh.min():.0f}..{sh.max():.0f} '
        f'from {len(points)} points; worst {np.abs(resid).max():.0f} raw '
        f'off the curve')
    if np.abs(resid).max() > 120:
        log('POOR FIT - a point was probably not on the surface.')
    # A residual in raw ticks is hard to judge. The arm model turns the same
    # points into millimetres of height, which is the thing being calibrated.
    try:
        import pathplay
        pathplay._ARM_MODEL = None
        m = pathplay.arm_model(points)
        if m is not None:
            h = pathplay.height_mm(m, sh, el)
            log(f'   in millimetres: the surface reads {float(h.max()-h.min()):.1f} '
                f'mm thick (rms {m["RMS_MM"]:.1f} mm) - that is the accuracy '
                f'the cut can inherit')
    except Exception:
        pass
    return coef, float(sh.min()), float(sh.max()), \
        float(np.abs(resid).max()), points


def save(res, target, log):
    from paths import CORE
    coef, lo, hi, err, points = res
    p = CORE / OUT
    with p.open('w') as f:
        f.write('# Written by trimmer/level_plane.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# The constant-height curve: elbow = polyval(COEF, shoulder).\n')
        f.write('# ONLY VALID AT THE PITCH IT WAS TAUGHT AT. The head sticks out\n')
        f.write('# past the wrist, so its contribution to height is only constant\n')
        f.write('# while the pitch is. PITCH_TARGET records which one.\n\n')
        f.write('COEF = [' + ', '.join(repr(float(c)) for c in coef) + ']\n')
        f.write(f'SHOULDER_RANGE = ({float(lo)!r}, {float(hi)!r})\n')
        f.write(f'WORST_RESIDUAL_RAW = {float(err)!r}\n')
        f.write(f'PITCH_TARGET = {int(target)}\n\n')
        f.write('# the raw touches, so a bad calibration can be looked at\n')
        f.write('TOUCHES = [\n')
        # thinned for the file - evenly across the reach, so the record still
        # spans what was measured rather than whichever end had most samples
        keep = points
        if len(points) > MAX_STORED:
            order = sorted(range(len(points)),
                           key=lambda i: points[i][tc.SHOULDER_ID])
            pick = np.linspace(0, len(order) - 1, MAX_STORED).astype(int)
            keep = [points[order[i]] for i in sorted(set(pick))]
        for q in keep:
            f.write('    {' + ', '.join(f'{k}: {int(v)}'
                                        for k, v in sorted(q.items())) + '},\n')
        f.write(']\n')
    # Drop the compiled cache. Python decides a .pyc is current from the
    # source's (mtime, size); rewrite this file within the same second at the
    # same length - easily done, the numbers barely change - and a fresh
    # process will load the OLD calibration and report success.
    for c in (CORE / '__pycache__').glob('height_plane.*.pyc'):
        try:
            c.unlink()
        except OSError:
            pass
    log(f'saved -> core/{OUT}')
    log('now set LEVEL_HEIGHT_PATHS = (\'top\',) in core/trimmer_config.py, '
        'then check it with trimmer/show_plan.py BEFORE running the arm.')


def main():
    target = None
    try:
        target = pitch_target()
    except Exception:
        pass
    if target is None:
        print('no taught pass to take the cutting pitch from - teach the top '
              'pass first')
        return
    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return

    free = [tc.BASE_ID, tc.SHOULDER_ID, tc.ELBOW_ID]
    held = [tc.WRIST_ID] + ([tc.HEAD_ID] if tc.HEAD_ID in arm.ids else [])
    points, msg = [], ''
    drag = False
    last_drag = None
    kb = Keys()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def log(t):
        nonlocal msg
        msg = t
        print(f'  {t}')

    try:
        arm.enable()
        arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
        arm.release(free)
    except TrimmerError as e:
        print(f'could not set the torque: {e}')
        arm.close()
        return

    print(f'\nThe WRIST is held at the cutting pitch ({target}) and follows you.')
    print('The base, shoulder and elbow are free - SUPPORT THE ARM.')
    print('Rest the head on the surface, press SPACE, move to a different')
    print('reach, repeat. Spread them from folded right in to stretched out.\n')
    try:
        while True:
            kb.pump()
            try:
                pose = arm.read_pose()
            except TrimmerError:
                pose = {}
            kb.pump()

            # keep the head at the cutting pitch as the arm is moved by hand
            if pose and tc.SHOULDER_ID in pose and tc.ELBOW_ID in pose:
                want = target - pose[tc.SHOULDER_ID] - pose[tc.ELBOW_ID]
                if 0 <= want <= 4095:
                    try:
                        arm.move_to(tc.WRIST_ID, int(want))
                    except TrimmerError:
                        pass
            kb.pump()

            disp = np.zeros((320, 940, 3), np.uint8)
            sh = [p[tc.SHOULDER_ID] for p in points]
            span = (max(sh) - min(sh)) if len(sh) > 1 else 0
            ok_span = span >= MIN_SPAN_RAW
            rows = [
                (f'{len(points)} touch point(s)   shoulder span {span} raw'
                 f'   {"OK" if ok_span else f"need {MIN_SPAN_RAW}+"}',
                 (90, 255, 90) if ok_span and len(points) >= MIN_POINTS
                 else (0, 255, 255)),
                (f'wrist HELD at the cutting pitch {target} - base, shoulder '
                 f'and elbow are free', (60, 200, 255)),
            ]
            if pose:
                p_now = (pose.get(tc.SHOULDER_ID, 0) + pose.get(tc.ELBOW_ID, 0)
                         + pose.get(tc.WRIST_ID, 0))
                d = p_now - target
                rows.append((f'now  shoulder {pose.get(tc.SHOULDER_ID,"-")}   '
                             f'elbow {pose.get(tc.ELBOW_ID,"-")}   '
                             f'wrist {pose.get(tc.WRIST_ID,"-")}   '
                             f'pitch {d:+d} off',
                             (255, 220, 90) if abs(d) <= PITCH_TOL_RAW
                             else (80, 140, 255)))
                if abs(d) > PITCH_TOL_RAW:
                    rows.append(('the head is being pushed off its angle - '
                                 'ease the pressure before recording',
                                 (80, 140, 255)))
            for i, q in enumerate(points[-5:]):
                rows.append((f'   #{len(points)-len(points[-5:])+i+1}  '
                             f'shoulder {q[tc.SHOULDER_ID]}  '
                             f'elbow {q[tc.ELBOW_ID]}', (170, 170, 170)))
            if drag:
                rows.append((f'DRAGGING - {len(points)} points   keep the head '
                             f'ON the surface, press lightly', (60, 60, 255)))
            rows.append(('SPACE point   g DRAG   u undo   f fit and save   '
                         'q quit', (200, 200, 200)))
            if msg:
                rows.append((msg, (255, 220, 90)))
            for i, (t_, c_) in enumerate(rows):
                cv2.putText(disp, t_, (12, 26 + i * 26), FONT, 0.5, c_, 1,
                            cv2.LINE_AA)
            cv2.imshow(WIN, disp)

            kb.pump(1)
            if kb.hit(ord('q')):
                break
            # --- continuous drag capture ---------------------------------
            if drag and pose and all(j in pose for j in
                                     (tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID)):
                d = abs(pose[tc.SHOULDER_ID] + pose[tc.ELBOW_ID]
                        + pose[tc.WRIST_ID] - target)
                moved = (DRAG_STEP_RAW if last_drag is None else
                         max(abs(pose[j] - last_drag[j])
                             for j in (tc.SHOULDER_ID, tc.ELBOW_ID)))
                if d > PITCH_TOL_RAW:
                    log(f'not recording - pressing too hard, the head is {d} '
                        f'raw off its angle')
                elif moved >= DRAG_STEP_RAW:
                    points.append(dict(pose))
                    last_drag = dict(pose)
            if kb.hit(ord('g')):
                drag = not drag
                last_drag = None
                log('DRAGGING - rest the head on the surface and sweep it in '
                    'and out. g again to stop.' if drag
                    else f'stopped - {len(points)} points')
            if kb.hit(32) and pose:
                d = abs(pose[tc.SHOULDER_ID] + pose[tc.ELBOW_ID]
                        + pose[tc.WRIST_ID] - target)
                if d > PITCH_TOL_RAW:
                    log(f'not recorded - the wrist is {d} raw off the cutting '
                        f'pitch. Ease the pressure and try again.')
                else:
                    points.append(dict(pose))
                    log(f'recorded #{len(points)} at shoulder '
                        f'{pose[tc.SHOULDER_ID]}')
            if kb.hit(ord('u')) and points:
                points.pop()
                log(f'undid one - {len(points)} left')
            if kb.hit(ord('f')):
                res = fit(points, target, log)
                if res:
                    save(res, target, log)
    finally:
        try:
            arm.torque_off()
        except Exception:
            pass
        arm.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
