"""Measure where the head ACTUALLY goes, against where it was told to go.

    python trimmer/check_level.py              # one cycle
    python trimmer/check_level.py --cycles 5   # repeat until it stops improving

WHY THIS EXISTS
---------------
Every attempt at levelling the top pass so far has been a model of the arm's
geometry, and every one of them has predicted a level cut that did not happen.
The last one predicts 0.2 mm of variation. The arm does not agree.

At that point the model is not the thing to keep improving. There are two
things it cannot express, and neither has ever been measured on this rig:

    THE ARM MAY NOT BE SQUARE TO THE SURFACE.  The height model assumes the
    base rotates about a vertical axis, so swinging the base moves the head
    sideways and not up or down. If the arm is mounted even slightly tilted -
    or the table is not level relative to it - that is false, and the error
    changes across the sweep. The pass swings the base from 1437 to 1990,
    much wider than the touches used for calibration covered.

    THE ARM SAGS.  A commanded position is a request. These servos hold
    against gravity with a finite stiffness, so the arm settles low, and it
    settles lower the further it is stretched out - which looks exactly like
    a cut that is level in software and sloped in reality. The calibration
    cannot see this: the touches were recorded with the arm supported by
    hand, resting on the surface, so they record where it really was. Play
    the same numbers back under motor torque and the arm arrives somewhere
    else.

So this stops guessing and reads the encoders. It steps through the pass,
lets each pose settle, and reports the difference between the commanded
height and the achieved one, in millimetres.

The arm moves. Nothing is written.

Keys
    q       stop
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
import sys
import importlib
import time

import numpy as np

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError
import pathplay
import selflearn
from pathplay import goto, arm_model, height_mm

SETTLE_S = 1.2          # long enough for the servo to stop creeping
NAME = 'top'


GAIN = 0.7          # how much of each cycle's leftover error to take on
MAX_CORRECTION = 400


def save_table(errs):
    """Add this cycle's leftover error to the correction already in force.

    ONE MEASUREMENT IS NOT ENOUGH, and the reason is worth stating plainly:
    the correction changes the command, and the error the arm makes depends
    on the command. Measure once, subtract it, and the arm now makes a
    DIFFERENT, smaller error. Subtract that too and it shrinks again.

    So each cycle adds what is left over rather than replacing what came
    before, which is how a machine is taught to repeat a path accurately -
    run it, see where it actually went, fold the difference back in, run it
    again. A few passes and there is nothing left to fold in.

    The gain is below one on purpose. Taking the whole leftover error every
    time assumes the arm responds exactly as expected and overshoots when it
    does not; 0.7 converges nearly as fast and does not ring.
    """
    from paths import CORE
    old = []
    try:
        from sag_table import SAG
        old = list(SAG)
    except Exception:
        pass

    def previous(at):
        best, dist = None, None
        for row in old:
            d = sum(abs(int(at[j]) - int(row['at'][j]))
                    for j in (tc.SHOULDER_ID, tc.ELBOW_ID)
                    if j in row['at'] and j in at)
            if dist is None or d < dist:
                best, dist = row, d
        return best['err'] if best is not None and dist is not None and dist < 120 else {}

    merged, growth = [], []
    for e in errs:
        at = e.pop('at')
        was = previous(at)
        now = {}
        for j, v in e.items():
            acc = int(round(was.get(j, 0) + GAIN * v))
            now[j] = max(-MAX_CORRECTION, min(MAX_CORRECTION, acc))
        growth.append(max((abs(now.get(j, 0) - was.get(j, 0)) for j in now),
                          default=0))
        merged.append({'at': at, 'err': now})

    p = CORE / 'sag_table.py'
    with p.open('w') as f:
        f.write('# Written by trimmer/check_level.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# How far each joint OVERSHOT its commanded position, measured.\n')
        f.write('# Subtracted from the command so the arm lands where it is\n')
        f.write('# meant to. Re-measure after changing the path, the payload,\n')
        f.write('# the speed or the servo gains - it describes THIS setup.\n\n')
        f.write('SAG = [\n')
        for m in merged:
            f.write('    {'
                    + f"'at': {{{', '.join(f'{k}: {v}' for k, v in sorted(m['at'].items()))}}}, "
                    + f"'err': {{{', '.join(f'{k}: {v}' for k, v in sorted(m['err'].items()))}}}"
                    + '},\n')
        f.write(']\n')
    for c in (CORE / '__pycache__').glob('sag_table.*.pyc'):
        try:
            c.unlink()
        except OSError:
            pass
    print(f'\ncorrection updated -> core/sag_table.py ({len(merged)} poses, '
          f'changed by up to {max(growth) if growth else 0} raw this cycle)')
    if not getattr(tc, 'SAG_COMPENSATE', False):
        print('SAG_COMPENSATE is still False in core/trimmer_config.py - set it '
              'to True or none of this is used.')
    else:
        print('run this again: each cycle folds in what is left, and the '
              'leftover should shrink each time.')


def main():
    try:
        from box_paths import BOX_PATHS
        from height_plane import TOUCHES
    except Exception as e:
        print(f'need a taught pass and a taught plane first ({e})')
        return
    wps = BOX_PATHS.get(NAME) or []
    if len(wps) < 4:
        print(f'no {NAME} pass taught')
        return

    model = arm_model(TOUCHES)
    if model is None:
        print('could not build the arm model - needs at least 3 touch points '
              'in core/height_plane.py from trimmer/level_plane.py')
        return

    # exactly what the player would send, and what it is aiming for
    targets, cmd = pathplay.prepare(wps, NAME, log=lambda m: print(f'  {m}'))

    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
        pathplay.set_rate(tc.MOVE_SPEED)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return

    cycles = 1
    if '--cycles' in sys.argv:
        try:
            cycles = max(1, int(sys.argv[sys.argv.index('--cycles') + 1]))
        except Exception:
            cycles = 1
    history = []
    for cycle in range(cycles):
        if cycle:
            # pick up the correction just written, then rebuild the commands
            sys.modules.pop('sag_table', None)
            importlib.invalidate_caches()
            print(f'\n{"=" * 66}\ncycle {cycle + 1} of {cycles}\n{"=" * 66}')
            targets, cmd = pathplay.prepare(wps, NAME, log=lambda m: print(f'  {m}'))
        worst = run_cycle(arm, model, targets, cmd, history)
        if worst is None:
            break
        if len(history) > 1 and history[-1] > history[-2] * 0.9:
            print('\nno longer improving - stopping here')
            break
    try:
        if tc.REST_POSE:
            goto(arm, tc.REST_POSE)
        arm.torque_off()
    except Exception:
        pass
    arm.close()
    if len(history) > 1:
        print('\n' + '=' * 66)
        print('worst height error, cycle by cycle:')
        for i, h in enumerate(history):
            print(f'   cycle {i + 1}: {h:5.1f} mm' + ('   <- start' if not i else ''))
        print('=' * 66)
    return


def run_cycle(arm, model, targets, cmd, history):
    print(f'\nstepping through {len(cmd)} poses, settling {SETTLE_S}s on each.')
    print('Ctrl-C to stop.\n')
    print(f'{"wp":>3} {"base":>6} {"shoulder cmd/act":>18} '
          f'{"elbow cmd/act":>15} {"height error":>13}')
    rows, errs = [], []
    try:
        for i, w in enumerate(cmd):
            goto(arm, w, log=lambda m: None)
            time.sleep(SETTLE_S)
            try:
                act = arm.read_pose()
            except TrimmerError:
                continue
            if tc.SHOULDER_ID not in act or tc.ELBOW_ID not in act:
                continue
            t = targets[i]
            hc = float(height_mm(model, t[tc.SHOULDER_ID], t[tc.ELBOW_ID]))
            ha = float(height_mm(model, act[tc.SHOULDER_ID],
                                 act[tc.ELBOW_ID]))
            rows.append((w[tc.BASE_ID], w[tc.SHOULDER_ID], hc, ha))
            # measured against the TARGET, not the command: what is left over
            # after the correction already in force
            errs.append({j: int(act[j]) - int(t[j])
                         for j in (tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID)
                         if j in act and j in t} | {'at': dict(t)})
            print(f'{i+1:3d} {w[tc.BASE_ID]:6d} '
                  f'{t[tc.SHOULDER_ID]:8d}/{act[tc.SHOULDER_ID]:<8d} '
                  f'{t[tc.ELBOW_ID]:7d}/{act[tc.ELBOW_ID]:<7d} '
                  f'{ha - hc:+10.1f} mm')
    except KeyboardInterrupt:
        print('\nstopped')
        return None

    if len(rows) < 4:
        print('\nnot enough measurements')
        return None
    save_table(errs)
    base = np.array([r[0] for r in rows], float)
    sh = np.array([r[1] for r in rows], float)
    err = np.array([r[3] - r[2] for r in rows], float)
    ach = np.array([r[3] for r in rows], float)

    print('\n' + '=' * 66)
    print(f'ACHIEVED height varied {ach.max() - ach.min():.1f} mm  '
          f'(the model predicted it would be level)')
    print(f'commanded vs achieved: {err.mean():+.1f} mm on average, '
          f'{err.min():+.1f} to {err.max():+.1f}')
    print('=' * 66)
    print('\nwhere does the error come from?\n')
    for label, x in (('REACH   (sag: worse when stretched out)', sh),
                     ('BASE    (arm not square to the surface)', base)):
        if x.max() - x.min() < 30:
            print(f'  {label}: not enough spread to tell')
            continue
        c = float(np.corrcoef(x, err)[0, 1])
        k = float(np.polyfit(x, err, 1)[0]) * (x.max() - x.min())
        print(f'  {label}')
        print(f'      correlation {c:+.2f}, {k:+.1f} mm across its range')
    print('\n  REACH correlation means SAG - a position servo holds by making')
    print('  torque from its own error, so under load it must sit off target,')
    print('  and further off the more it is stretched out.')
    print('  BASE correlation means the arm is not square to the surface.')
    history.append(float(np.abs(err).max()))

    # Direction matters as much as position, and it is worth saying so: a
    # measured average cannot be fed forward if the error depends on which way
    # the joint just came from.
    if len(rows) > 6:
        d = np.diff(sh, prepend=sh[0])
        up, dn = d < -20, d > 20
        if up.sum() > 1 and dn.sum() > 1:
            print(f'\n  approaching from below: {err[dn].mean():+.1f} mm')
            print(f'  approaching from above: {err[up].mean():+.1f} mm')
            if abs(err[up].mean() - err[dn].mean()) > 2:
                print('  -> the error depends on the DIRECTION of approach, '
                      'which is')
                print('     stiction. Feeding a fixed correction forward '
                      'cannot fix that;')
                print('     stiffening the servo can, and so can measuring it '
                      'and taking it off the command, which is what the '
                      'correction above does.')
    return float(np.abs(err).max())


if __name__ == '__main__':
    main()
