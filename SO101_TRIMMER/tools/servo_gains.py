"""Read - and optionally stiffen - the servos' own position loop.

    python tools/servo_gains.py              # just show them
    python tools/servo_gains.py --p 32       # set the position P gain

WHY
---
Measured on this arm, the shoulder settles up to 66 raw - nearly 6 degrees -
past where it was told to stop, and the elbow up to 36. That is 24 mm of head
height, and it is the reason the cut is not level: the geometry was right and
the arm was not going where it was sent.

Two things cause it, and they are both in the servo, not in the path:

    SAG.  A position servo holds by generating torque proportional to its
    error. Under gravity it therefore MUST sit at an error - that is how it
    makes the holding torque. A low P gain means a large error for the same
    torque, and the error grows with the load, which is why it is worse with
    the arm stretched out.

    STICTION.  Below some torque nothing moves at all. The measured error was
    +48 raw when the joint had just been lifting and +14 when it had just been
    dropping - it depends on which way it approached, which is friction, and
    which is why it cannot be corrected by any model of where the arm is.

Raising P attacks both: more torque per unit of error, so it settles closer,
and it is more likely to break through stiction.

CAUTION
-------
Too much P makes the joint buzz or oscillate around its target, which is worse
than sitting still in the wrong place. Raise it in steps and measure after each
one with trimmer/check_level.py rather than jumping to a big number. The
original values are printed before anything is written, so they can be put
back.
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
import time

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError, REG_LOCK

# Feetech STS position-loop registers.
REG_P, REG_D, REG_I = 21, 22, 23
NAMES = {tc.BASE_ID: 'base', tc.SHOULDER_ID: 'shoulder', tc.ELBOW_ID: 'elbow',
         tc.WRIST_ID: 'wrist', tc.HEAD_ID: 'HEAD'}
# The joints that carry the arm's weight. The base turns about a vertical axis
# and holds nothing up, so stiffening it buys nothing.
LOADED = (tc.SHOULDER_ID, tc.ELBOW_ID)
MAX_P = 100


def read1(arm, sid, reg):
    try:
        v = arm._read(sid, reg, 1)
        return None if v is None else int(v[0])
    except TrimmerError:
        return None


def main():
    want = None
    if '--p' in sys.argv:
        try:
            want = int(sys.argv[sys.argv.index('--p') + 1])
        except Exception:
            print('usage: --p <number>')
            return 1
        if not 1 <= want <= MAX_P:
            print(f'refusing: keep P between 1 and {MAX_P}')
            return 1
    try:
        arm = TrimmerArm()
    except TrimmerError as e:
        print(f'could not talk to the arm: {e}')
        return 1

    print('\n' + '=' * 58)
    print(f'{"joint":>10} {"P":>5} {"I":>5} {"D":>5}')
    print('=' * 58)
    before = {}
    for sid in arm.ids:
        p, i, d = (read1(arm, sid, REG_P), read1(arm, sid, REG_I),
                   read1(arm, sid, REG_D))
        before[sid] = p
        mark = '   <- carries the arm' if sid in LOADED else ''
        print(f'{NAMES.get(sid, sid):>10} {str(p):>5} {str(i):>5} '
              f'{str(d):>5}{mark}')
    print('=' * 58)

    if want is None:
        print('\nTo stiffen the two joints that hold the arm up:')
        print('    python tools/servo_gains.py --p 32')
        print('then measure with:  python trimmer/check_level.py')
        print('\nRaise it in steps. If a joint starts to buzz or hunt around')
        print('its target, that is too far - come back down.')
        arm.close()
        return 0

    print(f'\nsetting P = {want} on {", ".join(NAMES[s] for s in LOADED)}')
    print(f'(was {", ".join(f"{NAMES[s]} {before.get(s)}" for s in LOADED)})')
    try:
        if not input('go ahead? [y/N]: ').strip().lower().startswith('y'):
            print('nothing changed')
            arm.close()
            return 0
    except EOFError:
        arm.close()
        return 0

    for sid in LOADED:
        if sid not in arm.ids:
            continue
        try:
            arm._write(sid, REG_LOCK, [0])
            time.sleep(0.02)
            arm._write(sid, REG_P, [want])
            time.sleep(0.03)
            arm._write(sid, REG_LOCK, [1])
            time.sleep(0.05)
        except TrimmerError as e:
            print(f'  {NAMES[sid]}: write failed ({e})')
    print()
    for sid in LOADED:
        if sid in arm.ids:
            print(f'  {NAMES[sid]:>10} P now {read1(arm, sid, REG_P)}')
    print('\nNow measure it:  python trimmer/check_level.py')
    print(f'To undo:  python tools/servo_gains.py --p '
          f'{before.get(tc.SHOULDER_ID, 16)}')
    arm.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
