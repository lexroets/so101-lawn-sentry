"""Move the head servo's zero, so its usable arc stops straddling it.

    python tools/set_head_zero.py            # report where things sit
    python tools/set_head_zero.py --centre   # put the CURRENT angle at 2048

THE PROBLEM THIS SOLVES
-----------------------
The trimmer head's usable arc runs 1000 down through 0, wraps to 4095, and
carries on down to 3000. Flat sits at 0 - exactly on the wrap.

That is a mounting accident, and it makes half the arc unreachable. A Feetech
servo in position mode drives to an ABSOLUTE encoder value; it does not take
the short way round. Commanding 4095 then 0 does not step one tick, it sweeps
backwards through 2048 - through the forbidden middle, winding the cable.

In practice the crossing IS made to work: pathplay steps the head across the
wrap and reads back after every step, so a servo that starts going the long
way is caught within about 50 ms and stopped. That is a guard, though, not a
cure - it turns a cable-wrecking sweep into an aborted path.

THE FIX
-------
Shift the servo's zero so the whole usable arc lands in one continuous block
around the middle. Put flat at 2048 and the arc becomes roughly 1048..3048:
no wrap anywhere in it, every angle reachable from every other, and the
forbidden region moves out to the far side where it belongs.

This writes the servo's position-correction register, which is EEPROM and
persists. Nothing moves while it happens - torque is off throughout.

AFTERWARDS
----------
Nothing needs re-teaching. Every stored head angle is wrong by exactly the
same offset, which is a shift and not a loss of information, so this rewrites
them: the arc limits in trimmer_config.py and the head value of every waypoint
in box_paths.py all move by the same amount. The paths describe the same
physical motion afterwards as before.
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

# Feetech STS position correction: signed, 12 bits plus a sign bit in b11.
REG_POS_CORRECTION = 31
CENTRE = 2048


def encode_offset(v):
    v = int(v)
    if v < 0:
        return [(-v) & 0xFF, (((-v) >> 8) & 0x07) | 0x08]
    return [v & 0xFF, (v >> 8) & 0x07]


def migrate(shift):
    """Move the config limits and every taught head angle by the same shift.

    The servo now reports a different number for the same physical angle, so
    anything stored in the old numbering is stale. All of it is stale by the
    SAME amount, though, which makes this a rename rather than a re-teach.
    """
    import re
    from paths import CORE

    def moved(v):
        return (int(v) + shift) % 4096

    cfg = CORE / 'trimmer_config.py'
    txt = cfg.read_text()
    lo, hi = moved(tc.HEAD_ARC_LO), moved(tc.HEAD_ARC_HI)
    for name, val in (('HEAD_ARC_LO', lo), ('HEAD_ARC_HI', hi),
                      ('HEAD_FLAT_RAW', CENTRE)):
        txt = re.sub(rf'^{name} = -?\d+', f'{name} = {val}', txt,
                     count=1, flags=re.M)
    cfg.write_text(txt)
    print(f'\ntrimmer_config.py: clockwise limit {tc.HEAD_ARC_LO} -> {lo}, '
          f'anticlockwise {tc.HEAD_ARC_HI} -> {hi}, flat -> {CENTRE}')
    if lo < hi:
        print(f'  the arc is now {lo}..{hi} - one block, no wrap in it.')

    paths_file = CORE / 'box_paths.py'
    if paths_file.is_file():
        txt = paths_file.read_text()
        n = [0]

        def bump(m):
            n[0] += 1
            return f'{tc.HEAD_ID}: {moved(m.group(1))}'

        txt = re.sub(rf'\b{tc.HEAD_ID}: (\d+)', bump, txt)
        paths_file.write_text(txt)
        print(f'box_paths.py: {n[0]} taught head angles shifted by {shift:+d} '
              f'- the paths are unchanged, they just read differently.')

    off = CORE / 'box_offsets.py'
    if off.is_file():
        off.write_text('# cleared by tools/set_head_zero.py - the head zero '
                       'moved, so old offsets no longer mean anything\n'
                       'BOX_OFFSETS = {}\n')
        print('box_offsets.py: cleared.')
    print('\nNothing to re-teach. Check it before trusting it:')
    print('    python trimmer/check_paths.py')


def main():
    centre = '--centre' in sys.argv
    try:
        arm = TrimmerArm()
    except TrimmerError as e:
        print(f'could not talk to the arm: {e}')
        return 1
    hid = tc.HEAD_ID
    if hid not in arm.ids:
        print(f'the head joint (id {hid}) is not answering')
        arm.close()
        return 1
    try:
        arm.torque_off()
    except TrimmerError:
        pass

    here = arm.read_position(hid)
    print('\n' + '=' * 64)
    print(f'head joint (id {hid}) reads {here}')
    print(f'usable arc is {tc.HEAD_ARC_HI} (ACW limit) -> 0/4095 (flat) -> '
          f'{tc.HEAD_ARC_LO} (CW limit)')
    straddles = tc.HEAD_ARC_LO > tc.HEAD_ARC_HI
    print(f'straddles the servo zero: {"YES - half the arc is unreachable" if straddles else "no"}')
    print('=' * 64)

    # --centre is still honoured, but it is no longer the only way in. Run
    # from an editor's Run button and no arguments reach the script at all,
    # so a flag-only entry point looks lyike a tool that does nothing.
    if not centre and straddles:
        print('\nThe arc straddles the zero, which is what makes the head sweep')
        print('the long way. This can move the servo zero so it no longer does.')
        print('\nHOLD THE HEAD FLAT - whatever angle it is at right now becomes')
        print('the new middle of its range.')
        try:
            centre = input('\nfix it now? [y/N]: ').strip().lower().startswith('y')
        except EOFError:
            centre = False
        if not centre:
            print('nothing changed')
            arm.close()
            return 0
    elif not centre:
        print('\nThe arc does not straddle the zero - nothing to fix.')
        arm.close()
        return 0

    shift = CENTRE - int(here)
    print(f'\nThis will make the CURRENT position read {CENTRE} instead of '
          f'{here}  (shift {shift:+d}).')
    print('Hold the head FLAT before continuing - whatever it is at now')
    print('becomes the new middle of the range.')
    print('EEPROM write: it persists across power cycles.')
    try:
        if not input('go ahead? [y/N]: ').strip().lower().startswith('y'):
            print('nothing changed')
            arm.close()
            return 0
    except EOFError:
        arm.close()
        return 0

    try:
        arm._write(hid, REG_LOCK, [0])
        time.sleep(0.02)
        arm._write(hid, REG_POS_CORRECTION, encode_offset(shift))
        time.sleep(0.05)
        arm._write(hid, REG_LOCK, [1])
        time.sleep(0.1)
    except TrimmerError as e:
        print(f'the write failed: {e}')
        arm.close()
        return 1

    now = arm.read_position(hid)
    print(f'\nhead now reads {now}   (wanted about {CENTRE})')
    if abs(int(now) - CENTRE) < 60:
        migrate(shift)
    else:
        print('\nThat did not take. Not every firmware exposes this register at')
        print('address 31, and some refuse it with torque enabled. The')
        print('alternative is mechanical: unbolt the head, rotate it a quarter')
        print('turn on its spline, and refit - same effect, no EEPROM.')
    arm.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
