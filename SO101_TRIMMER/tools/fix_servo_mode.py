"""Put a servo back into position mode when it will not stop rotating.

    python tools/fix_servo_mode.py            # report every joint's mode
    python tools/fix_servo_mode.py 5          # fix joint 5
    python tools/fix_servo_mode.py 5 --wheel  # deliberately make it a wheel

A Feetech servo has an operating mode, and it changes what the speed register
means:

    POSITION   goal speed is a LIMIT - "get to this angle, no faster than this"
    WHEEL      goal speed is a COMMAND - "turn at this rate, forever"

So a joint that has ended up in wheel mode does not hold an angle at all. It
starts spinning the instant anything sets a motion profile, ignores every
position sent to it, and looks for all the world like a broken servo or a
software bug. It is neither - it is one byte in EEPROM.

That byte is why this is a separate tool rather than something done at
startup: EEPROM survives power cycles, so it is written once, deliberately,
rather than silently on every run.

The arm is not moved. Torque is left off throughout.
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

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError, MODE_POSITION, MODE_WHEEL

NAMES = {tc.BASE_ID: 'base', tc.SHOULDER_ID: 'shoulder', tc.ELBOW_ID: 'elbow',
         tc.WRIST_ID: 'wrist', tc.HEAD_ID: 'HEAD (trimmer head mount)'}


def describe(m):
    if m is None:
        return 'unreadable'
    return {MODE_POSITION: 'position  (correct for a joint)',
            MODE_WHEEL: 'WHEEL - will rotate continuously'}.get(m, f'mode {m}')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    want_wheel = '--wheel' in sys.argv

    try:
        arm = TrimmerArm()
    except TrimmerError as e:
        print(f'could not talk to the arm: {e}')
        return 1
    try:
        arm.torque_off()          # nothing should move during this
    except TrimmerError:
        pass

    print('\n' + '=' * 60)
    print('SERVO OPERATING MODES')
    print('=' * 60)
    modes = {}
    for sid in arm.ids:
        modes[sid] = arm.read_mode(sid)
        print(f'  id {sid}  {NAMES.get(sid, "?"):28s} {describe(modes[sid])}')
    print('=' * 60)

    if not args:
        bad = [s for s, m in modes.items() if m == MODE_WHEEL]
        if bad:
            print(f'\n{len(bad)} joint(s) in wheel mode. To fix the first:')
            print(f'    python tools/fix_servo_mode.py {bad[0]}')
        else:
            print('\nAll joints are in position mode - nothing to fix.')
        arm.close()
        return 0

    try:
        sid = int(args[0])
    except ValueError:
        print(f'"{args[0]}" is not a servo id')
        arm.close()
        return 1
    if sid not in arm.ids:
        print(f'servo {sid} is not answering; ids present are {list(arm.ids)}')
        arm.close()
        return 1

    target = MODE_WHEEL if want_wheel else MODE_POSITION
    print(f'\nSetting id {sid} ({NAMES.get(sid, "?")}) to '
          f'{"WHEEL" if want_wheel else "POSITION"} mode.')
    print('This writes EEPROM and persists across power cycles.')
    try:
        if not input('go ahead? [y/N]: ').strip().lower().startswith('y'):
            print('nothing changed')
            arm.close()
            return 0
    except EOFError:
        arm.close()
        return 0

    got = arm.set_mode(sid, target)
    print(f'\nid {sid} now reports: {describe(got)}')
    if got == target:
        print('Done. Power-cycle the arm, then rerun this to confirm it stuck.')
    else:
        print('The write did not take. Some firmware refuses EEPROM writes')
        print('while torque is enabled or while the servo is moving - make')
        print('sure nothing else has the port open and try once more.')
    arm.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
