"""Print exactly what the arm will be commanded, without moving anything.

    python trimmer/show_plan.py

Two rounds of "this changed nothing" is a good reason to stop reasoning about
what the code does and look at what it actually produces. This applies every
transform the player applies - height levelling, pitch lock, smoothing,
pacing - and prints the result. If the numbers here do not differ from the
taught ones, the change is not reaching the arm, and that is a different
problem from the change being wrong.
"""
import sys as _sys, pathlib as _pl
_d = _pl.Path(__file__).resolve().parent
while not (_d / 'paths.py').is_file() and _d != _d.parent:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
import paths as _paths
_paths.bootstrap()

import trimmer_config as tc
import pathplay
from box_paths import BOX_PATHS


def main():
    print(f'\nconfig loaded from: {tc.__file__}')
    print(f'pathplay loaded from: {pathplay.__file__}')
    print(f'LEVEL_HEIGHT_PATHS = {getattr(tc, "LEVEL_HEIGHT_PATHS", "MISSING")}')
    print(f'LEVEL_PITCH_PATHS  = {getattr(tc, "LEVEL_PITCH_PATHS", "MISSING")}')
    print(f'LEVEL_HEIGHT_TRIM_MM = {getattr(tc, "LEVEL_HEIGHT_TRIM_MM", "MISSING")}'
          f'   (mm above the plane taught in level_plane.py)')
    for name, wps in BOX_PATHS.items():
        if len(wps) < 4:
            continue
        print('\n' + '=' * 66)
        print(f'{name}: {len(wps)} taught waypoints')
        print('=' * 66)
        out = wps
        if name in getattr(tc, 'LEVEL_HEIGHT_PATHS', ()):
            out = pathplay.level_height(out, name, log=lambda m: print(' ', m))
        else:
            print('  height levelling NOT applied to this path')
        if name in getattr(tc, 'LEVEL_PITCH_PATHS', ()):
            out = pathplay.level_pitch(out, name, log=lambda m: print(' ', m))
        else:
            print('  pitch lock NOT applied to this path')
        print(f'\n{"wp":>3} {"base":>6} {"shoulder":>8}  '
              f'{"elbow taught":>12} {"-> commanded":>12}  '
              f'{"wrist taught":>12} {"-> commanded":>12}')
        changed = 0
        for i, (a, b) in enumerate(zip(wps, out)):
            if a[tc.ELBOW_ID] != b[tc.ELBOW_ID] or a[tc.WRIST_ID] != b[tc.WRIST_ID]:
                changed += 1
            if i < 8 or i >= len(wps) - 2:
                print(f'{i+1:3d} {a[tc.BASE_ID]:6d} {a[tc.SHOULDER_ID]:8d}  '
                      f'{a[tc.ELBOW_ID]:12d} {b[tc.ELBOW_ID]:12d}  '
                      f'{a[tc.WRIST_ID]:12d} {b[tc.WRIST_ID]:12d}')
            elif i == 8:
                print('  ...')
        print(f'\n{changed} of {len(wps)} waypoints differ from the taught file.')
        if not changed:
            print('  NOTHING IS BEING CHANGED - the transforms are not running.')


if __name__ == '__main__':
    main()
