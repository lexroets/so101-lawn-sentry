"""See, bookmark and roll back calibrations.

    python tools/calibrations.py                 list everything kept
    python tools/calibrations.py keep "demo"     bookmark the current one
    python tools/calibrations.py restore 3       go back to number 3
    python tools/calibrations.py show 3          print it
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
import sys

import cal_store
from paths import DATA


def show_list(rows):
    if not rows:
        print('\nnothing kept yet - the next calibration you save will start the list.')
        return
    live = None
    try:
        live = json.loads(cal_store.LIVE.read_text()).get('created')
    except (OSError, ValueError):
        pass
    print(f'\n{"#":>3}  {"when":<17} {"method":<20} {"lawn":<14} rest  note')
    print('-' * 78)
    for i, e in enumerate(rows, 1):
        mark = '*' if live and e['created'] == live else ' '
        print(f'{i:>3}{mark} {e["created"][:16]:<17} {e["type"]:<20} '
              f'{e["lawn"]:<14} {"yes" if e["rest"] else " - ":<4}  {e["note"]}')
    print('\n  * = the one in use now')
    print('  restore with:  python tools/calibrations.py restore <number>\n')


def main():
    rows = cal_store.entries()
    args = sys.argv[1:]
    if not args:
        show_list(rows)
        # Thonny cannot pass arguments, so offer the one thing you would
        # actually want next rather than printing a command line you have no
        # way to type.
        if rows:
            try:
                pick = input('restore which number? (ENTER to leave it alone): ').strip()
            except EOFError:
                pick = ''
            if pick:
                args = ['restore', pick]
        if not args:
            return
    cmd = args[0]

    if cmd == 'keep':
        label = args[1] if len(args) > 1 else 'kept'
        p = cal_store.archive(label=label)
        print(f'kept as {p.name}' if p else 'no current calibration to keep')
        return

    if cmd in ('restore', 'show'):
        if len(args) < 2:
            print(f'which one? run with no arguments to see the list')
            return
        try:
            e = rows[int(args[1]) - 1]
        except (ValueError, IndexError):
            print('no entry with that number')
            return
        if cmd == 'show':
            print(e['path'].read_text())
            return
        carry = None
        if not e['rest']:
            print(f'\nNOTE: {e["name"]} has no rest pose recorded, so launch.py')
            print('cannot put the arm back where this was measured. A ground')
            print('plane is only true for the camera pose it was measured at.')
            try:
                live = json.loads(cal_store.LIVE.read_text()).get('rest_pose')
            except (OSError, ValueError):
                live = None
            if live:
                print(f'\nThe calibration in use now has a rest pose: {live}')
                print('Lending it to this one is only sound if the arm has not')
                print('been re-mounted between the two. It is a guess, and a')
                print('wrong rest pose aims the plane where the camera is not')
                print('looking - which is the failure this system keeps having.')
                if input('lend it? [y/N]: ').strip().lower().startswith('y'):
                    carry = cal_store.LIVE
            print()
        cal_store.restore(e['path'])
        if carry is not None:
            # LIVE is now the restored one; take the pose from the archive copy
            # of what we just replaced
            prev = cal_store.entries()
            src = next((x['path'] for x in prev
                        if 'replaced_by_restore' in x['name']), None)
            if src and cal_store.carry_rest_pose(cal_store.LIVE, src):
                print('rest pose carried over - launch.py will use it')
        print(f'restored {e["name"]} -> data/lawn_calibration.json')
        print('run:  python launch.py')
        return

    print(__doc__)


if __name__ == '__main__':
    main()
