"""Check the taught box paths before running them. Nothing moves.

    python trimmer/check_paths.py

Everything that has gone wrong replaying a path was visible in the taught
numbers beforehand: a head angle in the forbidden middle, a move that crosses
the servo zero, or a waypoint sitting a long way from its neighbours because
it was captured while the arm was on its way somewhere else.

None of those need the arm to find. So this reads core/box_paths.py and says
what will happen, with a hand nowhere near the power switch.
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
import trimmer_config as tc
from pathplay import to_arc, head_ok, crosses_wrap, head_deg, head_label

PATHS = ['top', 'side_1', 'side_2', 'side_3', 'side_4']
NAMES = {tc.BASE_ID: 'base', tc.SHOULDER_ID: 'shldr', tc.ELBOW_ID: 'elbow',
         tc.WRIST_ID: 'wrist', tc.HEAD_ID: 'HEAD'}
# A jump bigger than this between neighbouring waypoints is worth a look. It is
# not wrong in itself - a deliberate retract is a big move - but it is also
# what an accidentally-recorded waypoint looks like.
BIG_MOVE_RAW = 900


def main():
    try:
        from box_paths import BOX_PATHS
    except Exception as e:
        print(f'no taught paths yet ({e})')
        print('run:  python trimmer/teach_box.py')
        return 1

    problems, notes = [], []
    print('\n' + '=' * 70)
    print('TAUGHT PATH CHECK')
    # Once the servo zero has been moved the arc no longer wraps, and printing
    # it as though it did - "down through 0/4095" - is just wrong.
    print(f'head arc: {tc.HEAD_ARC_HI} (anticlockwise limit) down to '
          f'{tc.HEAD_ARC_LO} (clockwise limit), flat at {tc.HEAD_FLAT_RAW}')
    if tc.HEAD_ARC_LO < tc.HEAD_ARC_HI:
        print(f'forbidden: everything outside {tc.HEAD_ARC_LO}-{tc.HEAD_ARC_HI}'
              f'  - one clean block, no wrap inside it')
    else:
        print(f'forbidden: {tc.HEAD_ARC_HI + 1}-{tc.HEAD_ARC_LO - 1}  '
              f'- the half containing 2048')
    print('=' * 70)

    prev_end = None
    for name in PATHS:
        wps = BOX_PATHS.get(name, [])
        if not wps:
            notes.append(f'{name} has no waypoints')
            print(f'\n{name}: (empty)')
            continue
        print(f'\n{name}: {len(wps)} waypoints')
        for i, wp in enumerate(wps):
            h = wp.get(tc.HEAD_ID)
            flag = ''
            if h is not None and not head_ok(h):
                flag = '  <-- HEAD IN THE FORBIDDEN MIDDLE'
                problems.append(f'{name} waypoint {i+1}: head {h} is unreachable')
            line = '  '.join(
                f'{NAMES.get(j, j)}:{wp[j]:5d}' for j in sorted(wp)
                if j != tc.HEAD_ID)
            if h is not None:
                line += f'   HEAD {head_label(h)}'
            print(f'   {i+1}. {line}{flag}')
            if i:
                moved = {j: abs(wp[j] - wps[i-1][j]) for j in wp if j in wps[i-1]}
                big = {j: d for j, d in moved.items() if d > BIG_MOVE_RAW}
                if big:
                    txt = ', '.join(f'{NAMES.get(j, j)} {d} raw '
                                    f'({d/4096*360:.0f} deg)'
                                    for j, d in sorted(big.items()))
                    print(f'      big jump from {i}: {txt}')
                    notes.append(f'{name} {i}->{i+1}: {txt}')
                if (tc.HEAD_ID in wp and tc.HEAD_ID in wps[i-1]
                        and crosses_wrap(wps[i-1][tc.HEAD_ID], wp[tc.HEAD_ID])):
                    print('      head crosses the servo zero here')
                    notes.append(f'{name} {i}->{i+1}: head crosses zero - the '
                                 f'move is stepped and verified, but watch it')
        # the join from the previous path
        if prev_end is not None and tc.HEAD_ID in prev_end and tc.HEAD_ID in wps[0]:
            a, b = prev_end[tc.HEAD_ID], wps[0][tc.HEAD_ID]
            d = abs(to_arc(b) - to_arc(a))
            if d > tc.HEAD_MAX_STEP_RAW:
                problems.append(f'joining into {name}: head moves {d} raw '
                                f'({d/4096*360:.0f} deg) in one go - too far, '
                                f'play-all will refuse')
            elif d > 200:
                notes.append(f'joining into {name}: head turns {d/4096*360:.0f} deg')
        prev_end = wps[-1]

    print('\n' + '=' * 70)
    if problems:
        print(f'  {len(problems)} PROBLEM(S):')
        for p in problems:
            print(f'   - {p}')
    else:
        print('  No blocking problems - the paths will play.')
    if notes:
        print('\n  Worth a look:')
        for n in notes:
            print(f'   - {n}')
    print('=' * 70 + '\n')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
