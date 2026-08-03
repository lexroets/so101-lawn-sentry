"""Put the lawn the right way round, without recalibrating.

Sliding and rotating the rectangle onto the real lawn settles the GEOMETRY, but
it leaves one thing open that the picture cannot show you: which corner counts
as (0, 0) and which axis is the width. The dashboard is not neutral about that.
It draws the mower pinned to the bottom-centre of the map and swings the view
cone from there, which is only sensible if the camera is behind the near edge -
that is, at negative y, near x = W/2.

The current calibration has the camera at x=3.52, y=+1.68 on a 3.5 x 2.1 m
lawn: off the far right corner. The plane is right, the labelling is a quarter
turn out, and every detection would land on the map rotated.

That is fixable arithmetic, not another calibration session. There are only
four ways to label a rectangle's corners, and exactly one puts the camera
behind the near edge. This finds it and rewrites the homography accordingly,
swapping the stated width and length when the fix is an odd quarter turn.

    python tools/fix_orientation.py            # show what it would do
    python tools/fix_orientation.py --apply    # do it (the old one is kept)
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
import argparse
import math

import cv2
import numpy as np

import cal_store


def quarter_turn(H, W, L, k):
    """Relabel the lawn's corners by k quarter turns. The ground does not move.

    A rotation in lawn coordinates is just a 3x3 applied after H, so this is
    exact - no re-fitting, nothing lost.
    """
    k %= 4
    if k == 0:
        return H, W, L
    a = math.radians(90.0 * k)
    c, s = math.cos(a), math.sin(a)
    # rotate about the origin, then shift the rectangle back into 0..W, 0..L
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    corners = np.array([[0, 0], [W, 0], [W, L], [0, L]], float)
    rot = (R[:2, :2] @ corners.T).T
    shift = -rot.min(axis=0)
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
    newH = T @ R @ np.asarray(H, float)
    newW, newL = (W, L) if k % 2 == 0 else (L, W)
    return newH / newH[2, 2], newW, newL


def camera_xy(H, w_px, h_px):
    """Where the camera sits, in lawn metres, straight from the homography.

    The camera centre is the one point that projects nowhere, which in
    homography terms is the null vector of the image-to-lawn map's inverse -
    or more simply, decompose and read the translation back out.
    """
    from camera_intrinsics import get as get_K
    K, _ = get_K(int(w_px), int(h_px), prefer_device=False, quiet=True)
    M = np.linalg.inv(K) @ np.linalg.inv(np.asarray(H, float))
    r1, r2, t = M[:, 0].copy(), M[:, 1].copy(), M[:, 2].copy()
    n = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    if n < 1e-12:
        return None
    r1, r2, t = r1 / n, r2 / n, t / n
    Rm = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(Rm)
    Rm = U @ Vt
    C = -Rm.T @ t
    return float(C[0]), float(C[1]), abs(float(C[2]))


def score(cx, cy, W, L):
    """How wrong this labelling is: the camera should sit behind the near edge.

    Behind means y below 0. Off to one side is tolerable; being level with the
    lawn or beyond the far edge is not, and neither is sitting out past a
    corner. Lower is better.
    """
    return (max(0.0, cy) * 3.0             # anywhere at or past y=0 is bad
            + abs(cx - W / 2.0)            # ideally centred across the width
            + max(0.0, -cy - 6.0))         # implausibly far back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    cal = cal_store.load()
    H = np.asarray(cal['homography_image_to_lawn'], float)
    W = float(cal['lawn_width_m'])
    L = float(cal['lawn_length_m'])
    wpx = int(cal.get('camera_width_px', 640))
    hpx = int(cal.get('camera_height_px', 480))

    best = None
    print(f'\nlawn {W} x {L} m, {cal.get("calibration_type","?")}\n')
    print(f'{"turn":>6}  {"lawn":>12}  {"camera lands at":>22}   verdict')
    print('-' * 66)
    for k in range(4):
        Hk, Wk, Lk = quarter_turn(H, W, L, k)
        pos = camera_xy(Hk, wpx, hpx)
        if pos is None:
            continue
        cx, cy, cz = pos
        sc = score(cx, cy, Wk, Lk)
        ok = 'behind the near edge' if cy < -0.05 else 'NOT behind the lawn'
        print(f'{k*90:5d}d  {Wk:5.2f} x {Lk:4.2f}  '
              f'x={cx:6.2f} y={cy:6.2f} h={cz:4.2f}   {ok}')
        if best is None or sc < best[0]:
            best = (sc, k, Hk, Wk, Lk, cx, cy, cz)

    if best is None:
        print('\ncould not decompose the homography - nothing changed')
        return
    _, k, Hk, Wk, Lk, cx, cy, cz = best
    print()
    if k == 0:
        print('Already the right way round. Nothing to do.')
        return
    print(f'Best is a {k*90} degree turn: the camera then sits {-cy:.2f} m '
          f'behind the near edge,')
    print(f'{cz:.2f} m up, {abs(cx - Wk/2):.2f} m off centre.')
    if k % 2:
        print(f'The stated size swaps to {Wk} x {Lk} m - an odd quarter turn '
              f'means what you\ncalled the width runs away from the camera, '
              f'not across it.')
    # Thonny runs a file with no arguments and there is no obvious way to add
    # one, so asking is the only way this is usable from the editor. From a
    # terminal, --apply skips the question.
    if not args.apply:
        try:
            answer = input('\nApply this turn? [y/N]: ').strip().lower()
        except EOFError:
            answer = ''
        if not answer.startswith('y'):
            print('Nothing written.')
            return

    cal['homography_image_to_lawn'] = Hk.tolist()
    cal['lawn_width_m'] = Wk
    cal['lawn_length_m'] = Lk
    cal.pop('image_points_tl_tr_br_bl', None)      # rederived below
    cal.pop('reference_frame_points', None)
    cal.pop('_corners_derived', None)
    cal['camera_position_m'] = [round(cx, 3), round(cy, 3), round(cz, 3)]
    cal['orientation_fixed_deg'] = k * 90
    cal['note'] = (str(cal.get('note', '')) + f' | relabelled {k*90} deg so the '
                   f'camera is behind the near edge').strip(' |')
    cal_store.save(cal_store.normalise(cal), label='orientation_fixed')
    print(f'\nDone - lawn relabelled by {k*90} degrees, now {Wk} x {Lk} m.')
    print('The previous one is kept:  python tools/calibrations.py')
    print('Run:  python launch.py')


if __name__ == '__main__':
    main()
