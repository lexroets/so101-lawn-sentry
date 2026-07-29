"""Re-express an existing calibration to cover the whole lawn.

If you calibrated on a small reference box - which is what you must do when the
marker is only readable close to the camera - the saved map describes the BOX,
not the lawn. The geometry is already right; only the origin is in the wrong
place, and moving an origin is a translation applied after the homography.

So this needs no re-walking and no camera. Tell it where the box sits and how
big the lawn really is:

    python calibrate/set_lawn.py --lawn 3.5 2.1 --offset 1.25 0.5

`--offset` is where the box's FIRST corner (the one you placed the marker on
first) sits in lawn coordinates, measured from the lawn corner nearest the
camera on the left:

        X across the lawn (the 3.5 m direction)
        Y away from the camera (the 2.1 m direction)

Get it wrong and everything on the map is shifted by the same amount, so
measure those two numbers with a tape rather than estimating.
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
import json
import shutil
from datetime import datetime

import numpy as np

from paths import DATA


def main():
    ap = argparse.ArgumentParser(
        description='Move a box calibration into full-lawn coordinates.')
    ap.add_argument('--lawn', nargs=2, type=float, required=True,
                    metavar=('W', 'L'), help='real lawn size in metres')
    ap.add_argument('--offset', nargs=2, type=float, required=True,
                    metavar=('X', 'Y'),
                    help="where the box's first corner sits on the lawn")
    ap.add_argument('--file', default=None, help='calibration to modify')
    ap.add_argument('--out', default=None, help='write here instead')
    args = ap.parse_args()

    src = _pl.Path(args.file) if args.file else DATA / 'lawn_calibration.json'
    dst = _pl.Path(args.out) if args.out else src
    if not src.is_file():
        raise SystemExit(f'no calibration at {src}')

    d = json.loads(src.read_text())
    H = np.asarray(d['homography_image_to_lawn'], float)

    box = d.get('reference_box_m') or [d['lawn_width_m'], d['lawn_length_m']]
    W, L = float(args.lawn[0]), float(args.lawn[1])
    ox, oy = float(args.offset[0]), float(args.offset[1])

    if d.get('lawn_origin_offset_m'):
        raise SystemExit(
            'this calibration has already been moved into lawn coordinates.\n'
            'Re-run the calibration, or pass --file with the original.')

    if ox < 0 or oy < 0 or ox + box[0] > W + 1e-6 or oy + box[1] > L + 1e-6:
        print(f'  WARNING: a {box[0]} x {box[1]} m box at ({ox}, {oy}) does not')
        print(f'  fit inside a {W} x {L} m lawn. Check the offset.')

    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], float)
    H2 = T @ H
    H2 /= H2[2, 2]

    if dst == src:
        bak = src.with_name(src.stem + '_BOXONLY.json')
        shutil.copy2(src, bak)
        print(f'  original kept at {bak.name}')

    d['homography_image_to_lawn'] = H2.tolist()
    d['lawn_width_m'] = W
    d['lawn_length_m'] = L
    d['reference_box_m'] = [float(box[0]), float(box[1])]
    d['lawn_origin_offset_m'] = [ox, oy]
    d['moved_to_lawn_at'] = datetime.now().isoformat(timespec='seconds')
    dst.write_text(json.dumps(d, indent=2))

    frac = (box[0] * box[1]) / max(W * L, 1e-9)
    print(f'\n  {dst.name} now describes a {W} x {L} m lawn.')
    print(f'  Calibrated on a {box[0]} x {box[1]} m box at ({ox}, {oy}) m '
          f'- {frac*100:.0f}% of the area.')
    print('\n  Accuracy is best near the box and degrades further out. The')
    print('  reported marker error still applies AT the box; treat the far')
    print('  corners as a few centimetres looser.')
    print(f'\n  Next:  python launch.py')


if __name__ == '__main__':
    main()
