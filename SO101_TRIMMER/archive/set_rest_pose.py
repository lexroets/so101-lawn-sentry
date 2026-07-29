"""Stamp the arm's CURRENT pose into the existing calibration.

The homography describes the ground as seen from one camera pose. If the
calibration does not record which pose that was, launch.py has nothing to
return the arm to, so the map gets plotted from wherever the arm happens to be
sitting - which looks like a randomly rotated lawn.

If your calibration is already good and the arm is sitting where it was when
you made it, this saves the pose without redoing anything:

    python calibrate/set_rest_pose.py

Aim the camera at the lawn the way the calibration expects FIRST, then run it.
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
from datetime import datetime

from paths import DATA


def main():
    cal = DATA / 'lawn_calibration.json'
    if not cal.is_file():
        raise SystemExit(f'no calibration at {cal}')
    d = json.loads(cal.read_text())

    try:
        from trimmer_arm import TrimmerArm
        arm = TrimmerArm()
        pose = arm.read_pose()
        arm.close()
    except Exception as e:
        raise SystemExit(f'could not read the arm: {e}')
    if not pose:
        raise SystemExit('the servos did not reply')

    old = d.get('rest_pose')
    d['rest_pose'] = {str(k): int(v) for k, v in pose.items()}
    d['rest_pose_set'] = datetime.now().isoformat(timespec='seconds')
    cal.write_text(json.dumps(d, indent=2))

    print(f'\n  rest pose saved: {d["rest_pose"]}')
    if old:
        print(f'  (replaced {old})')
    print('\n  launch.py will now drive the arm here before scanning, so the')
    print('  lawn is plotted from the pose the calibration was measured at.')
    print('\n  IMPORTANT: this only helps if the camera is aimed the way it was')
    print('  when you calibrated. If the map still looks wrong, the calibration')
    print('  itself needs redoing:  python calibrate/calibrate_lawn_scan.py')


if __name__ == '__main__':
    main()
