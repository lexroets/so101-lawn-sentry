"""Why is the head landing short? Look, without moving the arm.

The reach table maps a measured distance to a taught arm pose. It can only be
right between the distances it was actually taught at, and it fails in two
quiet ways - neither of which announces itself while you are watching the arm:

  OUTSIDE THE TABLE   a target further away than the furthest taught point is
                      silently CLAMPED to that furthest pose. The arm reaches
                      as far as it was ever shown and stops, however far the
                      object really is. Consistent, always short, never long.

  A BAD TAUGHT POINT  reaching further means the shoulder rises and the elbow
                      opens. If one captured pose breaks that trend, then
                      interpolating through it pulls the head BACKWARDS as the
                      target gets further away.

This reads the camera and the table and says which is happening. The arm is
never commanded - it is safe to run with the arm parked.

    python trimmer/reach_check.py

    q   quit
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
import cv2
import numpy as np

import trimmer_config as tc
from trim_detector import build_pipeline, detect_patches
from camera_session import CameraSession

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'reach check - is the table covering your targets?'
SHOULDER, ELBOW = tc.SHOULDER_ID, tc.ELBOW_ID


def audit_table():
    """Report the taught table's range and any point that breaks the trend."""
    pts = sorted(tc.CUT_POINTS, key=lambda p: p[0])
    lo, hi = pts[0][0], pts[-1][0]
    print('\n' + '=' * 70)
    print('THE TAUGHT REACH TABLE')
    print('=' * 70)
    print(f'\n{"z (m)":>7} {"shoulder":>9} {"elbow":>7} {"wrist":>7}')
    bad = []
    for i, (z, p) in enumerate(pts):
        flag = ''
        if i:
            pz, pp = pts[i - 1]
            # Further away = shoulder up, elbow open. Either one reversing
            # means the head retracts across this segment.
            if pp[SHOULDER] > p[SHOULDER] or pp[ELBOW] > p[ELBOW]:
                # elbow direction depends on the arm's geometry, so only call
                # it out when BOTH reverse together - that is unambiguous
                if pp[SHOULDER] > p[SHOULDER] and pp[ELBOW] < p[ELBOW]:
                    flag = '  <-- REVERSES the trend'
                    bad.append((pz, z))
        print(f'{z:7.3f} {p[SHOULDER]:9d} {p[ELBOW]:7d} {p[tc.WRIST_ID]:7d}{flag}')
    print(f'\ncovers {lo:.3f} m to {hi:.3f} m')
    print(f'anything nearer than {lo:.2f} m or further than {hi:.2f} m is CLAMPED')
    print('to the end pose - the arm reaches as far as it was taught, no more.')
    for a, b in bad:
        print(f'\nWARNING: between {a:.3f} and {b:.3f} m the shoulder DROPS while')
        print('the elbow opens - across that stretch the head moves BACK as the')
        print('target gets further away. That point looks like a bad capture.')
    print('=' * 70 + '\n')
    return lo, hi, bad


def verdict(z, lo, hi, bad):
    if z > hi:
        return (f'BEYOND the table by {(z-hi)*100:.0f} cm - the arm will stop '
                f'about that far short', (60, 60, 255))
    if z < lo:
        return (f'NEARER than the table by {(lo-z)*100:.0f} cm - the arm will '
                f'over-reach', (60, 60, 255))
    for a, b in bad:
        if a <= z <= b:
            return ('inside the reversed segment - the pose here is unreliable',
                    (60, 160, 255))
    return ('inside the taught range', (90, 255, 90))


def main():
    lo, hi, bad = audit_table()
    print('Put a battery in front of the camera. Reading distances - q to quit.\n')
    cam = CameraSession(build_pipeline)
    seen = []
    try:
        while True:
            g = cam.get()
            if g is None:
                if cam.crashed() and not cam.restart():
                    break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            image = g['rgb'].getCvFrame()
            depth = g['depth'].getFrame().astype(np.float32) / 1000.0
            # the same thresholds reach_object.py uses, so this sees exactly
            # what it sees - a diagnostic that disagrees with the thing it is
            # diagnosing is worse than none
            patches = detect_patches(image, depth,
                                     min_proud=tc.OBJECT_MIN_PROUD_M,
                                     max_proud=tc.GRASS_MAX_PROUD_M,
                                     min_area=tc.OBJECT_MIN_AREA_PX,
                                     max_area=tc.OBJECT_MAX_AREA_PX)
            for p in patches:
                x, y, w, h = p['bbox']
                msg, col = verdict(p['z'], lo, hi, bad)
                cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)
                cv2.putText(image, f"{p['z']:.3f} m", (x, max(14, y - 22)),
                            FONT, 0.55, col, 2)
                cv2.putText(image, msg, (x, max(30, y - 6)), FONT, 0.42, col, 1)
            if patches:
                seen.append(patches[0]['z'])
                seen[:] = seen[-60:]
            hdr = [f'taught table: {lo:.2f} - {hi:.2f} m',
                   (f'nearest target: {patches[0]["z"]:.3f} m'
                    if patches else 'no target detected')]
            if len(seen) > 10:
                hdr.append(f'steady at {np.median(seen):.3f} m '
                           f'(spread {np.ptp(seen)*100:.1f} cm)')
            cv2.rectangle(image, (0, 0), (image.shape[1], 24 * len(hdr) + 8),
                          (16, 16, 16), -1)
            for i, t in enumerate(hdr):
                cv2.putText(image, t, (10, 20 + i * 23), FONT, 0.5,
                            (255, 220, 90), 1)
            cv2.imshow(WIN, image)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    if seen:
        z = float(np.median(seen))
        print(f'\nyour targets sit at about {z:.3f} m')
        if z > hi:
            print(f'The table stops at {hi:.3f} m, so every reach is clamped '
                  f'{(z-hi)*100:.0f} cm short.')
            print('Re-teach with a point at (or past) your working distance:')
            print('    python trimmer/calibrate/calibrate_reach.py')
        elif any(a <= z <= b for a, b in bad):
            print('That falls in the reversed segment - re-teach that point.')
            print('    python trimmer/calibrate/calibrate_reach.py')
        else:
            print('The table covers this, so the shortfall is a HEIGHT offset,')
            print('not a reach one. In reach_object.py press  l  to lower the')
            print('head until it lands, then  p  to save it.')


if __name__ == '__main__':
    main()
