"""Recalibrate the trimmer reach against REAL targets — run once.

Better than calibrate_trim.py for reaching objects: it measures the depth of
the DETECTED object (the same way reach_object.py does at runtime) and you
place the trimmer head physically at the object's top. Two steps per point
keep the depth reliable (the camera must see the target from far enough that
it isn't inside the >0.18m dead zone):

  STEP 1  Put a target (a Ryobi battery, standing proud of the desk) in view.
          With the arm RAISED so the camera sees it, press  z  to LOCK its
          distance. The screen shows the locked z.
  STEP 2  BY HAND (arm is limp — support it) lower the trimmer head to the
          TOP of that battery, then press ENTER to capture the pose paired
          with the locked z.

Do 3-4 targets spanning near-to-far (move the battery each time). Also press
r once with the arm raised clear to capture REST.

Writes trimmer_cut.py (REST_POSE + CUT_POINTS), replacing the old table.

Keys:  z lock distance · ENTER capture pose · r REST · u undo · f finish · q abort
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
from paths import CORE, ensure as _ensure
_ensure()
import os
import time
import cv2
import numpy as np

from trimmer_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, JOINT_IDS, MIN_RELIABLE_Z,
    OBJECT_MIN_PROUD_M, GRASS_MAX_PROUD_M, OBJECT_MIN_AREA_PX, OBJECT_MAX_AREA_PX,
)
from trimmer_arm import TrimmerArm
from trim_detector import build_pipeline, detect_patches

OUT_PATH = str(CORE / 'trimmer_cut.py')
MIN_Z_SEP = 0.02


def fmt(pose):
    return '{' + ', '.join(f'{s}: {pose[s]}' for s in sorted(pose)) + '}'


def main():
    arm = TrimmerArm()
    arm.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    cx = CAMERA_WIDTH // 2
    print(__doc__)
    print(f"Joints: {JOINT_IDS}. Arm is LIMP — support it.\n")

    rest = None
    pts = []                      # (z, pose)
    pending_z = None
    last_release = time.monotonic()
    obj_z = None
    message, message_until = '', 0.0
    window = 'SO-101 REACH CALIBRATION (z lock · ENTER pose · r rest · f finish)'
    try:
        while pipeline.isRunning():
            now = time.monotonic()
            if now - last_release > 2.0:
                arm.torque_off()
                last_release = now
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                objs = detect_patches(image, depth,
                                      min_proud=OBJECT_MIN_PROUD_M,
                                      max_proud=GRASS_MAX_PROUD_M,
                                      min_area=OBJECT_MIN_AREA_PX,
                                      max_area=OBJECT_MAX_AREA_PX)
                obj = min(objs, key=lambda p: abs(p['u'] - cx)) if objs else None
                obj_z = obj['z'] if obj else None
                if obj is not None:
                    x, y, w, h = obj['bbox']
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f"z={obj_z:.3f}m", (x, max(14, y - 6)),
                                0, 0.55, (0, 255, 0), 2)
                l1 = ("STEP 1: aim at target (raised), press z" if pending_z is None
                      else f"STEP 2: lower trimmer to top, press ENTER  "
                           f"[z locked {pending_z:.3f}]")
                cv2.putText(image, l1, (8, 24), 0, 0.55, (0, 255, 255), 2)
                cv2.putText(image, f"REST {'set' if rest else '-'}   "
                            f"points {len(pts)}", (8, 48), 0, 0.55,
                            (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 72), 0, 0.55, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                print("Aborted — nothing written.")
                return
            if key == ord('r'):
                rest = arm.read_pose()
                message, message_until = 'REST captured', now + 1.5
            elif key == ord('z'):
                if obj_z is None:
                    message = 'no target in view (raise arm / move closer to >0.2m)'
                    message_until = now + 2.5
                elif obj_z < MIN_RELIABLE_Z + 0.01:
                    message = f'z={obj_z:.3f} too close — move target back'
                    message_until = now + 2.5
                elif any(abs(obj_z - z) < MIN_Z_SEP for z, _ in pts):
                    message = f'z={obj_z:.3f} too near an existing point'
                    message_until = now + 2.5
                else:
                    pending_z = obj_z
                    message = f'z locked {pending_z:.3f} — now lower trimmer, ENTER'
                    message_until = now + 3.0
            elif key == ord('u'):
                if pts:
                    pts.pop()
                    message = f'undo — {len(pts)} points'
                    message_until = now + 1.5
            elif key == ord('f'):
                if rest is None or len(pts) < 2:
                    message = 'need REST + >= 2 points'
                    message_until = now + 2.0
                else:
                    break
            elif key in (13, 10):
                if pending_z is None:
                    message = 'lock a distance first (press z)'
                    message_until = now + 2.0
                else:
                    pose = arm.read_pose()
                    pts.append((pending_z, pose))
                    print(f"point {len(pts)}: z={pending_z:.3f} pose={fmt(pose)}")
                    message = f'captured point {len(pts)}'
                    message_until = now + 1.5
                    pending_z = None
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        arm.close()

    pts.sort(key=lambda p: p[0])
    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_reach.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Reach table taught against real targets (object depth +\n")
        f.write("# trimmer-head-at-top pose). Replaces any earlier table.\n")
        f.write(f"REST_POSE = {fmt(rest)}\n")
        f.write("CUT_POINTS = [\n")
        for z, pose in pts:
            f.write(f"    ({z:.4f}, {fmt(pose)}),\n")
        f.write("]\n")
    print(f"\nWrote {OUT_PATH}  (REST + {len(pts)} reach points)")
    print("Run  python reach_object.py  to check the head lands on the top.")


if __name__ == '__main__':
    main()
