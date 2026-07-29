"""Teach the LEVEL cut — run once (camera + arm, torque off).

You teach the arm poses that hold the trimmer at your chosen cut height above
the flat deck, at two or more distances. Because the deck is level and the
base yaws about a vertical axis, holding one of these poses and sweeping ONLY
the base keeps the trimmer at constant height — a level cut, no kinematics.

At each of 2-4 distances:
  1. Put a reference (a tuft of the target grass, or a small marker) straight
     ahead at that distance. The screen shows its measured depth z.
  2. BY HAND (arm is limp — support it), lower the trimmer so its cutting line
     sits at the height you want the grass left at. Keep the base pointing
     straight ahead.
  3. Press ENTER to capture (z, pose).

Also capture REST (trimmer raised clear of the deck) with 'r'.

Writes trimmer_cut.py:  REST_POSE + CUT_POINTS = [(z, {sid:raw}), ...].
Picked up automatically. Nothing from earlier phases is touched.

Keys:  ENTER capture cut point · r capture REST · u undo · f finish · q abort
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

from trimmer_config import CAMERA_WIDTH, CAMERA_HEIGHT, JOINT_IDS
from trimmer_arm import TrimmerArm
from trim_detector import build_pipeline, _measure_z

OUT_PATH = str(CORE / 'trimmer_cut.py')
MIN_Z_SEP = 0.02


def fmt(pose):
    return '{' + ', '.join(f'{s}: {pose[s]}' for s in sorted(pose)) + '}'


def main():
    arm = TrimmerArm()
    arm.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    cx, cy = CAMERA_WIDTH // 2, int(CAMERA_HEIGHT * 0.6)
    print(__doc__)
    print(f"Joints: {JOINT_IDS}. Arm is LIMP — support it.\n")

    rest = None
    cut = []                      # (z, pose)
    last_release = time.monotonic()
    z_now = None
    message, message_until = '', 0.0
    window = 'SO-101 TRIM CUT CALIBRATION (ENTER cut · r rest · f finish · q)'
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
                z_now = _measure_z(depth, cx, cy, win=8)
                cv2.drawMarker(image, (cx, cy), (0, 255, 255),
                               cv2.MARKER_CROSS, 22, 2)
                cv2.putText(image,
                            f"z={z_now:.3f}m" if z_now else "z=?  (aim at deck)",
                            (cx - 60, cy - 16), 0, 0.6, (0, 255, 255), 2)
                cv2.putText(image, f"REST {'set' if rest else '-'}   "
                            f"CUT pts {len(cut)}", (8, 24), 0, 0.6,
                            (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 48), 0, 0.55, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                print("Aborted — nothing written.")
                return
            if key == ord('r'):
                rest = arm.read_pose()
                message, message_until = 'REST captured', now + 1.5
            elif key == ord('u'):
                if cut:
                    cut.pop()
                    message = f'undo — {len(cut)} cut points'
                    message_until = now + 1.5
            elif key == ord('f'):
                if rest is None or len(cut) < 2:
                    message = 'need REST + >= 2 cut points'
                    message_until = now + 2.0
                else:
                    break
            elif key in (13, 10):
                if z_now is None:
                    message, message_until = 'no depth — aim at the deck', now + 2.0
                elif any(abs(z_now - z) < MIN_Z_SEP for z, _ in cut):
                    message = f'z={z_now:.3f} too close to an existing point'
                    message_until = now + 2.5
                else:
                    pose = arm.read_pose()
                    cut.append((z_now, pose))
                    print(f"cut point {len(cut)}: z={z_now:.3f} pose={fmt(pose)}")
                    message = f'captured z={z_now:.3f}'
                    message_until = now + 1.5
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        arm.close()

    cut.sort(key=lambda p: p[0])
    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_trim.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write(f"REST_POSE = {fmt(rest)}\n")
        f.write("CUT_POINTS = [\n")
        for z, pose in cut:
            f.write(f"    ({z:.4f}, {fmt(pose)}),\n")
        f.write("]\n")
    print(f"\nWrote {OUT_PATH}  (REST + {len(cut)} level-cut points)")
    print("Run  python main_trimmer.py  to trim autonomously.")


if __name__ == '__main__':
    main()
