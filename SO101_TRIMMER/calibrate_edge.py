"""Teach the VERTICAL edging pose — run once (camera + arm, torque off).

For edging a lawn against a driveway/path the trimmer is held VERTICAL, so its
spinning line cuts a clean vertical face down the border. This teaches that
orientation as a second reach table (EDGE_POINTS), separate from the flat
"cut the top" table (CUT_POINTS) — the run-time machinery is the same, only
the pose differs.

Same two-step-per-point flow as calibrate_reach.py (keeps the depth reliable
outside the >0.18m dead zone):

  STEP 1  Put a reference at the edge line, in view, arm RAISED. Press  z  to
          LOCK its distance (shown on screen).
  STEP 2  BY HAND (arm limp — support it) orient the trimmer VERTICAL and set
          it against the edge at the height you want it to cut down to, then
          press ENTER to capture the pose paired with the locked z.

Do 2-3 references spanning near-to-far. This does NOT need to be perfect — a
rough alignment is enough to start.

Writes trimmer_edge.py (EDGE_POINTS). REST is shared from the flat calibration.

Keys:  z lock distance · ENTER capture pose · u undo · f finish · q abort
"""
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

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'trimmer_edge.py')
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
    print(f"Joints: {JOINT_IDS}. Arm is LIMP — support it. Hold trimmer "
          "VERTICAL.\n")

    pts = []
    pending_z = None
    last_release = time.monotonic()
    obj_z = None
    message, message_until = '', 0.0
    window = 'SO-101 EDGE CALIBRATION (z lock · ENTER pose · f finish · q)'
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
                l1 = ("STEP 1: aim at edge (raised), press z" if pending_z is None
                      else f"STEP 2: hold trimmer VERTICAL at edge, ENTER  "
                           f"[z {pending_z:.3f}]")
                cv2.putText(image, l1, (8, 24), 0, 0.55, (0, 255, 255), 2)
                cv2.putText(image, f"edge points {len(pts)}", (8, 48), 0, 0.55,
                            (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 72), 0, 0.55, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                print("Aborted — nothing written.")
                return
            if key == ord('z'):
                if obj_z is None:
                    message = 'no reference in view (raise arm / >0.2m)'
                    message_until = now + 2.5
                elif obj_z < MIN_RELIABLE_Z + 0.01:
                    message = f'z={obj_z:.3f} too close — move it back'
                    message_until = now + 2.5
                elif any(abs(obj_z - z) < MIN_Z_SEP for z, _ in pts):
                    message = f'z={obj_z:.3f} too near an existing point'
                    message_until = now + 2.5
                else:
                    pending_z = obj_z
                    message = f'z locked {pending_z:.3f} — hold vertical, ENTER'
                    message_until = now + 3.0
            elif key == ord('u'):
                if pts:
                    pts.pop()
                    message = f'undo — {len(pts)} points'
                    message_until = now + 1.5
            elif key == ord('f'):
                if len(pts) < 2:
                    message = 'need >= 2 points'
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
                    print(f"edge point {len(pts)}: z={pending_z:.3f} "
                          f"pose={fmt(pose)}")
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
        f.write("# Written by calibrate_edge.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# VERTICAL edging reach table (trimmer held vertical at the\n")
        f.write("# lawn edge). Used by  reach_object.py edge.\n")
        f.write("EDGE_POINTS = [\n")
        for z, pose in pts:
            f.write(f"    ({z:.4f}, {fmt(pose)}),\n")
        f.write("]\n")
    print(f"\nWrote {OUT_PATH}  ({len(pts)} edge points)")
    print("Try it:  python reach_object.py edge")


if __name__ == '__main__':
    main()
