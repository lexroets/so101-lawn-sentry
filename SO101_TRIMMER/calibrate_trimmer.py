"""Teach the trimmer arm its poses by demonstration — run once.

The arm goes LIMP (torque off) so you move it by hand. You capture:

  1. REST      — a safe pose with the trimmer clear of the ground, that the
                 arm returns to between patches and on shutdown.
  2. SWEEP     — a series of waypoints that trace ONE cutting pass: lower the
                 trimmer into the grass and draw it across the patch. Capture
                 4-8 points along that path. Replay blends through them into a
                 continuous natural sweep.

Because the trimmer adds weight, SUPPORT THE ARM WITH ONE HAND while you press
Enter with the other — a fully limp arm will droop.

Writes trimmer_poses.py, picked up automatically by trimmer_config.py.
Nothing from the battery project is touched.

Terminal controls (type then Enter):
  <Enter>  capture the current pose as the next SWEEP waypoint
  r        (re)capture the REST pose
  u        undo the last SWEEP waypoint
  f        finish and write the file  (needs REST + >= 2 sweep points)
  q        abort without writing
"""
import os
import time

from trimmer_config import JOINT_IDS
from trimmer_arm import TrimmerArm

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'trimmer_poses.py')


def fmt(pose):
    return '{' + ', '.join(f'{s}: {pose[s]}' for s in sorted(pose)) + '}'


def main():
    arm = TrimmerArm()
    arm.torque_off()
    print(__doc__)
    print(f"Controlled joints: {JOINT_IDS}   (no roll, no gripper)")
    print("Arm is LIMP — support it by hand.\n")

    rest = None
    sweep = []
    last_release = time.monotonic()
    try:
        while True:
            # Keep the bus releasing torque so it stays limp between captures.
            if time.monotonic() - last_release > 2.0:
                arm.torque_off()
                last_release = time.monotonic()
            state = f"[REST {'set' if rest else '—'}]  [SWEEP {len(sweep)} pts]"
            cmd = input(f"{state}  Enter=capture sweep · r=rest · u=undo · "
                        "f=finish · q=abort > ").strip().lower()
            if cmd == 'q':
                print("Aborted — nothing written.")
                return
            if cmd == 'r':
                rest = arm.read_pose()
                print("  REST captured:", fmt(rest))
            elif cmd == 'u':
                if sweep:
                    sweep.pop()
                    print(f"  undo — {len(sweep)} sweep points left")
                else:
                    print("  nothing to undo")
            elif cmd == 'f':
                if rest is None:
                    print("  set a REST pose first (press r)")
                elif len(sweep) < 2:
                    print(f"  need >= 2 sweep points (have {len(sweep)})")
                else:
                    break
            elif cmd == '':
                p = arm.read_pose()
                sweep.append(p)
                print(f"  sweep point {len(sweep)}: {fmt(p)}")
            else:
                print("  unknown command")
    finally:
        arm.close()

    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_trimmer.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Trimmer arm taught poses (4 joints: base/shoulder/elbow/"
                "wrist).\n")
        f.write(f"REST_POSE = {fmt(rest)}\n")
        f.write("SWEEP = [\n")
        for p in sweep:
            f.write(f"    {fmt(p)},\n")
        f.write("]\n")
    print(f"\nWrote {OUT_PATH}")
    print(f"  REST + {len(sweep)} sweep waypoints.")
    print("Run  python run_trimmer.py  to replay the sweep smoothly.")


if __name__ == '__main__':
    main()
