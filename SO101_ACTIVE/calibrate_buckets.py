"""Teach the two sorting buckets BY HAND — run once, terminal only.

Three poses: a CARRY pose (battery upright while travelling), then a drop
pose over the GREEN bucket (Ryobi batteries) and one over the RED bucket
(pink batteries). Buckets catch the drop, so no insert precision is needed —
just hold the limp gripper over each bucket's mouth.

Writes buckets_calibration.py (picked up automatically, enables the
auto-sort demo: ENTER = sort every battery, closest first).
Delete it to revert.
"""
import os
import time

from config import ARM_JOINT_IDS
from actuator import BaseActuator

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'buckets_calibration.py')


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    print("Torque released — arm and gripper move freely by hand.\n")

    input("1/3  CARRY: move the arm to a travel pose that keeps a held\n"
          "     battery UPRIGHT. Press Enter... ")
    carry = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    print(f"     CARRY = {carry}\n")

    input("2/3  GREEN bucket (Ryobi): hold the limp gripper over its mouth,\n"
          "     base facing it. Press Enter... ")
    green_pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    green_base = actuator.read_position()
    print(f"     GREEN = {green_pose}  base = {green_base}\n")

    input("3/3  RED bucket (pink): hold the limp gripper over its mouth,\n"
          "     base facing it. Press Enter... ")
    red_pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    red_base = actuator.read_position()
    print(f"     RED = {red_pose}  base = {red_base}\n")

    actuator.close()

    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_buckets.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Two-bucket colour-sorting demo. Delete to revert.\n")
        f.write(f"PLACE_CARRY = {carry}\n")
        f.write(f"BUCKET_GREEN = {{'base': {green_base}, 'pose': {green_pose}}}\n")
        f.write(f"BUCKET_RED = {{'base': {red_base}, 'pose': {red_pose}}}\n")
        f.write("USE_BUCKETS = True\n")
    print(f"Wrote {OUT_PATH}")
    print("Done. Run main.py and press ENTER — it sorts every battery, "
          "closest first: Ryobi -> green bucket, pink -> red bucket.")


if __name__ == '__main__':
    main()
