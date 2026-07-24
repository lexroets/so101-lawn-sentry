"""Teach the sleeve-insertion sequence BY HAND — run once, terminal only.

The trick for perfect alignment: you place the battery INTO the sleeve
first, then wrap the limp gripper around it — that captured pose IS the
perfectly-aligned insert position. Three poses total:

  1. INSERT : battery standing in the sleeve, limp gripper wrapped around
              it as if it had just placed it (base facing the sleeve!)
  2. HOVER  : arm lifted straight up by hand until the battery would be
              clear of the sleeve slot
  3. CARRY  : a comfortable travel pose that keeps the battery UPRIGHT

Playback in main.py: grab -> CARRY (upright) -> base swings to the sleeve ->
HOVER above the slot -> SLOW descent to INSERT -> settle -> open -> retreat
up -> rest. The battery is placed, never dropped.

Writes place_calibration.py (picked up automatically). Delete it to revert
to the click-a-box drop behaviour.
"""
import os
import time

from config import ARM_JOINT_IDS
from actuator import BaseActuator

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'place_calibration.py')


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    print("Torque released — arm and gripper move freely by hand.\n")

    input("1/3  Stand the battery IN the sleeve. Move the arm by hand so the\n"
          "     gripper is wrapped around it, perfectly as if it had just\n"
          "     inserted it (base facing the sleeve). Then press Enter... ")
    insert = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    base_raw = actuator.read_position()
    print(f"     INSERT = {insert}  base = {base_raw}\n")

    input("2/3  Lift the arm straight UP by hand until the battery would be\n"
          "     fully clear of the sleeve slot. Press Enter... ")
    hover = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    base_hover = actuator.read_position()
    print(f"     HOVER = {hover}\n")
    if abs(base_hover - base_raw) > 30:
        print(f"     WARNING: the base rotated {abs(base_hover - base_raw)} raw "
              f"between insert and hover — keep it still, or redo this.\n")

    input("3/3  Move the arm to a comfortable CARRY pose that keeps the\n"
          "     battery UPRIGHT while travelling. Press Enter... ")
    carry = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    print(f"     CARRY = {carry}\n")

    actuator.close()

    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_place.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Taught sleeve-insertion sequence. Delete to revert to\n")
        f.write("# the click-a-box drop behaviour.\n")
        f.write(f"PLACE_INSERT = {insert}\n")
        f.write(f"PLACE_HOVER = {hover}\n")
        f.write(f"PLACE_CARRY = {carry}\n")
        f.write(f"PLACE_BASE_RAW = {base_raw}\n")
    print(f"Wrote {OUT_PATH}")
    print("Done. Run main.py — after each grab, the battery will be placed "
          "into the sleeve automatically (no box selection needed).")


if __name__ == '__main__':
    main()
