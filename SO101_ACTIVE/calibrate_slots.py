"""Teach the 5-slot battery holster BY HAND — run once, terminal only.

Same trick as before, per slot: stand a battery IN the slot, wrap the limp
gripper around it (that's the perfectly aligned insert pose), then lift the
arm straight up until the battery would clear the slot (hover pose). The
base angle to each slot is captured automatically.

Playback in main.py fills the LOWEST-NUMBERED EMPTY slot on each grab and
remembers which slots are full (keys 1-5 toggle a slot manually, 0 marks
all empty — e.g. after you empty the holster).

Writes slots_calibration.py (picked up automatically, enables slot mode).
Delete it to revert to click-to-place.
"""
import os
import time

from config import ARM_JOINT_IDS
from actuator import BaseActuator

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'slots_calibration.py')
MAX_SLOTS = 5


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    print("Torque released — arm and gripper move freely by hand.\n")

    input("CARRY: move the arm to a comfortable travel pose that keeps a\n"
          "held battery UPRIGHT. Press Enter... ")
    carry = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
    print(f"  CARRY = {carry}\n")

    slots = []
    for i in range(MAX_SLOTS):
        ans = input(f"SLOT {i + 1}: press Enter to teach it, or type d + Enter "
                    f"to finish with {len(slots)} slots... ").strip().lower()
        if ans == 'd':
            break
        input(f"  Stand a battery IN slot {i + 1}, wrap the limp gripper "
              f"around it (base facing the slot). Enter... ")
        insert = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
        base_raw = actuator.read_position()
        print(f"    insert = {insert}  base = {base_raw}")
        input(f"  Lift the arm straight UP until the battery would clear "
              f"slot {i + 1}. Enter... ")
        hover = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
        if abs(actuator.read_position() - base_raw) > 30:
            print("    WARNING: base rotated between insert and hover — "
                  "keep it still or re-teach this slot.")
        print(f"    hover  = {hover}\n")
        slots.append({'base': base_raw, 'hover': hover, 'insert': insert})

    actuator.close()

    if len(slots) < 1:
        print("No slots taught — nothing written.")
        return

    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_slots.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Taught holster slots. Delete to revert to click-to-place.\n")
        f.write(f"PLACE_CARRY = {carry}\n")
        f.write(f"SLOTS = {slots!r}\n")
        f.write("USE_SLOTS = True\n")
    print(f"\nWrote {OUT_PATH} with {len(slots)} slots.")
    print("Done. Run main.py — each grab fills the next empty slot "
          "(keys 1-5 toggle, 0 = all empty).")


if __name__ == '__main__':
    main()
