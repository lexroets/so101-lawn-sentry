"""Gripper calibration — run ONCE, takes 30 seconds, terminal only.

Releases the gripper servo so you can move the fingers by hand, records the
fully-OPEN and fully-CLOSED positions, and writes gripper_calibration.py
(picked up automatically by main.py). Delete that file to revert.
"""
import os
import time

from config import GRIPPER_ID
from actuator import BaseActuator

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'gripper_calibration.py')


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    print("Torque released — the gripper fingers move freely now.\n")

    input("1/2  Move the gripper fingers FULLY OPEN by hand, then press Enter... ")
    open_raw = actuator.read_position(GRIPPER_ID)
    print(f"     open = {open_raw}\n")

    input("2/2  Now move the fingers FULLY CLOSED (touching), then press Enter... ")
    closed_raw = actuator.read_position(GRIPPER_ID)
    print(f"     closed = {closed_raw}\n")

    actuator.close()

    if abs(open_raw - closed_raw) < 100:
        print("WARNING: open and closed are less than 100 raw apart — that "
              "doesn't look right. Run this again and move the fingers fully.")

    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_gripper.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write(f"GRIPPER_OPEN_RAW = {open_raw}\n")
        f.write(f"GRIPPER_CLOSED_RAW = {closed_raw}\n")
    print(f"Wrote {OUT_PATH}")
    print("Done. Run main.py — the gripper now opens on lock and closes on the cap.")


if __name__ == '__main__':
    main()
