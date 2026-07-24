"""Teach the SCAN pose — the arm posture that aims the camera at the lawn.

Torque is released so you can pose the arm by hand until the OAK-D looks
down/forward at the grass with a good working view (~0.3-0.8 m of ground in
frame). Press ENTER to capture joints 2/3/4 (shoulder, elbow, wrist). The base
is NOT saved — it is swept side to side during a scan.

Writes poop_scan_pose.py, which poop_config.py auto-loads.

    python calibrate_scan.py
"""
from trimmer_arm import TrimmerArm, TrimmerError
from poop_config import SHOULDER_ID, ELBOW_ID, WRIST_ID


def main():
    try:
        arm = TrimmerArm()
    except TrimmerError as exc:
        print(exc)
        return
    try:
        arm.torque_off()
        print("Torque OFF. Pose the arm by hand so the camera looks at the "
              "lawn (steady downward-forward view). Support it, then press "
              "ENTER to capture...")
        input()
        pose = {sid: arm.read_position(sid)
                for sid in (SHOULDER_ID, ELBOW_ID, WRIST_ID)}
        with open('poop_scan_pose.py', 'w') as f:
            f.write('"""Taught camera scan pose (joints 2/3/4). '
                    'Regenerate with calibrate_scan.py."""\n')
            f.write(f'SCAN_POSE = {pose!r}\n')
        print(f"Saved SCAN_POSE = {pose} -> poop_scan_pose.py")
        print("Run:  python poop_view.py   (check detection) "
              "or  python scan_poop.py  (full scan).")
    finally:
        arm.close()


if __name__ == '__main__':
    main()
