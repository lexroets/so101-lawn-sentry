"""Teach VERTICAL top-down grabs for LYING-DOWN batteries — run once.

Arm torque stays OFF; move the limbs by hand. Lay a battery DOWN on the
desk and, at each of 2-4 different distances:

  1. ENTER captures the battery's depth (it must be boxed on screen)
  2. BY HAND, bring the gripper straight DOWN over the battery from above —
     and ROTATE THE WRIST ROLL by hand so the open fingers straddle the
     battery's body crossways (across its 23 mm diameter, NOT along its
     length). ENTER captures the pose — including the roll, which is now a
     controlled joint.

At runtime the system detects a lying battery automatically (long + low
silhouette) and uses these taught poses instead of the upright side-grab.

Writes lying_calibration.py (picked up automatically).
Delete it to disable lying-battery grabs.

Keys: ENTER = capture · F = finish (needs >= 2 points) · q = abort
"""
from collections import deque
from datetime import timedelta
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import detect_candidates
from actuator import BaseActuator

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'lying_calibration.py')
MIN_Z_SEP = 0.015


def build_pipeline():
    pipeline = dai.Pipeline()
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rgb_out = rgb.requestOutput(size=(CAMERA_WIDTH, CAMERA_HEIGHT), fps=CAMERA_FPS, enableUndistortion=True)
    left_out = left.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    right_out = right.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    rgb_out.link(stereo.inputAlignTo)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=50))
    sync.setSyncAttempts(-1)
    rgb_out.link(sync.inputs['rgb'])
    stereo.depth.link(sync.inputs['depth'])
    return pipeline, sync.out.createOutputQueue(maxSize=1, blocking=False)


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    print("LYING-GRAB calibration. Torque released — limbs move by hand.")
    print("Lay a battery DOWN; teach a vertical top-down grab at 2-4 distances.")

    z_hist = deque(maxlen=10)
    phase = 'Z'            # Z -> POSE -> Z ...
    points = []
    pending_z = None
    last_key_t = 0.0
    last_release = time.monotonic()
    message, message_until = '', 0.0
    window = 'SO101 LYING-GRAB CALIBRATION (ENTER=capture, F=finish, q=abort)'
    aborted = True
    try:
        while pipeline.isRunning():
            now = time.monotonic()
            if now - last_release >= 2.0:
                actuator.torque_off()
                last_release = now
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                candidates = detect_candidates(image, depth)
                if candidates:
                    c = candidates[0]
                    z_hist.append(c['z'])
                    x, y, w, h = c['bbox']
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(image, f"z={c['z']:.3f}m", (x, max(12, y - 6)),
                                0, 0.45, (0, 255, 255), 1)
                if phase == 'Z':
                    prompt = (f'[{len(points)} pts] Battery LYING at a NEW '
                              f'distance, boxed. ENTER.'
                              + (' | F=finish' if len(points) >= 2 else
                                 f' (need {2 - len(points)} more)'))
                else:
                    prompt = ('BY HAND: gripper straight DOWN over it, roll '
                              'rotated so fingers straddle it crossways. ENTER.')
                cv2.putText(image, prompt, (8, 24), 0, 0.48, (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 48), 0, 0.5, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if now - last_key_t < 0.5:
                continue
            if key == ord('f') and phase == 'Z' and len(points) >= 2:
                aborted = False
                break
            if key in (13, 10):
                last_key_t = now
                if phase == 'Z':
                    if len(z_hist) < 5:
                        message, message_until = 'No steady detection', now + 2.5
                        continue
                    z_val = float(sorted(z_hist)[len(z_hist) // 2])
                    if any(abs(z_val - p[0]) < MIN_Z_SEP for p in points):
                        message = f'z={z_val:.3f} too close to an existing point'
                        message_until = now + 3.0
                        continue
                    pending_z = z_val
                    print(f"Point {len(points) + 1}: z = {pending_z:.3f} m")
                    phase = 'POSE'
                else:
                    pose = {sid: actuator.read_position(sid)
                            for sid in ARM_JOINT_IDS}
                    points.append((pending_z, pose))
                    print(f"Point {len(points)}: pose = {pose}")
                    z_hist.clear()
                    phase = 'Z'
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        actuator.close()

    if aborted or len(points) < 2:
        print("Aborted or too few points (need >= 2) — nothing written.")
        return
    points = sorted(points, key=lambda p: p[0])
    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_lying.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Vertical top-down grab poses for LYING batteries\n")
        f.write("# (includes the wrist roll). Delete to disable.\n")
        f.write(f"LYING_POINTS = {points!r}\n")
    print(f"\nWrote {OUT_PATH} with {len(points)} points.")
    print("Done. Run main.py — lying batteries are detected automatically "
          "and grabbed top-down.")


if __name__ == '__main__':
    main()
