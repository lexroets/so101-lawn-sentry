"""FULL calibration — teach the arm BY HAND at as many distances as you like.

Torque stays OFF the whole time; move the limbs freely. The more points you
teach (3-5 at different distances is ideal), the more accurate the reach
distancing becomes: main.py interpolates between the two taught poses that
bracket the cap's measured depth.

Flow (ENTER captures each step, video window must be focused):
  1. REST pose                       -> ENTER
  2. Stand the cap somewhere, boxed  -> ENTER (captures depth)
  3. Hand-place gripper ON the cap   -> ENTER (captures pose)
  4. Move the cap to a NEW distance and repeat 2-3 as many times as you like
     F = finish reach points (needs >= 2)
     G = finish reach points, then calibrate the gripper too (recommended)
  5. Gripper (if G): fully OPEN -> ENTER, fully CLOSED -> ENTER

Writes:
  reach_calibration.py    - used automatically by main.py
  gripper_calibration.py  - used automatically by main.py (if G)
  calibration_export.txt  - human-readable summary: upload this file to chat
                            so the system can be finalised from your data.

Keys: ENTER = capture · F = finish · G = gripper then finish · q = abort
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

_DIR = os.path.dirname(os.path.abspath(__file__))
REACH_PATH = os.path.join(_DIR, 'reach_calibration.py')
GRIP_PATH = os.path.join(_DIR, 'gripper_calibration.py')
EXPORT_PATH = os.path.join(_DIR, 'calibration_export.txt')


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


def write_outputs(rest, points, grip_open, grip_closed):
    points = sorted(points, key=lambda p: p[0])
    with open(REACH_PATH, 'w') as f:
        f.write("# Written by calibrate.py " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("# Multi-point taught-by-hand reach. Delete to revert.\n")
        f.write(f"CAL_REST_POSE = {rest}\n")
        f.write(f"REACH_POINTS = {points!r}\n")
        f.write(f"REACH_NEAR_M = {points[0][0]:.3f}\n")
        f.write(f"REACH_POSE_NEAR = {points[0][1]}\n")
        f.write(f"REACH_FAR_M = {points[-1][0]:.3f}\n")
        f.write(f"REACH_POSE_FAR = {points[-1][1]}\n")
        f.write("REACH_ABSOLUTE = True\n")
        f.write("USE_IK = False\n")
        f.write("JOINT_TEST_ON_START = False\n")
    if grip_open is not None and grip_closed is not None:
        with open(GRIP_PATH, 'w') as f:
            f.write("# Written by calibrate.py " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
            f.write(f"GRIPPER_OPEN_RAW = {grip_open}\n")
            f.write(f"GRIPPER_CLOSED_RAW = {grip_closed}\n")
    with open(EXPORT_PATH, 'w') as f:
        f.write("SO-101 CALIBRATION EXPORT  "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("(Upload this file to chat to finalise the system.)\n\n")
        f.write(f"rest_pose            : {rest}\n")
        f.write(f"reach points ({len(points)}):\n")
        for z, pose in points:
            f.write(f"  z={z:.3f}m -> {pose}\n")
        f.write(f"gripper open/closed  : {grip_open} / {grip_closed}\n\n")
        f.write("config snapshot:\n")
        f.write(f"  GOAL_MOVE_SPEED={GOAL_MOVE_SPEED} ARM_MOVE_SPEED={ARM_MOVE_SPEED} "
                f"GRIPPER_SPEED={GRIPPER_SPEED}\n")
        f.write(f"  CLOSED_LOOP_GAIN={CLOSED_LOOP_GAIN} RAW_PER_PIXEL={RAW_PER_PIXEL} "
                f"DEADBAND={IMAGE_DEADBAND_PX}px FPS={CAMERA_FPS}\n")
        f.write(f"  GRIP_DELAY_S={GRIP_DELAY_S} ARM_ARRIVE_TOL_RAW={ARM_ARRIVE_TOL_RAW} "
                f"GRIP_STALL_RAW={GRIP_STALL_RAW}\n")


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    print("Calibration started. Torque RELEASED — limbs move by hand.")
    print("(Support the arm: it may sag when the servos release.)")

    z_hist = deque(maxlen=10)
    phase = 'REST'          # REST -> Z -> POSE -> (Z... | G_OPEN -> G_CLOSED)
    rest = None
    points = []             # [(z, {sid: raw}), ...]
    pending_z = None
    grip_open = None
    grip_closed = None
    last_key_t = 0.0
    last_release = time.monotonic()
    message = ''
    message_until = 0.0
    window = 'SO101 CALIBRATION (ENTER=capture, F=finish, G=+gripper, q=abort)'
    aborted = True
    try:
        while pipeline.isRunning():
            now = time.monotonic()
            if now - last_release >= 2.0:
                actuator.torque_off()   # some firmware re-enables torque
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
                prompts = {
                    'REST': 'REST: arm in rest pose. ENTER.',
                    'Z': f'[{len(points)} pts] Cap at a NEW distance, boxed. '
                         f'ENTER.' + (' | F=finish G=+gripper' if len(points) >= 2 else ''),
                    'POSE': 'BY HAND: gripper ON the cap. ENTER.',
                    'G_OPEN': 'GRIPPER fully OPEN by hand. ENTER.',
                    'G_CLOSED': 'GRIPPER fully CLOSED by hand. ENTER.',
                }
                cv2.putText(image, prompts[phase], (8, 24), 0, 0.5, (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 48), 0, 0.5, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if now - last_key_t < 0.5:
                continue
            if key in (ord('f'), ord('g')) and phase == 'Z' and len(points) >= 2:
                last_key_t = now
                if key == ord('g'):
                    phase = 'G_OPEN'
                else:
                    aborted = False
                    break
            elif key in (13, 10):
                last_key_t = now
                if phase == 'REST':
                    rest = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    print(f"REST: {rest}")
                    phase = 'Z'
                elif phase == 'Z':
                    if len(z_hist) < 5:
                        message = 'No steady cap detection — reposition and retry'
                        message_until = now + 2.5
                        continue
                    pending_z = float(sorted(z_hist)[len(z_hist) // 2])
                    print(f"Point {len(points) + 1}: z = {pending_z:.3f} m")
                    phase = 'POSE'
                elif phase == 'POSE':
                    pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    points.append((pending_z, pose))
                    print(f"Point {len(points)}: pose = {pose}")
                    z_hist.clear()
                    phase = 'Z'
                elif phase == 'G_OPEN':
                    grip_open = actuator.read_position(GRIPPER_ID)
                    print(f"Gripper open: {grip_open}")
                    phase = 'G_CLOSED'
                elif phase == 'G_CLOSED':
                    grip_closed = actuator.read_position(GRIPPER_ID)
                    print(f"Gripper closed: {grip_closed}")
                    aborted = False
                    break
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        actuator.close()

    if aborted or rest is None or len(points) < 2:
        print("Aborted — nothing written.")
        return
    write_outputs(rest, points, grip_open, grip_closed)
    print(f"\nWrote {REACH_PATH}")
    if grip_open is not None:
        print(f"Wrote {GRIP_PATH}")
    print(f"Wrote {EXPORT_PATH}  <-- upload this file to chat")
    print("Done. Run main.py.")


if __name__ == '__main__':
    main()
