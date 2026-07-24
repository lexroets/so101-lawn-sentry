"""FULL system calibration — run after ANY camera adjustment.

Three parts, guided on screen:

PART A (automatic, ~20s): base step test. Put ONE object (the battery) on
  the desk in view and keep hands clear — the BASE will make four small
  rotations by itself and watch how the image shifts. This MEASURES:
    RAW_PER_PIXEL      (pixel -> rotation mapping for the new camera pose)
    BASE_DIRECTION_SIGN (turn direction, verified not guessed)
    CAMERA_LATENCY_S    (true image lag, for overshoot-free tracking)

PART B (by hand): multi-point reach teaching. Arm torque is OFF. Teach the
  gripper touching the object at 4-6 DIFFERENT distances — points closer
  than 15 mm in depth to an existing one are rejected (duplicate depths
  ruined earlier calibrations).

PART C (optional): gripper open/closed.

Writes: base_calibration.py, reach_calibration.py, gripper_calibration.py
        (all picked up automatically by main.py) and
        calibration_export.txt  <-- UPLOAD THIS TO CHAT to finish tuning.

Keys: ENTER = capture/advance · S = skip Part A · F = finish reach points
      G = finish reach + calibrate gripper · q = abort (nothing written)
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
BASE_PATH = os.path.join(_DIR, 'base_calibration.py')
REACH_PATH = os.path.join(_DIR, 'reach_calibration.py')
GRIP_PATH = os.path.join(_DIR, 'gripper_calibration.py')
EXPORT_PATH = os.path.join(_DIR, 'calibration_export.txt')

STEP_RAW = 120          # base step size for Part A
MIN_Z_SEP = 0.015       # reach points must differ in depth by this (m)


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


def write_outputs(base_result, rest, points, grip_open, grip_closed):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if base_result:
        with open(BASE_PATH, 'w') as f:
            f.write(f"# Measured by calibrate_full.py {stamp}\n")
            f.write(f"RAW_PER_PIXEL = {base_result['rpp']:.3f}\n")
            f.write(f"BASE_DIRECTION_SIGN = {base_result['sign']}\n")
            f.write(f"CAMERA_LATENCY_S = {base_result['lat']:.3f}\n")
    if points:
        points = sorted(points, key=lambda p: p[0])
        with open(REACH_PATH, 'w') as f:
            f.write(f"# Written by calibrate_full.py {stamp}\n")
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
            f.write(f"# Written by calibrate_full.py {stamp}\n")
            f.write(f"GRIPPER_OPEN_RAW = {grip_open}\n")
            f.write(f"GRIPPER_CLOSED_RAW = {grip_closed}\n")
    with open(EXPORT_PATH, 'w') as f:
        f.write(f"SO-101 FULL CALIBRATION EXPORT  {stamp}\n")
        f.write("(Upload this file to chat to finish tuning.)\n\n")
        if base_result:
            f.write("BASE (measured by step test):\n")
            f.write(f"  RAW_PER_PIXEL       : {base_result['rpp']:.3f}\n")
            f.write(f"  BASE_DIRECTION_SIGN : {base_result['sign']}\n")
            f.write(f"  CAMERA_LATENCY_S    : {base_result['lat']:.3f}\n")
            f.write(f"  samples             : {base_result['samples']}\n")
            f.write(f"  latency samples (s) : {base_result['lats']}\n\n")
        else:
            f.write("BASE: step test skipped/failed — config values unchanged\n\n")
        f.write(f"rest_pose : {rest}\n")
        f.write(f"reach points ({len(points)}):\n")
        for z, pose in sorted(points, key=lambda p: p[0]):
            f.write(f"  z={z:.3f}m -> {pose}\n")
        f.write(f"gripper open/closed : {grip_open} / {grip_closed}\n\n")
        f.write("config snapshot:\n")
        f.write(f"  GOAL_MOVE_SPEED={GOAL_MOVE_SPEED} ARM_MOVE_SPEED={ARM_MOVE_SPEED} "
                f"GRIPPER_SPEED={GRIPPER_SPEED}\n")
        f.write(f"  GAIN={CLOSED_LOOP_GAIN} DEADBAND={IMAGE_DEADBAND_PX}px FPS={CAMERA_FPS} "
                f"STAGE={REACH_STAGE_FRAC} DROP={DROP_POSE_FRAC}\n")


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    print("FULL calibration. Part A: base step test (base will move!).")
    print("Place ONE object on the desk in view, hands clear. S skips Part A.")

    # phases: A_DETECT A_WATCH A_POST  REST Z POSE  G_OPEN G_CLOSED
    phase = 'A_DETECT'
    a_active = False        # base torque enabled yet?
    a_trial = 0
    a_fail = 0
    a_samples = []          # (rpp, sign)
    lat_samples = []
    step_sign = 1
    u_hist = deque(maxlen=10)
    track_u = None
    u0 = p0 = cmd = 0
    t_cmd = 0.0
    moved_seen = False
    p_hist = deque(maxlen=3)
    state_t = time.monotonic()

    base_result = None
    rest = None
    points = []
    pending_z = None
    grip_open = None
    grip_closed = None
    z_hist = deque(maxlen=10)
    last_key_t = 0.0
    message = ''
    message_until = 0.0
    last_release = time.monotonic()
    window = 'SO101 FULL CALIBRATION'
    aborted = True

    def finish_a(success):
        nonlocal base_result, phase
        if success and a_samples:
            rpps = [s[0] for s in a_samples]
            signs = [s[1] for s in a_samples]
            sign = 1 if signs.count(1) >= signs.count(-1) else -1
            lat = (sorted(lat_samples)[len(lat_samples) // 2]
                   if lat_samples else CAMERA_LATENCY_S)
            lat = max(0.05, min(0.30, lat - 0.5 / CAMERA_FPS))
            base_result = {
                'rpp': sum(rpps) / len(rpps),
                'sign': sign,
                'lat': lat,
                'samples': [(round(r, 3), s) for r, s in a_samples],
                'lats': [round(x, 3) for x in lat_samples],
            }
            print(f"PART A done: RAW_PER_PIXEL={base_result['rpp']:.3f} "
                  f"SIGN={sign} LATENCY={lat:.3f}s")
        else:
            print("PART A skipped/failed — keeping current config values.")
        actuator.torque_off()
        phase = 'REST'

    try:
        while pipeline.isRunning():
            now = time.monotonic()
            if phase in ('REST', 'Z', 'POSE', 'G_OPEN', 'G_CLOSED') \
                    and now - last_release >= 2.0:
                actuator.torque_off()
                last_release = now
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                candidates = detect_candidates(image, depth)
                cand = None
                if candidates:
                    ref = track_u if track_u is not None else CAMERA_WIDTH // 2
                    cand = min(candidates, key=lambda c: abs(c['u'] - ref))
                    if track_u is not None and abs(cand['u'] - track_u) > 200:
                        cand = None
                if cand is not None:
                    track_u = cand['u']
                    u_hist.append(cand['u'])
                    z_hist.append(cand['z'])
                    x, y, w, h = cand['bbox']
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(image, f"z={cand['z']:.3f}m", (x, max(12, y - 6)),
                                0, 0.45, (0, 255, 255), 1)

                # ---------------- PART A: automatic base step test ----------
                if phase == 'A_DETECT':
                    prompt = (f'PART A {a_trial + 1}/4: hold still — waiting for '
                              f'steady object (S = skip)')
                    if len(u_hist) == 10 and max(u_hist) - min(u_hist) <= 4:
                        if not a_active:
                            actuator.enable_ids([BASE_SERVO_ID])
                            a_active = True
                            time.sleep(0.2)
                        u0 = sorted(u_hist)[5]
                        p0 = actuator.read_position()
                        cmd = STEP_RAW * step_sign
                        actuator.move_to(p0 + cmd)
                        t_cmd = now
                        moved_seen = False
                        u_hist.clear()
                        p_hist.clear()
                        phase, state_t = 'A_WATCH', now
                elif phase == 'A_WATCH':
                    prompt = 'PART A: base stepping...'
                    if cand is not None and not moved_seen and abs(cand['u'] - u0) > 4:
                        lat_samples.append(now - t_cmd)
                        moved_seen = True
                    try:
                        p_hist.append(actuator.read_position())
                    except Exception:
                        pass
                    settled = (len(p_hist) == 3
                               and max(p_hist) - min(p_hist) <= 2
                               and now - t_cmd > 0.4)
                    if settled:
                        u_hist.clear()
                        phase, state_t = 'A_POST', now
                    elif now - state_t > 4.0:
                        a_fail += 1
                        phase, state_t = 'A_DETECT', now
                elif phase == 'A_POST':
                    prompt = 'PART A: measuring shift...'
                    if len(u_hist) == 8 and max(u_hist) - min(u_hist) <= 5:
                        u1 = sorted(u_hist)[4]
                        du = u1 - u0
                        if abs(du) >= 20:
                            rpp = abs(cmd / du)
                            sign = -1 if du * cmd > 0 else 1
                            a_samples.append((rpp, sign))
                            print(f"  step {a_trial + 1}: cmd={cmd:+d} du={du:+d}px "
                                  f"-> rpp={rpp:.3f} sign={sign}")
                        else:
                            a_fail += 1
                        a_trial += 1
                        step_sign = -step_sign
                        u_hist.clear()
                        if a_trial >= 4 or a_fail >= 3:
                            finish_a(len(a_samples) >= 2)
                        else:
                            phase, state_t = 'A_DETECT', now
                    elif now - state_t > 4.0:
                        a_fail += 1
                        if a_fail >= 3:
                            finish_a(len(a_samples) >= 2)
                        else:
                            phase, state_t = 'A_DETECT', now

                # ---------------- PART B/C: hand teaching -------------------
                elif phase == 'REST':
                    prompt = 'PART B — arm limp. Put arm in REST pose. ENTER.'
                elif phase == 'Z':
                    prompt = (f'[{len(points)} pts] Object at a NEW distance '
                              f'(>=15mm from others), boxed. ENTER.'
                              + (' | F=finish G=+gripper' if len(points) >= 3 else
                                 f' (need {3 - len(points)} more)'))
                elif phase == 'POSE':
                    prompt = 'BY HAND: gripper ON the object. ENTER.'
                elif phase == 'G_OPEN':
                    prompt = 'GRIPPER fully OPEN by hand. ENTER.'
                elif phase == 'G_CLOSED':
                    prompt = 'GRIPPER fully CLOSED by hand. ENTER.'
                else:
                    prompt = phase

                cv2.putText(image, prompt, (8, 24), 0, 0.48, (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 48), 0, 0.5, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if now - last_key_t < 0.5:
                continue
            if key == ord('s') and phase.startswith('A_'):
                last_key_t = now
                finish_a(False)
            elif key in (ord('f'), ord('g')) and phase == 'Z' and len(points) >= 3:
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
                        message = 'No steady detection — reposition and retry'
                        message_until = now + 2.5
                        continue
                    z_val = float(sorted(z_hist)[len(z_hist) // 2])
                    if any(abs(z_val - p[0]) < MIN_Z_SEP for p in points):
                        message = (f'z={z_val:.3f} too close to an existing point '
                                   f'— move the object further/nearer')
                        message_until = now + 3.0
                        continue
                    pending_z = z_val
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

    if aborted or rest is None or len(points) < 3:
        print("Aborted or too few points (need >= 3) — nothing written.")
        return
    write_outputs(base_result, rest, points, grip_open, grip_closed)
    print(f"\nWrote {REACH_PATH}")
    if base_result:
        print(f"Wrote {BASE_PATH}")
    if grip_open is not None:
        print(f"Wrote {GRIP_PATH}")
    print(f"Wrote {EXPORT_PATH}  <-- upload this file to chat")
    print("Done. Run main.py.")


if __name__ == '__main__':
    main()
