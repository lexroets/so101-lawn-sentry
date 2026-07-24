"""Teach the image-heading -> wrist-roll mapping for lying batteries.

Run once. Arm torque is released so you move the wrist by hand. Lay a
battery DOWN and, at 2-4 DIFFERENT headings on the desk:

  1. The screen shows the battery's detected axis angle (a green line along
     it). Make sure the line lies along the battery.
  2. BY HAND, rotate ONLY the wrist roll so the open fingers would straddle
     the battery across its body (perpendicular to the green line). Leave the
     other joints wherever — only the roll is recorded.
  3. Press ENTER to capture (heading, roll_raw).

Use headings that span a wide range (e.g. near-horizontal, ~45 deg,
near-vertical) so the fit is well conditioned.

Writes roll_calibration.py, picked up automatically by roll_align.py.
Delete it to disable auto-roll.

Keys: ENTER capture · F finish (>=2 points) · q abort
"""
from datetime import timedelta
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import detect_candidates
from actuator import BaseActuator
import roll_align

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'roll_calibration.py')


def build_pipeline():
    pipeline = dai.Pipeline()
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rgb_out = rgb.requestOutput(size=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                fps=CAMERA_FPS, enableUndistortion=True)
    left_out = left.requestOutput(size=(640, 400),
                                  type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    right_out = right.requestOutput(size=(640, 400),
                                    type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
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
    print(__doc__)
    print("Torque released — move the wrist by hand.")

    samples = []               # (heading_deg, roll_raw)
    last_key_t = 0.0
    last_release = time.monotonic()
    last_deg = None
    message, message_until = '', 0.0
    window = 'SO-101 ROLL CALIBRATION (ENTER capture, F finish, q abort)'
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
                cands = [c for c in detect_candidates(image, depth)
                         if c['kind'] == 'target']
                last_deg = None
                if cands:
                    c = min(cands, key=lambda c: c['z'])   # nearest
                    res = roll_align.battery_axis_deg(depth, c['bbox'])
                    x, y, w, h = c['bbox']
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 200, 255), 2)
                    if res is not None:
                        deg, elong = res
                        cx, cy = x + w // 2, y + h // 2
                        L = max(w, h)
                        import math
                        dx = int(math.cos(math.radians(deg)) * L * 0.5)
                        dy = int(-math.sin(math.radians(deg)) * L * 0.5)
                        good = elong >= 1.8
                        cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy),
                                 (0, 255, 0) if good else (0, 0, 255), 2)
                        cv2.putText(image, f'{deg:5.1f} deg  elong {elong:.1f}'
                                    + ('' if good else '  (too round — rotate it)'),
                                    (x, max(14, y - 8)), 0, 0.5,
                                    (0, 255, 0) if good else (0, 0, 255), 2)
                        if good:
                            last_deg = deg
                cv2.putText(image, f'[{len(samples)} pts]  '
                            'rotate wrist to straddle it, ENTER'
                            + ('  |  F finish' if len(samples) >= 2 else
                               f' (need {2 - len(samples)} more)'),
                            (8, 24), 0, 0.5, (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 48), 0, 0.55, (0, 0, 255), 2)
                cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if now - last_key_t < 0.4:
                continue
            if key == ord('f') and len(samples) >= 2:
                aborted = False
                break
            if key in (13, 10):
                last_key_t = now
                if last_deg is None:
                    message = 'No clear battery axis — rotate it more side-on'
                    message_until = now + 2.5
                    continue
                roll_raw = actuator.read_position(ROLL_ID)
                samples.append((float(last_deg), int(roll_raw)))
                print(f"Point {len(samples)}: heading={last_deg:.1f} deg  "
                      f"roll_raw={roll_raw}")
                message = f'captured {last_deg:.0f} deg -> roll {roll_raw}'
                message_until = now + 2.0
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        actuator.close()

    if aborted or len(samples) < 2:
        print("Aborted or too few points — nothing written.")
        return
    fitres = roll_align.fit(samples)
    if fitres is None:
        print("Fit failed — nothing written.")
        return
    lo, hi = 0, 4095
    with open(OUT_PATH, 'w') as f:
        f.write("# Written by calibrate_roll.py "
                + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write(f"# {len(samples)} samples: {samples}\n")
        f.write(f"ROLL_REF_DEG = {fitres['ROLL_REF_DEG']:.3f}\n")
        f.write(f"ROLL_REF_RAW = {fitres['ROLL_REF_RAW']}\n")
        f.write(f"ROLL_RAW_PER_DEG = {fitres['ROLL_RAW_PER_DEG']:.4f}\n")
        f.write(f"ROLL_MIN_RAW = {lo}\n")
        f.write(f"ROLL_MAX_RAW = {hi}\n")
    print(f"\nWrote {OUT_PATH}")
    print(f"  {abs(fitres['ROLL_RAW_PER_DEG']):.2f} raw/deg  "
          f"(a full servo turn = 4096 raw = 360 deg -> ~11.4 ideal; "
          "if far off, re-check your captures)")
    print("Run main_ml.py — lying grabs now auto-align the wrist.")


if __name__ == '__main__':
    main()
