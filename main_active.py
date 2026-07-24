"""End-to-end autonomous base centring — ACTUATION ENABLED.

Same pipeline, detector, safety supervisor and planner as the dry run, but wired
to the real base servo. The base only rotates in small bounded steps, and only
while the safety supervisor reports a single, stable, verified target.

Controls (focus the OpenCV window):
    E or Esc : latch software emergency stop (cuts torque, needs restart)
    q        : quit cleanly

Keep a hand near the physical power switch the first few runs.
"""
from datetime import timedelta
import time

import cv2
import depthai as dai
import numpy as np

from config import *
from detector import detect_candidates
from safety import SafetySupervisor
from planner import propose_base_step
from actuator import BaseActuator, ActuatorError


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


def _handle_keys(supervisor, actuator):
    """Return True if the operator asked to quit."""
    key = cv2.waitKey(1) & 255
    if key in (27, ord('e')):
        supervisor.latch_stop('operator software stop')
        actuator.emergency_stop()
    if key == ord('q'):
        return True
    return False


def main():
    supervisor = SafetySupervisor()
    actuator = BaseActuator()          # opens the bus, pings the servo
    pipeline, queue = build_pipeline()
    pipeline.start()

    # Short countdown so the operator is ready before torque comes on.
    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    actuator.enable()
    print("Torque ENABLED. Base will centre on a single stable pen-cap target.")

    latest_proposal = None
    window = 'SO101 autonomous base centring — ACTUATION ENABLED'
    try:
        while pipeline.isRunning():
            group = queue.tryGet()
            if group is None:
                # Keep timeouts alive; camera timeout latches a hard stop.
                supervisor.is_stable()
                if supervisor.stop_latched:
                    actuator.emergency_stop()
                if _handle_keys(supervisor, actuator):
                    break
                continue

            supervisor.observe_frame()
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0
            candidates = detect_candidates(image, depth)
            ok, reason = supervisor.observe_candidates(candidates)

            present_raw = None
            if len(candidates) == 1:
                target = candidates[0]
                x, y, w, h = target['bbox']
                pixel_error = target['u'] - CAMERA_WIDTH // 2
                colour = (0, 255, 0) if ok else (0, 165, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)

                if ok and not supervisor.stop_latched:
                    present_raw = actuator.read_position()
                    try:
                        proposed_raw, proposal_reason = propose_base_step(
                            present_raw, pixel_error * BASE_DIRECTION_SIGN
                        )
                    except ValueError:
                        # At a verified raw limit — hold position rather than crash.
                        latest_proposal = None
                    else:
                        sent_raw = actuator.move_one_bounded_step(proposed_raw)
                        latest_proposal = (sent_raw, proposal_reason)
                else:
                    latest_proposal = None
            else:
                # Target lost / ambiguous: stop commanding, hold position.
                latest_proposal = None

            status = 'MOVING / CENTRING' if ok else f'HOLD: {reason}'
            cv2.putText(image, status, (8, 24), 0, 0.52,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
            if present_raw is not None:
                cv2.putText(image, f'PRESENT RAW {present_raw}', (8, 46), 0, 0.48, (255, 255, 0), 2)
            if latest_proposal:
                sent_raw, _ = latest_proposal
                cv2.putText(image, f'COMMANDED RAW {sent_raw}', (8, CAMERA_HEIGHT - 18),
                            0, 0.48, (0, 255, 255), 2)
            cv2.imshow(window, image)

            if _handle_keys(supervisor, actuator):
                break
    finally:
        actuator.close()      # latches stop, cuts torque, closes the port
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main()
