"""End-to-end autonomous architecture in non-actuating dry-run mode."""
from datetime import timedelta
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import detect_candidates
from safety import SafetySupervisor
from planner import propose_base_step
from actuator_disabled import DisabledBaseActuator, ActuationDisabled


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
    supervisor = SafetySupervisor()
    actuator = DisabledBaseActuator()
    pipeline, queue = build_pipeline()
    pipeline.start()
    simulated_raw = BASE_FORWARD_RAW
    latest_proposal = None
    window = 'SO101 autonomous architecture — ACTUATION DISABLED'
    try:
        while pipeline.isRunning():
            group = queue.tryGet()
            if group is None:
                ok, reason = supervisor.is_stable()
                if supervisor.stop_latched:
                    actuator.emergency_stop()
                key = cv2.waitKey(1) & 255
                if key in (27, ord('e')):
                    supervisor.latch_stop('operator software stop')
                    actuator.emergency_stop()
                if key == ord('q'):
                    break
                continue

            supervisor.observe_frame()
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0
            candidates = detect_candidates(image, depth)
            ok, reason = supervisor.observe_candidates(candidates)

            if len(candidates) == 1:
                target = candidates[0]
                x, y, w, h = target['bbox']
                pixel_error = target['u'] - CAMERA_WIDTH // 2
                colour = (0, 255, 0) if ok else (0, 165, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
                if ok:
                    latest_proposal = propose_base_step(simulated_raw, pixel_error)
                else:
                    latest_proposal = None
            else:
                latest_proposal = None
                actuator.emergency_stop()

            status = 'STABLE / COMMAND PROPOSAL' if ok else f'NOT PERMITTED: {reason}'
            cv2.putText(image, status, (8, 24), 0, 0.52, (0, 255, 0) if ok else (0, 0, 255), 2)
            if latest_proposal:
                proposed_raw, proposal_reason = latest_proposal
                cv2.putText(image, f'PROPOSED RAW {proposed_raw} — NO COMMAND SENT', (8, CAMERA_HEIGHT - 18), 0, 0.48, (0, 255, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('e')):
                supervisor.latch_stop('operator software stop')
                actuator.emergency_stop()
            if key == ord('q'):
                break
    finally:
        actuator.emergency_stop()
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass

if __name__ == '__main__':
    main()
