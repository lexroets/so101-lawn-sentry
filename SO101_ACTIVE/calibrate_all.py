"""MASTER CALIBRATION — one guided cycle for the whole sorting demo.

PART A (auto)   : base step test — pixel/rotation mapping, direction, latency.
                  ONE object on the desk, hands clear, the base moves itself.
PART B (auto)   : DETECTION tuning — put ONE Ryobi battery alone in view, it
                  samples its real size + colour; then ONE pink battery.
                  Size gates and both hue ranges are MEASURED, not guessed.
PART C (by hand): reach teaching — wrap the limp gripper around the TOP of
                  the battery at 3+ distances (>=15mm apart), then the
                  gripper fully open / fully closed.
PART D (by hand): bucket drop poses — hold the gripper HIGH over each
                  bucket's mouth (a hand's width above the rim at least).

Writes: base_calibration.py, detection_calibration.py, reach_calibration.py,
        gripper_calibration.py, buckets_calibration.py, calibration_export.txt
        (upload the export to chat for a final check).

Keys: ENTER = capture/advance · S = skip current part ·
      F = finish reach points (needs >= 3) · q = abort (nothing written)
"""
from collections import deque
from datetime import timedelta
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import _mask_depth_objects, _measure_depth
from actuator import BaseActuator

_DIR = os.path.dirname(os.path.abspath(__file__))
STEP_RAW = 120
MIN_Z_SEP = 0.015
B_SAMPLES = 40


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


def sample_object(image, depth):
    """Most central object on the desk, WITHOUT the size gates (used to
    MEASURE the battery, so it must not be pre-filtered)."""
    mask = _mask_depth_objects(depth)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = image.shape[:2]
    best = None
    for c in contours:
        a = cv2.contourArea(c)
        if a < 60 or a > 60000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x < 2 or y < 2 or x + w > w_img - 2 or y + h > h_img - 2:
            continue
        u, v = x + w // 2, y + h // 2
        z = _measure_depth(depth, u, v)
        if z is None:
            continue
        d = abs(u - w_img // 2)
        if best is None or d < best[0]:
            best = (d, (x, y, w, h), z)
    return None if best is None else (best[1], best[2])


def hue_range_from_hist(hist):
    """Dominant hue band: peak bin widened to 20%-of-peak, padded 5."""
    if hist.sum() < 500:
        return None
    peak = int(np.argmax(hist))
    thr = hist[peak] * 0.2
    lo = peak
    while lo > 0 and hist[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < 179 and hist[hi + 1] >= thr:
        hi += 1
    return (max(0, lo - 5), min(179, hi + 5))


def main():
    actuator = BaseActuator()
    actuator.torque_off()
    pipeline, queue = build_pipeline()
    pipeline.start()
    print("MASTER CALIBRATION. Part A: base step test (base WILL move).")
    print("ONE object on the desk, hands clear. S skips any part. q aborts.")

    phase = 'A_DETECT'
    # Part A state
    a_active = False
    a_trial = a_fail = 0
    a_samples, lat_samples = [], []
    step_sign = 1
    u_hist = deque(maxlen=10)
    track_u = None
    u0 = p0 = cmd = 0
    t_cmd = state_t = 0.0
    moved_seen = False
    p_hist = deque(maxlen=3)
    base_result = None
    # Part B state
    b_sizes = {'ryobi': [], 'pink': []}
    b_hues = {'ryobi': np.zeros(180, np.int64), 'pink': np.zeros(180, np.int64)}
    b_current = 'ryobi'
    b_count = 0
    det_result = None
    # Part C state
    rest = None
    points = []
    pending_z = None
    grip_open = grip_closed = None
    z_hist = deque(maxlen=10)
    # Part D state
    carry = green_pose = green_base = red_pose = red_base = None
    last_key_t = 0.0
    message, message_until = '', 0.0
    last_release = time.monotonic()
    window = 'SO101 MASTER CALIBRATION'
    aborted = True

    def finish_a(success):
        nonlocal base_result, phase
        if success and a_samples:
            rpps = [s[0] for s in a_samples]
            signs = [s[1] for s in a_samples]
            lat = (sorted(lat_samples)[len(lat_samples) // 2]
                   if lat_samples else CAMERA_LATENCY_S)
            base_result = {
                'rpp': sum(rpps) / len(rpps),
                'sign': 1 if signs.count(1) >= signs.count(-1) else -1,
                'lat': max(0.05, min(0.30, lat - 0.5 / CAMERA_FPS)),
                'samples': [(round(r, 3), s) for r, s in a_samples],
                'lats': [round(x, 3) for x in lat_samples],
            }
            print(f"PART A: rpp={base_result['rpp']:.3f} sign={base_result['sign']} "
                  f"lat={base_result['lat']:.3f}s")
        else:
            print("PART A skipped/failed — keeping current values.")
        actuator.torque_off()
        phase = 'B_WAIT'

    def finish_b():
        nonlocal det_result, phase
        sizes_all = b_sizes['ryobi'] + b_sizes['pink']
        if len(sizes_all) >= 15:
            mins = np.array([min(w, h) for w, h in sizes_all])
            maxs = np.array([max(w, h) for w, h in sizes_all])
            det_result = {
                'min_dim': (round(float(np.percentile(mins, 10)) * 0.7, 4),
                            round(float(np.percentile(mins, 90)) * 1.4, 4)),
                'max_dim': (round(float(np.percentile(maxs, 10)) * 0.7, 4),
                            round(float(np.percentile(maxs, 90)) * 1.4, 4)),
                'ryobi_hue': hue_range_from_hist(b_hues['ryobi']) or RYOBI_HUE,
                'pink_hue': hue_range_from_hist(b_hues['pink']) or PINK_HUE,
            }
            r, p = det_result['ryobi_hue'], det_result['pink_hue']
            if not (r[1] < p[0] - 5 or p[1] < r[0] - 5):
                print("WARNING: measured hue ranges overlap — the two "
                      "batteries were not distinguished. Pink falls back to "
                      "the standard magenta band (140,178).")
                det_result['pink_hue'] = (140, 178)
            print(f"PART B: min_dim={det_result['min_dim']} "
                  f"max_dim={det_result['max_dim']} "
                  f"ryobi_hue={det_result['ryobi_hue']} "
                  f"pink_hue={det_result['pink_hue']}")
        else:
            print("PART B skipped/insufficient — keeping current values.")
        phase = 'REST'

    try:
        while pipeline.isRunning():
            now = time.monotonic()
            if phase in ('REST', 'Z', 'POSE', 'G_OPEN', 'G_CLOSED',
                         'D_CARRY', 'D_GREEN', 'D_RED') \
                    and now - last_release >= 2.0:
                actuator.torque_off()
                last_release = now
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                samp = sample_object(image, depth)
                if samp is not None:
                    (x, y, w, h), z = samp
                    u = x + w // 2
                    track_u = u
                    u_hist.append(u)
                    z_hist.append(z)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(image, f"z={z:.3f} {w * z / FX_PIXELS * 1000:.0f}x"
                                       f"{h * z / FX_PIXELS * 1000:.0f}mm",
                                (x, max(12, y - 6)), 0, 0.42, (0, 255, 255), 1)

                # -------- PART A --------------------------------------------
                if phase == 'A_DETECT':
                    prompt = f'PART A {a_trial + 1}/4: hold still (S=skip)'
                    if len(u_hist) == 10 and max(u_hist) - min(u_hist) <= 4:
                        if not a_active:
                            actuator.enable_ids([BASE_SERVO_ID])
                            a_active = True
                            time.sleep(0.2)
                        u0 = sorted(u_hist)[5]
                        p0 = actuator.read_position()
                        cmd = STEP_RAW * step_sign
                        actuator.move_to(p0 + cmd)
                        t_cmd, moved_seen = now, False
                        u_hist.clear()
                        p_hist.clear()
                        phase, state_t = 'A_WATCH', now
                elif phase == 'A_WATCH':
                    prompt = 'PART A: base stepping...'
                    if samp and not moved_seen and abs(track_u - u0) > 4:
                        lat_samples.append(now - t_cmd)
                        moved_seen = True
                    try:
                        p_hist.append(actuator.read_position())
                    except Exception:
                        pass
                    if (len(p_hist) == 3 and max(p_hist) - min(p_hist) <= 2
                            and now - t_cmd > 0.4):
                        u_hist.clear()
                        phase, state_t = 'A_POST', now
                    elif now - state_t > 4.0:
                        a_fail += 1
                        phase, state_t = 'A_DETECT', now
                elif phase == 'A_POST':
                    prompt = 'PART A: measuring shift...'
                    if len(u_hist) == 8 and max(u_hist) - min(u_hist) <= 5:
                        du = sorted(u_hist)[4] - u0
                        if abs(du) >= 20:
                            a_samples.append((abs(cmd / du),
                                              -1 if du * cmd > 0 else 1))
                            print(f"  step: cmd={cmd:+d} du={du:+d}px")
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

                # -------- PART B --------------------------------------------
                elif phase == 'B_WAIT':
                    prompt = (f'PART B: ONLY the {b_current.upper()} battery in '
                              f'view, then ENTER (S=skip part)')
                elif phase == 'B_RUN':
                    prompt = f'PART B: sampling {b_current} {b_count}/{B_SAMPLES}'
                    if samp is not None:
                        (x, y, w, h), z = samp
                        w_m, h_m = w * z / FX_PIXELS, h * z / FX_PIXELS
                        # Sanity: only accept battery-plausible blobs, and
                        # sample hue from the INSET centre so the desk
                        # background can't dominate (that bug produced
                        # identical hue ranges for both batteries).
                        if not (0.008 <= min(w_m, h_m) <= 0.08
                                and 0.02 <= max(w_m, h_m) <= 0.20):
                            pass    # fragment/shadow — skip this frame
                        else:
                            b_sizes[b_current].append((w_m, h_m))
                            ix, iy = int(w * 0.25), int(h * 0.25)
                            roi = image[y + iy:y + h - iy, x + ix:x + w - ix]
                            if roi.size:
                                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                hue = hsv[:, :, 0][hsv[:, :, 1] >= COLOUR_S_MIN]
                                if hue.size:
                                    b_hues[b_current] += np.bincount(
                                        hue.ravel(), minlength=180)[:180]
                            b_count += 1
                        if b_count >= B_SAMPLES:
                            if b_current == 'ryobi':
                                b_current, b_count = 'pink', 0
                                phase = 'B_WAIT'
                                print("  Ryobi sampled. Now the PINK battery.")
                            else:
                                finish_b()

                # -------- PART C --------------------------------------------
                elif phase == 'REST':
                    prompt = 'PART C — arm limp. REST pose. ENTER.'
                elif phase == 'Z':
                    prompt = (f'[{len(points)} pts] Battery at a NEW distance, '
                              f'boxed. ENTER.'
                              + (' | F=finish' if len(points) >= 3 else
                                 f' (need {3 - len(points)} more)'))
                elif phase == 'POSE':
                    prompt = 'BY HAND: gripper around the TOP of the battery. ENTER.'
                elif phase == 'G_OPEN':
                    prompt = 'GRIPPER fully OPEN by hand. ENTER.'
                elif phase == 'G_CLOSED':
                    prompt = 'GRIPPER fully CLOSED by hand. ENTER.'

                # -------- PART D --------------------------------------------
                elif phase == 'D_CARRY':
                    prompt = 'PART D: CARRY pose (battery upright). ENTER.'
                elif phase == 'D_GREEN':
                    prompt = ('GREEN bucket: gripper HIGH over its mouth '
                              '(hand-width above rim!). ENTER.')
                elif phase == 'D_RED':
                    prompt = ('RED bucket: gripper HIGH over its mouth '
                              '(hand-width above rim!). ENTER.')
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
            if key == ord('s'):
                last_key_t = now
                if phase.startswith('A_'):
                    finish_a(False)
                elif phase.startswith('B_'):
                    finish_b()
            elif key == ord('f') and phase == 'Z' and len(points) >= 3:
                last_key_t = now
                phase = 'G_OPEN'
            elif key in (13, 10):
                last_key_t = now
                if phase == 'B_WAIT':
                    b_count = 0
                    phase = 'B_RUN'
                elif phase == 'REST':
                    rest = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    print(f"REST: {rest}")
                    phase = 'Z'
                elif phase == 'Z':
                    if len(z_hist) < 5:
                        message, message_until = 'No steady detection', now + 2.5
                        continue
                    z_val = float(sorted(z_hist)[len(z_hist) // 2])
                    if any(abs(z_val - p[0]) < MIN_Z_SEP for p in points):
                        message = f'z={z_val:.3f} too close to an existing point'
                        message_until = now + 3.0
                        continue
                    pending_z = z_val
                    print(f"Point {len(points) + 1}: z={pending_z:.3f}")
                    phase = 'POSE'
                elif phase == 'POSE':
                    pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    points.append((pending_z, pose))
                    print(f"Point {len(points)}: {pose}")
                    z_hist.clear()
                    phase = 'Z'
                elif phase == 'G_OPEN':
                    grip_open = actuator.read_position(GRIPPER_ID)
                    print(f"Gripper open: {grip_open}")
                    phase = 'G_CLOSED'
                elif phase == 'G_CLOSED':
                    grip_closed = actuator.read_position(GRIPPER_ID)
                    print(f"Gripper closed: {grip_closed}")
                    phase = 'D_CARRY'
                elif phase == 'D_CARRY':
                    carry = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    print(f"CARRY: {carry}")
                    phase = 'D_GREEN'
                elif phase == 'D_GREEN':
                    green_pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    green_base = actuator.read_position()
                    print(f"GREEN: {green_pose} base={green_base}")
                    phase = 'D_RED'
                elif phase == 'D_RED':
                    red_pose = {sid: actuator.read_position(sid) for sid in ARM_JOINT_IDS}
                    red_base = actuator.read_position()
                    print(f"RED: {red_pose} base={red_base}")
                    aborted = False
                    break
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        actuator.close()

    if aborted or rest is None or len(points) < 3 or red_pose is None:
        print("Aborted or incomplete — nothing written.")
        return

    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if base_result:
        with open(os.path.join(_DIR, 'base_calibration.py'), 'w') as f:
            f.write(f"# calibrate_all.py {stamp}\n")
            f.write(f"RAW_PER_PIXEL = {base_result['rpp']:.3f}\n")
            f.write(f"BASE_DIRECTION_SIGN = {base_result['sign']}\n")
            f.write(f"CAMERA_LATENCY_S = {base_result['lat']:.3f}\n")
    if det_result:
        with open(os.path.join(_DIR, 'detection_calibration.py'), 'w') as f:
            f.write(f"# calibrate_all.py {stamp} — MEASURED from your batteries\n")
            f.write(f"BATT_MIN_DIM_M = {det_result['min_dim']}\n")
            f.write(f"BATT_MAX_DIM_M = {det_result['max_dim']}\n")
            f.write(f"RYOBI_HUE = {det_result['ryobi_hue']}\n")
            f.write(f"PINK_HUE = {det_result['pink_hue']}\n")
    points = sorted(points, key=lambda p: p[0])
    with open(os.path.join(_DIR, 'reach_calibration.py'), 'w') as f:
        f.write(f"# calibrate_all.py {stamp} — TOP-grab poses\n")
        f.write(f"CAL_REST_POSE = {rest}\n")
        f.write(f"REACH_POINTS = {points!r}\n")
        f.write(f"REACH_NEAR_M = {points[0][0]:.3f}\n")
        f.write(f"REACH_POSE_NEAR = {points[0][1]}\n")
        f.write(f"REACH_FAR_M = {points[-1][0]:.3f}\n")
        f.write(f"REACH_POSE_FAR = {points[-1][1]}\n")
        f.write("REACH_ABSOLUTE = True\nUSE_IK = False\n")
        f.write("JOINT_TEST_ON_START = False\n")
        f.write("GRAB_POSE_FRAC = 1.0   # poses now taught at the TOP\n")
    with open(os.path.join(_DIR, 'gripper_calibration.py'), 'w') as f:
        f.write(f"# calibrate_all.py {stamp}\n")
        f.write(f"GRIPPER_OPEN_RAW = {grip_open}\n")
        f.write(f"GRIPPER_CLOSED_RAW = {grip_closed}\n")
    with open(os.path.join(_DIR, 'buckets_calibration.py'), 'w') as f:
        f.write(f"# calibrate_all.py {stamp}\n")
        f.write(f"PLACE_CARRY = {carry}\n")
        f.write(f"BUCKET_GREEN = {{'base': {green_base}, 'pose': {green_pose}}}\n")
        f.write(f"BUCKET_RED = {{'base': {red_base}, 'pose': {red_pose}}}\n")
        f.write("USE_BUCKETS = True\n")
    with open(os.path.join(_DIR, 'calibration_export.txt'), 'w') as f:
        f.write(f"SO-101 MASTER CALIBRATION EXPORT  {stamp}\n\n")
        if base_result:
            f.write(f"BASE: rpp={base_result['rpp']:.3f} sign={base_result['sign']} "
                    f"lat={base_result['lat']:.3f}\n  samples={base_result['samples']}"
                    f"\n  lats={base_result['lats']}\n")
        if det_result:
            f.write(f"DETECTION: min_dim={det_result['min_dim']} "
                    f"max_dim={det_result['max_dim']}\n"
                    f"  ryobi_hue={det_result['ryobi_hue']} "
                    f"pink_hue={det_result['pink_hue']}\n")
        f.write(f"REST: {rest}\nREACH ({len(points)} pts, TOP-grab):\n")
        for z, pose in points:
            f.write(f"  z={z:.3f} -> {pose}\n")
        f.write(f"GRIPPER: {grip_open}/{grip_closed}\n")
        f.write(f"CARRY: {carry}\nGREEN: base={green_base} {green_pose}\n")
        f.write(f"RED: base={red_base} {red_pose}\n")
    print("\nAll calibration files written. Upload calibration_export.txt "
          "to chat for a final check, then run main.py and press ENTER once.")


if __name__ == '__main__':
    main()
