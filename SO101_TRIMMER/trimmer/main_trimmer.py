"""Autonomous grass trimming — SO-101 trimmer arm.

Same framework as the battery sorter: detect a target, centre the base on it,
reach, act. Here the target is an uncut grass patch and the action is a LEVEL
cut.

Phase machine per patch:
    IDLE   find patches (nearest first); ENTER starts auto-trimming
    CENTRE PD-track the base until the patch sits at image centre (the set point)
    LOWER  drop the trimmer to the taught cut height for that depth (level)
    TRIM   sweep ONLY the base +/- about the set point — because the deck is
           level and the base axis is vertical, the tip stays at constant
           height, so the cut is level with no kinematics
    LIFT   raise to REST, return base forward, look for the next patch

Keys:  ENTER start/stop auto · SPACE trim the highlighted patch once ·
       e E-STOP · q quit
"""

# --- put the project's folders on sys.path (see paths.py) --------------------
import sys as _sys, pathlib as _pl
_d = _pl.Path(__file__).resolve().parent
while not (_d / 'paths.py').is_file() and _d != _d.parent:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
import paths as _paths
_paths.bootstrap()
# -----------------------------------------------------------------------------
import time
import cv2
import numpy as np

from trimmer_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, BASE_FORWARD_RAW, RAW_PER_PIXEL,
    BASE_DIRECTION_SIGN, CAMERA_LATENCY_S,
    BASE_ID, SHOULDER_ID, ELBOW_ID, WRIST_ID,
    MOVE_SPEED, MOVE_ACC, ARRIVE_TOL_RAW, SETTLE_S,
    CENTRE_GAIN, TRACK_ERR_SMOOTH, TRACK_MIN_CMD_INTERVAL_S,
    GOAL_UPDATE_MIN_RAW, CENTRE_DEADBAND_PX, CENTRE_STABLE_FRAMES,
    TRIM_SWEEP_HALF_RAW, TRIM_PASSES, TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC,
    REST_POSE, CUT_POINTS,
)
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline, detect_patches

ARM_JOINTS = (SHOULDER_ID, ELBOW_ID, WRIST_ID)   # base handled separately
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def cut_arm_pose(z):
    """Interpolate the taught level-cut ARM pose (joints 2/3/4 only) for a
    patch at depth z. Base is excluded — it stays centred on the patch."""
    pts = sorted(CUT_POINTS, key=lambda p: p[0])
    if z <= pts[0][0]:
        lo = hi = pts[0]
        t = 0.0
    elif z >= pts[-1][0]:
        lo = hi = pts[-1]
        t = 0.0
    else:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= z <= b[0]:
                lo, hi, t = a, b, (z - a[0]) / max(1e-6, b[0] - a[0])
                break
    return {sid: int(round(lo[1][sid] + t * (hi[1][sid] - lo[1][sid])))
            for sid in ARM_JOINTS if sid in lo[1] and sid in hi[1]}


def main():
    if REST_POSE is None or not CUT_POINTS:
        print("Not calibrated. Run  python calibrate_trim.py  first "
              "(need REST_POSE + CUT_POINTS).")
        return

    arm = TrimmerArm()
    pipeline, queue = build_pipeline()
    pipeline.start()
    centre_x = CAMERA_WIDTH // 2

    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    arm.enable()
    arm.set_profile(MOVE_SPEED, MOVE_ACC)
    arm.move_pose(REST_POSE)
    arm.wait_arrived(REST_POSE)
    arm.move_to(BASE_ID, BASE_FORWARD_RAW)
    print("READY. ENTER = auto-trim all patches. SPACE = trim highlighted. q = quit.")

    phase = 'IDLE'
    phase_start = time.monotonic()
    auto = False
    target = None
    base_centre = None
    centre_ok = 0
    trim_seq = []            # queued base goals for the TRIM sweep
    trim_goal = None         # the base goal currently being executed
    trk = {'es': None, 'cmd_t': 0.0}
    pos_hist = []            # (t, base_raw) for latency compensation
    last_enter = 0.0
    window = 'SO-101 Autonomous Trimmer'

    def log(m):
        print(f"[{time.monotonic():7.2f}] {m}")

    def base_at(t_wanted):
        for t, raw in reversed(pos_hist):
            if t <= t_wanted:
                return raw
        return pos_hist[0][1] if pos_hist else None

    def track_reset():
        trk['es'], trk['cmd_t'] = None, 0.0

    def track_goal(present_raw, err, now):
        es = float(err) if trk['es'] is None else (
            TRACK_ERR_SMOOTH * trk['es'] + (1.0 - TRACK_ERR_SMOOTH) * err)
        trk['es'] = es
        if abs(es) <= CENTRE_DEADBAND_PX:
            return None
        if now - trk['cmd_t'] < TRACK_MIN_CMD_INTERVAL_S:
            return None
        trk['cmd_t'] = now
        step = BASE_DIRECTION_SIGN * CENTRE_GAIN * es * RAW_PER_PIXEL
        return present_raw + step

    def go_rest_and_home():
        arm.set_profile(MOVE_SPEED, MOVE_ACC)
        arm.move_pose(REST_POSE)
        arm.wait_arrived(REST_POSE, tol=ARRIVE_TOL_RAW * 3)
        arm.move_to(BASE_ID, BASE_FORWARD_RAW)

    try:
        while pipeline.isRunning():
            now = time.monotonic()
            group = queue.tryGet()
            if group is None:
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0

            try:
                base_raw = arm.read_position(BASE_ID)
                pos_hist.append((now, base_raw))
                if len(pos_hist) > 40:
                    pos_hist.pop(0)
            except TrimmerError:
                base_raw = base_centre or BASE_FORWARD_RAW

            stats = {}
            patches = (detect_patches(image, depth, stats)
                       if phase in ('IDLE', 'CENTRE') else [])
            status = phase

            # keep a sticky highlight in IDLE
            if phase == 'IDLE':
                target = patches[0] if patches else None

            # ---- phase machine ----
            if phase == 'IDLE':
                status = ('AUTO — seeking grass' if auto else
                          'ENTER auto-trim · SPACE trim one')
                if auto and target is not None:
                    base_centre = None
                    centre_ok = 0
                    track_reset()
                    phase, phase_start = 'CENTRE', now
                    log(f"CENTRE on patch z={target['z']:.3f} "
                        f"area={target['area']:.0f}")

            elif phase == 'CENTRE':
                # re-associate to the patch nearest the previous one
                cur = None
                if patches:
                    cur = min(patches, key=lambda p: abs(p['u'] - centre_x)) \
                        if target is None else min(
                            patches, key=lambda p: (p['u'] - target['u']) ** 2
                            + (p['v'] - target['v']) ** 2)
                    target = cur
                if cur is None:
                    if now - phase_start > 2.0:
                        log("patch lost — back to IDLE")
                        phase, phase_start = 'IDLE', now
                else:
                    err = cur['u'] - centre_x
                    present = base_at(now - CAMERA_LATENCY_S) or base_raw
                    if abs(err) <= CENTRE_DEADBAND_PX:
                        centre_ok += 1
                        if centre_ok >= CENTRE_STABLE_FRAMES:
                            base_centre = base_raw
                            pose = cut_arm_pose(cur['z'])
                            arm.set_profile(MOVE_SPEED, MOVE_ACC)
                            arm.move_pose(pose)
                            phase, phase_start = 'LOWER', now
                            log(f"LOWER to level cut, z={cur['z']:.3f} "
                                f"pose={pose}")
                    else:
                        centre_ok = 0
                        g = track_goal(present, err, now)
                        if g is not None:
                            arm.move_to(BASE_ID, g)
                    status = f'CENTRING err={err:+d}px'

            elif phase == 'LOWER':
                status = 'LOWERING trimmer'
                cut_pose = cut_arm_pose(target['z'])
                # Once the arm reaches the cut height (or a timeout), build the
                # base-only sweep about the centred position and start it.
                if arm.arrived(cut_pose, tol=ARRIVE_TOL_RAW * 2) \
                        or now - phase_start > 3.0:
                    b = base_centre
                    seq = []
                    for _ in range(TRIM_PASSES):
                        seq += [b - TRIM_SWEEP_HALF_RAW,
                                b + TRIM_SWEEP_HALF_RAW]
                    seq.append(b)                 # finish centred
                    arm.set_profile(TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC,
                                    ids=[BASE_ID])
                    trim_goal = seq.pop(0)
                    trim_seq = seq
                    arm.move_to(BASE_ID, trim_goal)
                    phase, phase_start = 'TRIM', now
                    log(f"TRIM — {TRIM_PASSES} level passes "
                        f"+/-{TRIM_SWEEP_HALF_RAW} raw about {b}")

            elif phase == 'TRIM':
                status = f'TRIMMING ({len(trim_seq)} moves left)'
                # Advance to the next base goal each time the current is reached.
                if abs(base_raw - trim_goal) <= ARRIVE_TOL_RAW * 2:
                    if trim_seq:
                        trim_goal = trim_seq.pop(0)
                        arm.move_to(BASE_ID, trim_goal)
                    else:
                        arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                        phase, phase_start = 'LIFT', now
                        log("LIFT — trim complete")
                elif now - phase_start > 15.0:    # sweep stuck — bail safely
                    arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                    phase, phase_start = 'LIFT', now
                    log("TRIM watchdog — lifting")

            elif phase == 'LIFT':
                go_rest_and_home()
                phase, phase_start = 'IDLE', now
                target = None
                log("done — looking for next patch")

            # ---- overlay ----
            for p in patches:
                x, y, w, h = p['bbox']
                sel = target is not None and p is target
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (0, 255, 0) if sel else (0, 190, 255),
                              3 if sel else 1)
            cv2.line(image, (centre_x, 0), (centre_x, CAMERA_HEIGHT),
                     (80, 80, 80), 1)
            cv2.putText(image, f'[{phase}] {status}', (8, 24), _FONT, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key in (13, 10) and now - last_enter > 0.4:
                last_enter = now
                auto = not auto
                log(f"AUTO {'ON' if auto else 'OFF'}")
            if key == ord(' ') and phase == 'IDLE' and target is not None:
                base_centre = None
                centre_ok = 0
                track_reset()
                phase, phase_start = 'CENTRE', now
                log("SPACE — trimming highlighted patch")
            if key == ord('e'):
                arm.emergency_stop()
                log("E-STOP")
                break
    finally:
        try:
            arm.set_profile(MOVE_SPEED, MOVE_ACC)
            arm.move_pose(REST_POSE)
            arm.wait_arrived(REST_POSE)
            time.sleep(0.4)
            arm.torque_off()
        except TrimmerError:
            pass
        arm.close()
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        print("parked, torque off.")


if __name__ == '__main__':
    main()
