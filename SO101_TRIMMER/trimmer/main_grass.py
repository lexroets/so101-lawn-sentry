"""Autonomous grass trimming — the full loop.

Detect an uncut-grass region (CNN + GLCM + depth), centre the base on it,
lower the trimmer to the taught level-cut height, sweep the base to cut it
level, lift, move on. Camera crashes are survived (arm parks, camera
restarts). This is the battery-sorter framework pointed at grass.

    python main_grass.py            # ENTER = auto-trim, SPACE = trim one region

Keys:  ENTER toggle auto · SPACE trim highlighted region · e E-STOP · q quit
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
    BASE_DIRECTION_SIGN, CAMERA_LATENCY_S, BASE_ID,
    MOVE_SPEED, MOVE_ACC, ARRIVE_TOL_RAW,
    CENTRE_GAIN, TRACK_ERR_SMOOTH, TRACK_MIN_CMD_INTERVAL_S,
    CENTRE_DEADBAND_PX, CENTRE_STABLE_FRAMES, CENTRE_SPEED, CENTRE_ACC,
    TRIM_SWEEP_HALF_RAW, TRIM_PASSES, TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC,
    REST_POSE, CUT_POINTS, REACH_TRIM,
)
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline
from grass_detector import GrassDetector
from camera_session import CameraSession
from reach_object import reach_arm_pose, hover_pose

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    if REST_POSE is None or not CUT_POINTS:
        print("Not calibrated. Run  python calibrate_reach.py  first "
              "(need REST_POSE + CUT_POINTS for the level cut).")
        return

    arm = TrimmerArm()
    cam = CameraSession(build_pipeline)
    det = GrassDetector()
    centre_x = CAMERA_WIDTH // 2
    src = 'CNN+GLCM+depth' if det.clf.ready else 'GLCM+depth (no model yet)'
    print(f"[grass] detecting with {src}")

    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n); time.sleep(1)
    arm.enable()

    def park():
        arm.set_profile(MOVE_SPEED, MOVE_ACC)
        arm.move_pose(REST_POSE)
        arm.wait_arrived(REST_POSE, tol=ARRIVE_TOL_RAW * 3)
        arm.move_to(BASE_ID, BASE_FORWARD_RAW)

    park()
    print("READY. ENTER = auto-trim all · SPACE = trim highlighted · q = quit.")

    phase, phase_start = 'IDLE', time.monotonic()
    auto = False
    target = None
    full_pose = None
    base_centre = None
    centre_ok = 0
    trim_seq, trim_goal = [], None
    trk = {'es': None, 'cmd_t': 0.0}
    pos_hist = []
    last_key = 0.0
    window = 'SO-101 Autonomous Grass Trimmer'

    def log(m):
        print(f"[{time.monotonic():7.2f}] {m}")

    def base_at(tw):
        for t, raw in reversed(pos_hist):
            if t <= tw:
                return raw
        return pos_hist[0][1] if pos_hist else None

    def track_reset():
        trk['es'], trk['cmd_t'] = None, 0.0

    def track_goal(present, err, now):
        es = float(err) if trk['es'] is None else (
            TRACK_ERR_SMOOTH * trk['es'] + (1.0 - TRACK_ERR_SMOOTH) * err)
        trk['es'] = es
        if abs(es) <= CENTRE_DEADBAND_PX:
            return None
        if now - trk['cmd_t'] < TRACK_MIN_CMD_INTERVAL_S:
            return None
        trk['cmd_t'] = now
        return present + BASE_DIRECTION_SIGN * CENTRE_GAIN * es * RAW_PER_PIXEL

    def engage(tgt):
        nonlocal phase, phase_start, target, centre_ok, base_centre
        target, centre_ok, base_centre = tgt, 0, None
        track_reset()
        arm.set_profile(CENTRE_SPEED, CENTRE_ACC, ids=[BASE_ID])
        phase, phase_start = 'CENTRE', time.monotonic()
        log(f"CENTRE on grass z={tgt['z']:.3f} area={tgt['area']:.0f} "
            f"score={tgt['score']:.2f}")

    try:
        while True:                      # runs through camera restarts
            now = time.monotonic()
            group = cam.get()
            if group is None:
                if cam.crashed():
                    log("CAMERA STOPPED — parking + restarting")
                    try:
                        park()
                    except TrimmerError:
                        pass
                    phase, phase_start = 'IDLE', now
                    if not cam.restart():
                        log("camera restart failed — exiting")
                        break
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

            regions = (det.detect(image, depth)
                       if phase in ('IDLE', 'CENTRE') else [])
            status = phase

            if phase == 'IDLE':
                target = regions[0] if regions else None
                status = 'AUTO — seeking grass' if auto else 'ENTER auto · SPACE one'
                if auto and target is not None:
                    engage(target)

            elif phase == 'CENTRE':
                cur = None
                if regions:
                    cur = (min(regions, key=lambda p: (p['u'] - target['u']) ** 2
                               + (p['v'] - target['v']) ** 2) if target
                           else min(regions, key=lambda p: abs(p['u'] - centre_x)))
                    target = cur
                if cur is None:
                    if now - phase_start > 2.0:
                        log("region lost — back to IDLE"); park()
                        phase, phase_start = 'IDLE', now
                else:
                    err = cur['u'] - centre_x
                    present = base_at(now - CAMERA_LATENCY_S) or base_raw
                    if abs(err) <= CENTRE_DEADBAND_PX:
                        centre_ok += 1
                        if centre_ok >= CENTRE_STABLE_FRAMES:
                            base_centre = base_raw
                            full_pose = reach_arm_pose(cur['z'], CUT_POINTS,
                                                       REACH_TRIM)
                            arm.set_profile(MOVE_SPEED, MOVE_ACC)
                            arm.move_pose(hover_pose(full_pose))
                            phase, phase_start = 'LOWER', now
                            log(f"LOWER to level cut z={cur['z']:.3f}")
                    else:
                        centre_ok = 0
                        g = track_goal(present, err, now)
                        if g is not None:
                            arm.move_to(BASE_ID, g)
                    status = f'CENTRING err={err:+d}px'

            elif phase == 'LOWER':
                status = 'LOWERING trimmer'
                if arm.arrived(hover_pose(full_pose), tol=ARRIVE_TOL_RAW * 2) \
                        or now - phase_start > 2.0:
                    arm.move_pose(full_pose)
                if arm.arrived(full_pose, tol=ARRIVE_TOL_RAW * 2) \
                        or now - phase_start > 3.5:
                    b = base_centre
                    seq = []
                    for _ in range(TRIM_PASSES):
                        seq += [b - TRIM_SWEEP_HALF_RAW, b + TRIM_SWEEP_HALF_RAW]
                    seq.append(b)
                    arm.set_profile(TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC, ids=[BASE_ID])
                    trim_goal = seq.pop(0); trim_seq = seq
                    arm.move_to(BASE_ID, trim_goal)
                    phase, phase_start = 'TRIM', now
                    log(f"TRIM — {TRIM_PASSES} level passes +/-{TRIM_SWEEP_HALF_RAW}")

            elif phase == 'TRIM':
                status = f'TRIMMING ({len(trim_seq)} moves left)'
                if abs(base_raw - trim_goal) <= ARRIVE_TOL_RAW * 2:
                    if trim_seq:
                        trim_goal = trim_seq.pop(0)
                        arm.move_to(BASE_ID, trim_goal)
                    else:
                        arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                        phase, phase_start = 'LIFT', now
                        log("LIFT — region done")
                elif now - phase_start > 15.0:
                    arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                    phase, phase_start = 'LIFT', now
                    log("TRIM watchdog — lifting")

            elif phase == 'LIFT':
                park()
                phase, phase_start, target = 'IDLE', now, None
                log("done — next region")

            # ---- overlay ----
            for p in regions:
                x, y, w, h = p['bbox']
                sel = target is not None and p is target
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (80, 220, 100) if sel else (0, 190, 255),
                              3 if sel else 2)
            cv2.line(image, (centre_x, 0), (centre_x, CAMERA_HEIGHT), (80, 80, 80), 1)
            cv2.putText(image, f'[{phase}] {status}   {src}', (8, 24), _FONT,
                        0.55, (0, 255, 255), 2)
            cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key in (13, 10) and now - last_key > 0.4:
                last_key = now; auto = not auto; log(f"AUTO {'ON' if auto else 'OFF'}")
            if key == ord(' ') and phase == 'IDLE' and target is not None \
                    and now - last_key > 0.3:
                last_key = now; engage(target)
            if key == ord('e'):
                arm.emergency_stop(); log("E-STOP"); break
    finally:
        try:
            park(); time.sleep(0.4); arm.torque_off()
        except TrimmerError:
            pass
        arm.close()
        cv2.destroyAllWindows()
        cam.stop()
        print("parked, torque off.")


if __name__ == '__main__':
    main()
