"""Move the trimmer head to a detected object.

    python reach_object.py          # TOP mode  — flat, cut the top
    python reach_object.py edge     # EDGE mode — vertical (flat + 90deg wrist)

Both modes use the SAME taught flat table (CUT_POINTS). EDGE just rotates the
wrist 90 deg so the trimmer stands vertical — no separate edge calibration.
Each mode keeps its own fine trim (reach_trim.py / reach_trim_edge.py).

Bench test on a desk: an object (a Ryobi battery, standing proud of the desk
plane) is detected, the base centres on it, and the arm lowers the trimmer to
the object using the taught reach table.

Near-range note: the OAK-D Lite cannot measure closer than ~0.18m (below that
it wraps and reads FURTHER). Keep the object beyond MIN_RELIABLE_Z; readings
below it are rejected. If the trimmer lands slightly off the top, nudge
REACH_TRIM in trimmer_config.py — no re-teaching needed.

Keys:  CLICK an object to reach it · SPACE reach the highlighted one ·
       ENTER toggle auto ·
       o/l raise/lower the head (HEIGHT, shoulder) ·
       k/j tilt the head to keep it FLAT (wrist) ·
       p SAVE the current trim (persists across runs) ·
       r lift back to REST · e E-STOP · q quit
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
import os
import sys
import time
import cv2
import numpy as np

from trimmer_config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, BASE_FORWARD_RAW, RAW_PER_PIXEL,
    BASE_DIRECTION_SIGN, CAMERA_LATENCY_S, MIN_RELIABLE_Z,
    BASE_ID, SHOULDER_ID, ELBOW_ID, WRIST_ID,
    MOVE_SPEED, MOVE_ACC, ARRIVE_TOL_RAW,
    CENTRE_GAIN, TRACK_ERR_SMOOTH, TRACK_MIN_CMD_INTERVAL_S,
    CENTRE_DEADBAND_PX, CENTRE_STABLE_FRAMES, CENTRE_SPEED, CENTRE_ACC,
    OBJECT_MIN_PROUD_M, OBJECT_MAX_AREA_PX, OBJECT_MIN_AREA_PX,
    GRASS_MAX_PROUD_M, REACH_TRIM, HEIGHT_TRIM_STEP, APPROACH_HOVER_FRAC,
    REST_POSE, CUT_POINTS, REACH_TRIM_EDGE, EDGE_WRIST_OFFSET_RAW,
)
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline, detect_patches

ARM_JOINTS = (SHOULDER_ID, ELBOW_ID, WRIST_ID)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
HOLD_S = 2.0            # how long to sit at the object top in auto mode


def reach_arm_pose(z, table, base_trim, lock_angle=False):
    """Taught ARM pose (joints 2/3/4) for an object at depth z, plus the
    base_trim offset. For EDGE mode base_trim carries a +90deg wrist term so
    the trimmer stands vertical. Base excluded — it stays centred on the object.

    lock_angle=True holds a CONSTANT head angle: shoulder+elbow+wrist is a
    fixed sum (the median of the taught points), so the wrist is COMPUTED
    from the interpolated shoulder/elbow instead of interpolated on its own.
    This keeps the head at one consistent orientation even when the taught
    poses were inconsistent (which hand-teaching a vertical angle usually is)."""
    pts = sorted(table, key=lambda p: p[0])
    if z <= pts[0][0]:
        lo = hi = pts[0]; t = 0.0
    elif z >= pts[-1][0]:
        lo = hi = pts[-1]; t = 0.0
    else:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= z <= b[0]:
                lo, hi, t = a, b, (z - a[0]) / max(1e-6, b[0] - a[0])
                break
    pose = {}
    for sid in ARM_JOINTS:
        if sid in lo[1] and sid in hi[1]:
            pose[sid] = int(round(lo[1][sid] + t * (hi[1][sid] - lo[1][sid])))
    # Shoulder (height) + elbow trims go on first.
    for sid in (SHOULDER_ID, ELBOW_ID):
        if sid in pose:
            pose[sid] += base_trim.get(sid, 0)
    if lock_angle and all(s in pose for s in ARM_JOINTS):
        # Hold a constant head angle: wrist = target_sum - shoulder - elbow,
        # computed AFTER the shoulder/elbow trims so tuning height never tilts
        # the head. base_trim[wrist] (k/j) shifts the locked angle itself.
        sums = sorted(p[SHOULDER_ID] + p[ELBOW_ID] + p[WRIST_ID]
                      for _, p in pts
                      if all(s in p for s in ARM_JOINTS))
        target = sums[len(sums) // 2]          # median: robust to bad captures
        pose[WRIST_ID] = (target - pose[SHOULDER_ID] - pose[ELBOW_ID]
                          + base_trim.get(WRIST_ID, 0))
    elif WRIST_ID in pose:
        pose[WRIST_ID] += base_trim.get(WRIST_ID, 0)
    return pose


def hover_pose(full):
    """Part-way from REST to the full reach pose — a safe approach, no lunge."""
    return {sid: int(round(REST_POSE[sid]
                           + APPROACH_HOVER_FRAC * (full[sid] - REST_POSE[sid])))
            for sid in full}


def main():
    edge = len(sys.argv) > 1 and sys.argv[1].lower() == 'edge'
    if REST_POSE is None or not CUT_POINTS:
        print("Not calibrated. Run  python calibrate_reach.py  first.")
        return
    # BOTH modes use the flat CUT_POINTS table. EDGE just rotates the wrist 90
    # deg (so the trimmer stands vertical) and carries its own fine trim.
    table = CUT_POINTS
    if edge:
        save_base = dict(REACH_TRIM_EDGE)      # what 'p' saves (fine trim only)
        # effective offset = flat trim + 90deg wrist + edge fine trim
        base_trim = {
            SHOULDER_ID: REACH_TRIM.get(SHOULDER_ID, 0) + save_base.get(SHOULDER_ID, 0),
            ELBOW_ID: REACH_TRIM.get(ELBOW_ID, 0) + save_base.get(ELBOW_ID, 0),
            WRIST_ID: (REACH_TRIM.get(WRIST_ID, 0) + EDGE_WRIST_OFFSET_RAW
                       + save_base.get(WRIST_ID, 0)),
        }
        persist_name, save_var, mode_name = ('reach_trim_edge.py',
                                             'REACH_TRIM_EDGE', 'EDGE (vertical)')
    else:
        save_base = dict(REACH_TRIM)
        base_trim = dict(REACH_TRIM)
        persist_name, save_var, mode_name = ('reach_trim.py',
                                             'REACH_TRIM', 'TOP (flat)')
    print(f"MODE: {mode_name}")

    arm = TrimmerArm()
    pipeline, queue = build_pipeline()
    pipeline.start()
    centre_x = CAMERA_WIDTH // 2

    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n); time.sleep(1)
    arm.enable()
    arm.set_profile(MOVE_SPEED, MOVE_ACC)
    arm.move_pose(REST_POSE)
    arm.wait_arrived(REST_POSE)
    arm.move_to(BASE_ID, BASE_FORWARD_RAW)
    print("READY. CLICK to reach · SPACE = highlighted · ENTER = auto · "
          "o/l = HEIGHT · k/j = FLAT · r = lift · q = quit.")

    phase, phase_start = 'IDLE', time.monotonic()
    auto = False
    target = None
    full_pose = None
    centre_ok = 0
    trk = {'es': None, 'cmd_t': 0.0}
    pos_hist = []
    last_key = 0.0
    # Live DELTAS on top of REACH_TRIM: shoulder = height, wrist = head angle.
    live_trim = {SHOULDER_ID: 0, WRIST_ID: 0}
    click = {'pt': None}
    window = 'SO-101 Trimmer — reach to object top'

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def log(m):
        print(f"[{time.monotonic():7.2f}] {m}")

    def apply_height(pose):
        """Apply the live shoulder (height) + wrist (angle) nudges."""
        p = dict(pose)
        for sid, d in live_trim.items():
            if sid in p:
                p[sid] += d
        return p

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

    def lift_to_rest():
        arm.set_profile(MOVE_SPEED, MOVE_ACC)
        arm.move_pose(REST_POSE)
        arm.wait_arrived(REST_POSE, tol=ARRIVE_TOL_RAW * 3)
        arm.move_to(BASE_ID, BASE_FORWARD_RAW)

    def engage(t):
        nonlocal phase, phase_start, target, centre_ok
        target = t
        centre_ok = 0
        track_reset()
        arm.set_profile(CENTRE_SPEED, CENTRE_ACC, ids=[BASE_ID])
        phase, phase_start = 'CENTRE', time.monotonic()
        log(f"CENTRE on object z={t['z']:.3f} area={t['area']:.0f}")

    last_frame_t = time.monotonic()
    try:
        while pipeline.isRunning():
            now = time.monotonic()
            group = queue.tryGet()
            if group is None:
                # Camera-crash watchdog: the OAK-D can drop off the bus. If no
                # frame arrives for a while, stop and park rather than freeze
                # with the trimmer held down at an object.
                if now - last_frame_t > 3.0:
                    log("CAMERA STOPPED (no frames 3s) — parking arm")
                    break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            last_frame_t = now
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0
            try:
                base_raw = arm.read_position(BASE_ID)
                pos_hist.append((now, base_raw))
                if len(pos_hist) > 40:
                    pos_hist.pop(0)
            except TrimmerError:
                base_raw = BASE_FORWARD_RAW

            patches = (detect_patches(image, depth, min_proud=OBJECT_MIN_PROUD_M,
                                      max_proud=GRASS_MAX_PROUD_M,
                                      min_area=OBJECT_MIN_AREA_PX,
                                      max_area=OBJECT_MAX_AREA_PX)
                       if phase in ('IDLE', 'CENTRE') else [])
            status = phase

            # ---- click to select + engage a specific object ----
            if click['pt'] is not None:
                mx, my = click['pt']
                click['pt'] = None
                if phase == 'IDLE' and patches:
                    pick = min(patches, key=lambda p: (p['u'] - mx) ** 2
                               + (p['v'] - my) ** 2)
                    if ((pick['u'] - mx) ** 2 + (pick['v'] - my) ** 2) ** 0.5 <= 90:
                        log(f"CLICK -> object at ({pick['u']},{pick['v']}) "
                            f"z={pick['z']:.3f}")
                        engage(pick)

            if phase == 'IDLE':
                target = patches[0] if patches else None
                status = ('AUTO — seeking object' if auto else
                          'SPACE reach · ENTER auto')
                if auto and target is not None:
                    engage(target)

            elif phase == 'CENTRE':
                cur = None
                if patches:
                    cur = (min(patches, key=lambda p: (p['u'] - target['u']) ** 2
                               + (p['v'] - target['v']) ** 2)
                           if target else min(patches,
                                              key=lambda p: abs(p['u'] - centre_x)))
                    target = cur
                if cur is None:
                    if now - phase_start > 2.0:
                        log("object lost — back to REST")
                        lift_to_rest(); phase, phase_start = 'IDLE', now
                else:
                    err = cur['u'] - centre_x
                    present = base_at(now - CAMERA_LATENCY_S) or base_raw
                    if abs(err) <= CENTRE_DEADBAND_PX:
                        centre_ok += 1
                        if centre_ok >= CENTRE_STABLE_FRAMES:
                            full_pose = reach_arm_pose(cur['z'], table, base_trim)
                            arm.set_profile(MOVE_SPEED, MOVE_ACC)
                            arm.move_pose(hover_pose(apply_height(full_pose)))
                            phase, phase_start = 'APPROACH', now
                            log(f"APPROACH hover, z={cur['z']:.3f}")
                    else:
                        centre_ok = 0
                        g = track_goal(present, err, now)
                        if g is not None:
                            arm.move_to(BASE_ID, g)
                    status = f'CENTRING err={err:+d}px'

            elif phase == 'APPROACH':
                status = 'APPROACHING (hover)'
                if arm.arrived(hover_pose(apply_height(full_pose)),
                               tol=ARRIVE_TOL_RAW * 2) or now - phase_start > 2.5:
                    arm.move_pose(apply_height(full_pose))   # to the object top
                    phase, phase_start = 'REACH', now
                    log(f"REACH to object top: {apply_height(full_pose)}")

            elif phase == 'REACH':
                status = 'MOVING TO OBJECT TOP'
                if arm.arrived(apply_height(full_pose), tol=ARRIVE_TOL_RAW * 2) \
                        or now - phase_start > 3.0:
                    phase, phase_start = 'HOLD', now
                    log("AT OBJECT TOP")

            elif phase == 'HOLD':
                status = (f"AT TOP — o/l height({live_trim[SHOULDER_ID]:+d}) "
                          f"k/j flat({live_trim[WRIST_ID]:+d}) · r lift")
                if auto and now - phase_start >= HOLD_S:
                    lift_to_rest(); phase, phase_start = 'IDLE', now
                    log("lift — next object")

            # ---- overlay ----
            for p in patches:
                x, y, w, h = p['bbox']
                sel = target is not None and p is target
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (0, 255, 0) if sel else (0, 190, 255),
                              3 if sel else 1)
                if sel:
                    cv2.putText(image, f"z={p['z']:.3f}m", (x, max(14, y - 6)),
                                _FONT, 0.5, (0, 255, 0), 2)
            cv2.line(image, (centre_x, 0), (centre_x, CAMERA_HEIGHT),
                     (80, 80, 80), 1)
            cv2.putText(image, f'[{phase}] {status}', (8, 24), _FONT, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow(window, image)

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key == ord(' ') and phase == 'IDLE' and target is not None \
                    and now - last_key > 0.3:
                last_key = now
                engage(target)
            if key in (13, 10) and now - last_key > 0.3:
                last_key = now
                auto = not auto
                log(f"AUTO {'ON' if auto else 'OFF'}")
            if key == ord('r') and phase in ('APPROACH', 'REACH', 'HOLD'):
                lift_to_rest(); phase, phase_start = 'IDLE', now
                log("lifted to REST")
            # live tuning: o/l = HEIGHT (shoulder), k/j = head ANGLE (wrist).
            _tuned = None
            if key == ord('o'):
                live_trim[SHOULDER_ID] += HEIGHT_TRIM_STEP; _tuned = SHOULDER_ID
            elif key == ord('l'):
                live_trim[SHOULDER_ID] -= HEIGHT_TRIM_STEP; _tuned = SHOULDER_ID
            elif key == ord('k'):
                live_trim[WRIST_ID] += HEIGHT_TRIM_STEP; _tuned = WRIST_ID
            elif key == ord('j'):
                live_trim[WRIST_ID] -= HEIGHT_TRIM_STEP; _tuned = WRIST_ID
            if _tuned is not None and full_pose is not None \
                    and phase in ('APPROACH', 'REACH', 'HOLD'):
                arm.move_pose(apply_height(full_pose))
                cfg2 = save_base.get(SHOULDER_ID, 0) + live_trim[SHOULDER_ID]
                cfg4 = save_base.get(WRIST_ID, 0) + live_trim[WRIST_ID]
                log(f"trim shoulder{live_trim[SHOULDER_ID]:+d} "
                    f"wrist{live_trim[WRIST_ID]:+d}  ->  {save_var} = "
                    f"{{2: {cfg2}, 3: 0, 4: {cfg4}}}  (press p to save)")
            if key == ord('p'):
                c2 = save_base.get(SHOULDER_ID, 0) + live_trim[SHOULDER_ID]
                c3 = save_base.get(ELBOW_ID, 0)
                c4 = save_base.get(WRIST_ID, 0) + live_trim[WRIST_ID]
                from paths import CORE as _C
                path = str(_C / persist_name)
                with open(path, 'w') as f:
                    f.write("# Saved by reach_object.py (p key) "
                            + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
                    f.write(f"{save_var} = {{2: {c2}, 3: {c3}, 4: {c4}}}\n")
                log(f"SAVED {save_var} = {{2: {c2}, 3: {c3}, 4: {c4}}} "
                    f"-> {persist_name} (persists next run)")
            if key == ord('e'):
                arm.emergency_stop(); log("E-STOP"); break
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
