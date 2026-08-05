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

CLICK any detected object to send the head there — from any phase, and it
lifts out of the way first if it needs to. The click is never silent: it marks
where you clicked and says on screen when there was nothing to pick.

Every key acts ONCE per press. Holding o/l/k/j repeats, deliberately, because
those are nudges you tune by feel. If nothing responds at all, click on the
video window first — keys only reach the focused window.

Once the head is on top of a patch it CUTS: a flat drop straight down through
it, then a base-only sweep left and right. The head stays flat and the base
axis is vertical, so the tip holds one height through the whole arc - a level
cut with no kinematics to get wrong.

There is also a memory of where it has already cut, so a trimmed patch is not
found and cut again - but it is OFF by default (REMEMBER_CLEARED below), so
right now every patch stays fair game however many times you hit it.

Keys:  CLICK an object to reach it · SPACE reach the highlighted one ·
       ENTER toggle auto · c cut now · x forget cleared areas ·
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
    GRASS_MAX_PROUD_M, REACH_TRIM, HEIGHT_TRIM_STEP,
    REST_POSE, CUT_POINTS, REACH_TRIM_EDGE, EDGE_WRIST_OFFSET_RAW,
    TRIM_SWEEP_HALF_RAW, TRIM_PASSES, TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC,
)
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline, detect_patches

ARM_JOINTS = (SHOULDER_ID, ELBOW_ID, WRIST_ID)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
HOLD_S = 1.0            # pause at the object top before cutting
# --- the cut -----------------------------------------------------------------
# How far the head drops THROUGH the patch once it has landed on top of it,
# as a MAGNITUDE in raw ticks. The direction is not set here and must not be:
# it is worked out from the arm's own geometry.
#
# The first version had it as a signed number and I picked the sign by
# reasoning about which way the shoulder was wired. It was wrong, so the head
# rose off the target instead of cutting into it. There is no need to guess:
# the arm has just travelled from REST to sitting on top of the patch, and
# that direction IS downward, by definition. Carrying on the same way is
# further down whatever the wiring does.
CUT_DROP_RAW = 120      # magnitude only - always a positive number
CUT_DROP_SPEED = 260    # deliberate, not a stab
CUT_DROP_ACC = 12
# How far a click may land from a detection and still count. Generous on
# purpose: the objects are small on screen and asking for pixel accuracy with
# a mouse is the reason the old 90 px limit felt like the feature did nothing.
CLICK_MAX_PX = 160

# --- remembering what has been cut ------------------------------------------
# OFF for now. When it is on, anywhere the head has cut is skipped for a few
# minutes, which stops the loop re-finding a patch it has already trimmed -
# a cut patch is shorter but still stands proud of the deck, so it comes
# straight back. Worth turning on once the cut itself is dialled in; while
# you are still tuning, being able to hit the same patch repeatedly is more
# useful than the machine refusing to.
REMEMBER_CLEARED = False


_warned = {}


def _warn_outside(z, edge, which):
    """Say once per distance that the reach is being clamped, and by how much.

    Once, not every frame: at camera rate this would bury everything else in
    the log, and the point is to be noticed, not to shout.
    """
    key = (round(z, 2), which)
    if key in _warned:
        return
    _warned[key] = True
    print(f'[reach] target at {z:.3f} m is {which} than the taught table '
          f'(edge {edge:.3f} m).')
    print(f'        The pose is clamped to the end of the table, so the head '
          f'will stop about {abs(z-edge)*100:.0f} cm short.')
    print(f'        Fix by teaching a point out there: '
          f'python trimmer/calibrate/calibrate_reach.py')


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
    # Outside the taught range the pose is CLAMPED to the nearest end. That is
    # the safe thing to do - extrapolating a hand-taught arm pose can drive the
    # head into the table - but done silently it looks like the arm simply
    # under-reaching by a fixed amount, with nothing to explain it. So say so.
    if z <= pts[0][0]:
        lo = hi = pts[0]; t = 0.0
        _warn_outside(z, pts[0][0], 'nearer')
    elif z >= pts[-1][0]:
        lo = hi = pts[-1]; t = 0.0
        _warn_outside(z, pts[-1][0], 'further')
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


def main():
    edge = len(sys.argv) > 1 and sys.argv[1].lower() == 'edge'
    if REST_POSE is None or not CUT_POINTS:
        print("Not calibrated. Run  python calibrate_reach.py  first.")
        return
    # BOTH modes use the flat CUT_POINTS table. EDGE just rotates the wrist 90
    # deg (so the trimmer stands vertical) and carries its own fine trim.
    table = CUT_POINTS
    # Audit the table before anything moves. Both of the ways it can be wrong -
    # a target outside the taught range, or a mis-taught point that makes the
    # arm retract as the target recedes - are properties of the table itself,
    # so there is no reason to discover them by watching the head miss.
    import reach_table as _rt
    _rt.report(table)
    reach_stats = {'ok': 0, 'clamped': 0, 'reversed': 0, 'near': 0}
    # where the head has already cut, so the same patch is not chased forever
    from cleared import ClearedZones, base_for_detection
    cleared = ClearedZones()
    # Servo reads block for up to 80 ms each. Throttling them is what keeps
    # the GUI event loop - and therefore the keyboard - alive.
    _check_at = {}
    _arrived_cache = False

    def _due(what, now, every=0.25):
        if now - _check_at.get(what, 0.0) < every:
            return False
        _check_at[what] = now
        return True

    cut_requested = False
    cut_pose = None
    sweep_seq, sweep_goal, sweep_centre = [], 0, 0
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
    click_mark = None       # where the last click landed, for a moment
    click_msg = None        # why a click did nothing, if it did nothing
    window = 'SO-101 Trimmer — reach to object top'

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    from keys import Keys
    kb = Keys()

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
                if kb.poll(1).hit(ord('q')):
                    break
                continue
            last_frame_t = now
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0
            try:
                kb.pump()          # the read below blocks; keep keys alive
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
            # Drop anything sitting where the head has already cut. A patch
            # that has just been trimmed is shorter but still stands proud of
            # the deck, so without this the loop finds it again, cuts it
            # again, and never reaches the second patch - which looks like the
            # machine working and is actually it stuck.
            if REMEMBER_CLEARED and patches and len(cleared):
                kept, skipped = [], []
                for p_ in patches:
                    b = base_for_detection(p_['u'], base_raw, centre_x,
                                           RAW_PER_PIXEL, BASE_DIRECTION_SIGN)
                    p_['base_for'] = b
                    (skipped if cleared.is_cleared(b, p_['z']) else kept).append(p_)
                cleared_now = skipped
                patches = kept
            else:
                cleared_now = []
            status = phase

            # ---- click to pick the target -------------------------------
            # This worked only while IDLE and only within 90 px of a detection,
            # and said nothing when it did neither - so a click during a reach,
            # or a click slightly off the object, simply vanished and the
            # feature felt broken. Now it works from any phase (lifting first
            # if the arm is out), takes the nearest detection whatever the
            # distance, and says so on screen when there is nothing to pick.
            if click['pt'] is not None:
                mx, my = click['pt']
                click['pt'] = None
                click_mark = (mx, my, now)
                if not patches:
                    click_msg = ('nothing detected there - is the object '
                                 'standing proud of the desk?', now)
                else:
                    pick = min(patches, key=lambda p: (p['u'] - mx) ** 2
                               + (p['v'] - my) ** 2)
                    d = ((pick['u'] - mx) ** 2 + (pick['v'] - my) ** 2) ** 0.5
                    if d > CLICK_MAX_PX:
                        click_msg = (f'nearest object is {d:.0f} px away - '
                                     f'click closer to it', now)
                    else:
                        if phase != 'IDLE':
                            lift_to_rest()
                            phase, phase_start = 'IDLE', now
                            log('clicked a new target - lifting first')
                        log(f"CLICK -> object at ({pick['u']},{pick['v']}) "
                            f"z={pick['z']:.3f}")
                        target = pick
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
                            _ok, _short, _msg = _rt.verdict(cur['z'], table)
                            if _ok:
                                reach_stats['ok'] += 1
                            elif _short:
                                reach_stats['clamped'] += 1
                            elif 'reversed' in _msg:
                                reach_stats['reversed'] += 1
                            else:
                                reach_stats['near'] += 1
                            arm.set_profile(MOVE_SPEED, MOVE_ACC)
                            arm.move_pose(apply_height(full_pose))   # straight to the object top
                            phase, phase_start = 'REACH', now
                            log(f"REACH to object top: {apply_height(full_pose)}")
                    else:
                        centre_ok = 0
                        g = track_goal(present, err, now)
                        if g is not None:
                            arm.move_to(BASE_ID, g)
                    status = f'CENTRING err={err:+d}px'

            elif phase == 'REACH':
                status = 'MOVING TO OBJECT TOP'
                # arrived() reads EVERY joint - three blocking round trips at
                # up to 80 ms each. Every frame, that is what starved the
                # keyboard. Check a few times a second instead; the arm takes
                # far longer than that to get anywhere.
                kb.pump()
                if _due('reach', now):
                    _arrived_cache = arm.arrived(apply_height(full_pose),
                                                 tol=ARRIVE_TOL_RAW * 2)
                if _arrived_cache or now - phase_start > 3.0:
                    phase, phase_start = 'HOLD', now
                    log("AT OBJECT TOP")

            elif phase == 'HOLD':
                status = (f"AT TOP — o/l height({live_trim[SHOULDER_ID]:+d}) "
                          f"k/j flat({live_trim[WRIST_ID]:+d}) · "
                          f"c CUT · r lift")
                # Landing on top of the patch is only the aim. The cut is
                # dropping through it and sweeping - so that happens next,
                # rather than lifting straight back off.
                if (auto or cut_requested) and now - phase_start >= HOLD_S:
                    cut_requested = False
                    # Keep going the way the arm was already going. REST is
                    # up and clear; the reach pose is down on the target; so
                    # the sign of (reach - rest) is "downward" for this arm,
                    # measured rather than assumed.
                    _here = apply_height(full_pose)
                    _down = _here[SHOULDER_ID] - REST_POSE[SHOULDER_ID]
                    _sign = 1 if _down >= 0 else -1
                    cut_pose = dict(_here)
                    cut_pose[SHOULDER_ID] += _sign * abs(CUT_DROP_RAW)
                    arm.set_profile(CUT_DROP_SPEED, CUT_DROP_ACC)
                    arm.move_pose(cut_pose)
                    phase, phase_start = 'CUT', now
                    log(f"CUT — dropping {_sign * abs(CUT_DROP_RAW):+d} raw "
                        f"into the patch (down is {'+' if _sign > 0 else '-'} "
                        f"on this arm)")

            elif phase == 'CUT':
                # Straight down, flat, into the grass. One move, then sweep.
                status = 'CUTTING DOWN'
                kb.pump()
                if _due('cut', now):
                    _arrived_cache = arm.arrived(cut_pose, tol=ARRIVE_TOL_RAW * 2)
                if _arrived_cache or now - phase_start > 2.5:
                    # The sweep moves ONLY the base. The head is flat and the
                    # base axis is vertical, so the tip stays at one height
                    # through the whole arc - a level cut with no kinematics
                    # to get wrong. Same trick main_trimmer.py uses.
                    b = arm.read_position(BASE_ID) or BASE_FORWARD_RAW
                    sweep_centre = b
                    sweep_seq = []
                    for _ in range(TRIM_PASSES):
                        sweep_seq += [b - TRIM_SWEEP_HALF_RAW,
                                      b + TRIM_SWEEP_HALF_RAW]
                    sweep_seq.append(b)          # finish back on centre
                    arm.set_profile(TRIM_SWEEP_SPEED, TRIM_SWEEP_ACC,
                                    ids=[BASE_ID])
                    sweep_goal = sweep_seq.pop(0)
                    arm.move_to(BASE_ID, sweep_goal)
                    phase, phase_start = 'SWEEP', now
                    log(f"SWEEP — {TRIM_PASSES} passes "
                        f"+/-{TRIM_SWEEP_HALF_RAW} raw about {b}")

            elif phase == 'SWEEP':
                status = f'SWEEPING ({len(sweep_seq)} moves left)'
                kb.pump()
                here_raw = arm.read_position(BASE_ID)
                if here_raw is not None and \
                        abs(here_raw - sweep_goal) <= ARRIVE_TOL_RAW * 2:
                    if sweep_seq:
                        sweep_goal = sweep_seq.pop(0)
                        arm.move_to(BASE_ID, sweep_goal)
                    else:
                        arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                        if REMEMBER_CLEARED and target is not None:
                            cleared.mark(sweep_centre, target['z'])
                            log(f"cleared — {cleared.summary()}")
                        lift_to_rest()
                        phase, phase_start = 'IDLE', now
                        log("cut complete — lifting")
                elif now - phase_start > 20.0:
                    arm.set_profile(MOVE_SPEED, MOVE_ACC, ids=[BASE_ID])
                    lift_to_rest()
                    phase, phase_start = 'IDLE', now
                    log("SWEEP watchdog — lifting")

            # ---- overlay ----
            # Say on screen whether the table can actually reach the selected
            # target. This is the one number that explains a consistent miss,
            # and it was previously invisible until someone measured the gap
            # with a ruler.
            if target is not None:
                _ok, _short, _msg = _rt.verdict(target['z'], table)
                if not _ok:
                    _t = (f"TARGET {target['z']:.3f} m - {_msg}"
                          + (f", the head will stop {_short:.0f} cm short"
                             if _short else ''))
                    cv2.rectangle(image, (0, image.shape[0] - 30),
                                  (image.shape[1], image.shape[0]), (0, 0, 90), -1)
                    cv2.putText(image, _t, (8, image.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 160, 255), 1)

            # show where a click landed, and why it did nothing if it did
            if click_mark is not None and now - click_mark[2] < 1.0:
                cv2.drawMarker(image, (click_mark[0], click_mark[1]),
                               (255, 220, 90), cv2.MARKER_CROSS, 22, 2)
            if click_msg is not None and now - click_msg[1] < 2.0:
                cv2.putText(image, click_msg[0], (10, image.shape[0] - 40),
                            _FONT, 0.5, (90, 200, 255), 1, cv2.LINE_AA)

            for p in cleared_now:
                x, y, w, h = p['bbox']
                cv2.rectangle(image, (x, y), (x + w, y + h), (120, 120, 120), 1)
                cv2.putText(image, 'cleared', (x, max(12, y - 5)),
                            _FONT, 0.42, (150, 150, 150), 1)
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

            # ONE press, ONE action. cv2.waitKey reports which key is DOWN, not
            # when it was pressed, so acting on the value acted repeatedly while
            # a key was held - one tap of ENTER used to toggle auto several
            # times. keys.Keys watches for the transition instead. The trims
            # still repeat when held, deliberately, because nudging by feel is
            # what they are for.
            kb.poll(1)
            key = kb.key
            if kb.hit(ord('q')):
                break
            if kb.hit(ord(' ')) and phase == 'IDLE' and target is not None:
                engage(target)
            if kb.hit(13, 10):
                auto = not auto
                log(f"AUTO {'ON' if auto else 'OFF'}")
            if kb.hit(ord('c')) and phase == 'HOLD':
                cut_requested = True
                log('cut requested')
            if kb.hit(ord('x')):
                n = cleared.forget_all()
                log(f'forgot {n} cleared area(s) - all fair game again')
            if kb.hit(ord('r')) and phase in ('REACH', 'HOLD'):
                lift_to_rest(); phase, phase_start = 'IDLE', now
                log("lifted to REST")
            # live tuning: o/l = HEIGHT (shoulder), k/j = head ANGLE (wrist).
            _tuned = None
            if kb.tap(ord('o')):
                live_trim[SHOULDER_ID] += HEIGHT_TRIM_STEP; _tuned = SHOULDER_ID
            elif kb.tap(ord('l')):
                live_trim[SHOULDER_ID] -= HEIGHT_TRIM_STEP; _tuned = SHOULDER_ID
            elif kb.tap(ord('k')):
                live_trim[WRIST_ID] += HEIGHT_TRIM_STEP; _tuned = WRIST_ID
            elif kb.tap(ord('j')):
                live_trim[WRIST_ID] -= HEIGHT_TRIM_STEP; _tuned = WRIST_ID
            if _tuned is not None and full_pose is not None \
                    and phase in ('REACH', 'HOLD'):
                arm.move_pose(apply_height(full_pose))
                cfg2 = save_base.get(SHOULDER_ID, 0) + live_trim[SHOULDER_ID]
                cfg4 = save_base.get(WRIST_ID, 0) + live_trim[WRIST_ID]
                log(f"trim shoulder{live_trim[SHOULDER_ID]:+d} "
                    f"wrist{live_trim[WRIST_ID]:+d}  ->  {save_var} = "
                    f"{{2: {cfg2}, 3: 0, 4: {cfg4}}}  (press p to save)")
            if kb.hit(ord('p')):
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
            if kb.hit(ord('e')):
                arm.emergency_stop(); log("E-STOP"); break
    finally:
        # A session summary, because a systematic miss is obvious in the
        # aggregate and easy to shrug off one reach at a time.
        _n = sum(reach_stats.values())
        if _n:
            print(f'\n[reach] {_n} reach(es): {reach_stats["ok"]} inside the '
                  f'taught range, {reach_stats["clamped"]} clamped beyond it, '
                  f'{reach_stats["reversed"]} in a reversed segment, '
                  f'{reach_stats["near"]} too near.')
            if reach_stats['ok'] < _n:
                print('[reach] every one of those will have landed short. '
                      'Re-teach the table:')
                print('        python trimmer/calibrate/calibrate_reach.py')
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
