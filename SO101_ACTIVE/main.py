"""SO-101 pick-and-place — ACTUATION ENABLED.

Start the program with the arm in its REST position: that pose is captured
and used as the carry/home pose.

Full cycle (both the target and the box in the camera's FOV):
  SELECT   : targets boxed yellow, drop-boxes magenta. ENTER locks the
             highlighted TARGET (e.g. the battery).
  SEL_BOX  : ENTER confirms the highlighted BOX (X = no box, grab only).
             Nothing moves until both are chosen.
  CENTRE   : base tracks the target closed-loop (latency-compensated)
  STAGE1/VERIFY/FULL : staged reach with mid-flight re-aim
  GRIP     : stall-detected close
  CARRY    : arm lifts to rest (gripper closed), base swings to the box
  PLACE    : base re-centres on the BOX visually, arm lowers to just above
             the rim (DROP_POSE_FRAC of a full reach)
  DROP     : gripper opens, object falls in, arm retracts. Cycle done.

Keys: ENTER = advance/confirm, or release+rest from any active phase
      X     = (in box selection) skip the box — grab only
      B     = (when gripped, no box selected) bring object home held
      E/Esc = latched emergency stop (cuts ALL torque — support the arm!)
      q     = quit (parks at rest first)
"""
from collections import deque
from datetime import timedelta
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import (detect_candidates, candidates_from_yolo,
                      classify_colour, _measure_depth)
from safety import SafetySupervisor
from actuator import BaseActuator, ActuatorError
from production import GrabVerifier, Watchdog, SelfCheck, SessionMetrics

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_log.txt')
_t0 = time.monotonic()


def log(msg):
    line = f"[{time.monotonic() - _t0:8.2f}s] {msg}"
    print(line)
    if DEBUG:
        try:
            with open(_LOG_PATH, 'a') as f:
                f.write(line + '\n')
        except Exception:
            pass


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
    frame_queue = sync.out.createOutputQueue(maxSize=1, blocking=False)

    det_queue = None
    labels = None
    if USE_YOLO:
        nn = pipeline.create(dai.node.DetectionNetwork).build(
            rgb, dai.NNModelDescription(YOLO_MODEL)
        )
        labels = nn.getClasses()
        det_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
    return pipeline, frame_queue, det_queue, labels


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = (236, 240, 241)          # near-white text
_DIM = (150, 160, 165)          # secondary text
_OK = (80, 220, 100)            # green
_WARN = (0, 190, 255)           # amber
_HOT = (255, 90, 200)           # magenta (top-down grab)
_BAD = (60, 60, 235)            # red


def _panel(img, x, y, w, h, alpha=0.62):
    """Translucent dark plate — keeps text readable over any background."""
    x, y = max(0, x), max(0, y)
    w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y:y + h, x:x + w]
    cv2.addWeighted(roi, 1.0 - alpha,
                    np.full_like(roi, 18), alpha, 0, dst=roi)


def _text(img, s, x, y, scale=0.46, colour=_INK, thick=1):
    cv2.putText(img, s, (x, y), _FONT, scale, colour, thick, cv2.LINE_AA)


def draw_hud(image, phase, status, status_colour, grab_top, auto_on,
             scanning, sweep, n_ryobi, n_pink, fps, target=None,
             stopped=False):
    """One consistent, readable overlay instead of scattered captions."""
    W = image.shape[1]

    # ---- top bar: title | phase | fps -----------------------------------
    _panel(image, 0, 0, W, 46, 0.68)
    _text(image, 'SO-101  AUTONOMOUS BATTERY SORTER', 12, 19, 0.50, _INK, 1)
    _text(image, status, 12, 37, 0.46, status_colour, 1)
    chip = str(phase)
    (tw, _th), _bl = cv2.getTextSize(chip, _FONT, 0.44, 1)
    cv2.rectangle(image, (W - tw - 90, 8), (W - 78, 28),
                  status_colour, 1, cv2.LINE_AA)
    _text(image, chip, W - tw - 84, 22, 0.44, status_colour, 1)
    if HUD_SHOW_FPS:
        _text(image, f'{fps:4.1f} fps', W - 68, 22, 0.44, _DIM, 1)

    # ---- grab-mode banner: ALWAYS visible --------------------------------
    gy = 54
    g_col = _HOT if grab_top else _OK
    _panel(image, 0, gy, 250, 30, 0.68)
    cv2.rectangle(image, (0, gy), (250, gy + 30), g_col, 1, cv2.LINE_AA)
    _text(image, 'GRAB:', 10, gy + 20, 0.48, _DIM, 1)
    _text(image, 'TOP-DOWN (vertical)' if grab_top else 'SIDE (upright)',
          62, gy + 20, 0.48, g_col, 2)
    _text(image, '[L] toggles', 258, gy + 20, 0.40, _DIM, 1)

    # ---- bottom-left: mode + tallies -------------------------------------
    H = image.shape[0]
    _panel(image, 0, H - 58, 300, 58, 0.62)
    if auto_on and scanning:
        _text(image, f'AUTO-SORT: SCANNING  {sweep}', 10, H - 36, 0.46, _WARN, 1)
    elif auto_on:
        _text(image, 'AUTO-SORT: RUNNING', 10, H - 36, 0.46, _OK, 1)
    else:
        _text(image, 'AUTO-SORT: idle   [ENTER] to start', 10, H - 36, 0.44,
              _DIM, 1)
    _text(image, f'RYOBI -> GREEN : {n_ryobi}', 10, H - 20, 0.44, _OK, 1)
    _text(image, f'PINK  -> RED   : {n_pink}', 165, H - 20, 0.44, _HOT, 1)

    # ---- selected target read-out ----------------------------------------
    if target is not None:
        col = target.get('colour', 'unknown')
        c_col = _OK if col == 'ryobi' else _HOT if col == 'pink' else _WARN
        _panel(image, W - 190, H - 58, 190, 58, 0.62)
        _text(image, 'TARGET', W - 180, H - 40, 0.44, _DIM, 1)
        _text(image, col.upper(), W - 118, H - 40, 0.46, c_col, 2)
        _text(image, f"{target['width_m'] * 1000:.0f} x "
                     f"{target['height_m'] * 1000:.0f} mm",
              W - 180, H - 24, 0.42, _INK, 1)
        _text(image, f"{target['z']:.2f} m", W - 78, H - 24, 0.42, _INK, 1)

    if stopped:
        _panel(image, 0, H // 2 - 26, W, 52, 0.75)
        _text(image, 'EMERGENCY STOP LATCHED', W // 2 - 150, H // 2 + 8,
              0.80, _BAD, 2)


def draw_candidate(image, c, colour, thick=2):
    """Corner-bracket box + compact label — reads far better on a projector
    than a full rectangle with a debug string on it."""
    x, y, w, h = c['bbox']
    k = max(8, min(w, h) // 4)          # bracket arm length
    for (cx, sx) in ((x, 1), (x + w, -1)):
        for (cy, sy) in ((y, 1), (y + h, -1)):
            cv2.line(image, (cx, cy), (cx + sx * k, cy), colour, thick,
                     cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx, cy + sy * k), colour, thick,
                     cv2.LINE_AA)
    if thick >= 3:                      # only label the selected target
        col = c.get('colour')
        tag = (col.upper() if col in ('ryobi', 'pink')
               else c['kind'].upper())
        ty = max(14, y - 8)
        (tw, th), _ = cv2.getTextSize(tag, _FONT, 0.44, 1)
        _panel(image, x, ty - th - 4, tw + 10, th + 8, 0.55)
        _text(image, tag, x + 5, ty - 2, 0.44, colour, 1)


def main():
    if DEBUG:
        with open(_LOG_PATH, 'a') as f:
            f.write('\n==== NEW RUN ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ====\n')
    supervisor = SafetySupervisor()
    actuator = BaseActuator()
    pipeline, frame_queue, det_queue, labels = build_pipeline()
    pipeline.start()

    # Production helpers (all no-ops unless enabled in config.py).
    grabber = GrabVerifier(log)
    watchdog = Watchdog(log)
    metrics = SessionMetrics(log)

    # ---- Startup self-check: verify the rig BEFORE the arm can move --------
    if SELFCHECK_ENABLED:
        first_frame_ok = False
        for _ in range(80):                  # up to ~8s for the camera to boot
            if frame_queue.tryGet() is not None:
                first_frame_ok = True
                break
            time.sleep(0.1)
        if not SelfCheck(log).run(actuator, first_frame_ok):
            if SELFCHECK_ABORT_ON_FAIL:
                log("SELFCHECK aborted the run — fix the problems above.")
                actuator.close()
                try:
                    pipeline.stop()
                except Exception:
                    pass
                return

    print("Enabling torque in 3s — keep a hand near the power switch...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    actuator.enable()
    log(f"START rest_pose={actuator.rest_pose} gain={CLOSED_LOOP_GAIN} "
        f"deadband={IMAGE_DEADBAND_PX}px latency={CAMERA_LATENCY_S}s "
        f"drop_frac={DROP_POSE_FRAC} gripper={GRIPPER_OPEN_RAW}/{GRIPPER_CLOSED_RAW}")
    test = actuator.grip_test()
    if test is None:
        log("GRIPPER: not calibrated — grips will be skipped")
    else:
        cmd, got = test
        if abs(cmd - got) > 80:
            log(f"GRIPPER TEST FAILED: commanded {cmd}, servo reads {got}. "
                f"The gripper is NOT responding — check its cable/power, or "
                f"re-run calibrate_full.py part C. Grabs won't hold until fixed!")
        else:
            log(f"GRIPPER TEST OK — opened to {got}")
    # Drive to the CANONICAL rest pose so every run starts identically
    # (rest comes from the calibration, not from wherever the arm was left).
    log(f"Moving to rest pose {actuator.rest_pose} + base forward centre...")
    actuator.go_rest()
    time.sleep(2.0)
    print("ENTER = lock target, then ENTER = confirm box. X = grab only.")

    centre_x = CAMERA_WIDTH // 2
    phase = 'IDLE'    # IDLE SEL_BOX CENTRE STAGE1 VERIFY FULL GRIPPED
    #                   CARRY PLACE_CENTRE PLACE_REACH DROP HOME
    phase_start = 0.0
    current = None          # highlighted target in IDLE
    box_pick = None         # highlighted box in SEL_BOX
    box_sel = None          # confirmed box {'base_goal', 'z'}
    goal_raw = None
    locked_point = None
    locked_z = None
    z_hist = deque(maxlen=Z_MEDIAN_SAMPLES)
    full_pose = None
    stage_goals = None
    arrived_time = None
    grip_time = None
    drop_pose = None
    low_pose = None
    drop_time = None
    carry_step = 0
    last_check = 0.0
    last_corr = 0.0
    last_enter = 0.0
    last_dets = []
    last_det_time = 0.0
    stats = {}
    last_stats_log = time.monotonic()
    pos_hist = deque(maxlen=12)
    centre_ok = 0           # consecutive centred frames (CENTRE phase)
    verify_ok = 0           # consecutive aligned frames (VERIFY phase)
    assoc_base = None       # base raw at the last confirmed target sighting
    force_lying = False     # L key: force the taught top-down grab
    orient_dbg = ''         # last orientation measurements (for the log)
    window = 'SO-101 Autonomous Battery Sorter'

    mouse_click = {'pt': None}

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def present_at(t):
        if not pos_hist:
            return None
        return min(pos_hist, key=lambda p: abs(p[0] - t))[1]

    def base_speed():
        if len(pos_hist) < 2:
            return 0.0
        dt = pos_hist[-1][0] - pos_hist[0][0]
        return abs(pos_hist[-1][1] - pos_hist[0][1]) / dt if dt > 0 else 0.0

    def reset_cycle():
        nonlocal phase, goal_raw, locked_point, locked_z, full_pose, stage_goals
        nonlocal arrived_time, box_sel, box_pick, drop_pose, low_pose, drop_time
        nonlocal carry_step, grip_time, centre_ok, verify_ok, assoc_base
        nonlocal cur_slot, locked_colour, auto_last_seen, cycle_end_t
        nonlocal scanning, scan_target, scan_sweeps
        if scanning:
            # Never leave the base on the slow scan profile.
            scanning = False
            scan_target = None
            scan_sweeps = 0
            try:
                actuator.set_base_profile(GOAL_MOVE_SPEED, GOAL_MOVE_ACC)
            except ActuatorError:
                pass
        cur_slot = None
        locked_colour = None
        auto_last_seen = time.monotonic()   # give detection time post-cycle
        cycle_end_t = time.monotonic()
        low_pose = None
        centre_ok = 0
        verify_ok = 0
        assoc_base = None
        phase = 'IDLE'
        goal_raw = None
        locked_point = None
        locked_z = None
        full_pose = None
        stage_goals = None
        arrived_time = None
        box_sel = None
        box_pick = None
        drop_pose = None
        drop_time = None
        grip_time = None
        carry_step = 0
        z_hist.clear()
        pos_hist.clear()

    # Sleeve mode only if taught AND explicitly enabled in config —
    # default is click-to-place anywhere. Slot mode (5-slot holster) takes
    # priority once taught with calibrate_slots.py.
    place_mode = USE_PLACE_SEQUENCE and PLACE_INSERT is not None
    slots_mode = USE_SLOTS and bool(SLOTS)
    buckets_mode = USE_BUCKETS and BUCKET_GREEN and BUCKET_RED
    slot_full = [False] * (len(SLOTS) if SLOTS else 0)
    cur_slot = None
    seq_base = seq_hover = seq_insert = None
    prev_pts = []           # last frame's candidate positions (stability)
    auto_mode = False       # ENTER in bucket mode: sort ALL batteries
    auto_last_seen = 0.0
    locked_colour = None    # 'ryobi' / 'pink' / 'unknown' of the locked target
    colour_votes = {'ryobi': 0, 'pink': 0}   # per-frame votes while centring
    orient_votes = {'lying': 0, 'upright': 0}  # orientation votes
    locked_lying = False    # grab with the taught top-down (lying) poses?
    cycle_end_t = 0.0       # when the last cycle finished (settle before relock)
    # ---- IDLE scan state ----
    scanning = False        # sweeping the base looking for out-of-view targets
    scan_dir = SCAN_FIRST_DIR   # +1 = right; reset to this at every scan start
    scan_next_t = 0.0       # when the next scan step may fire
    scan_target = None      # raw position the base is stepping toward
    scan_sweeps = 0         # completed passes with nothing found
    n_ryobi = n_pink = 0    # tally shown on the HUD
    fps_ema = None          # smoothed frame interval
    last_frame_t = time.monotonic()

    # Sweep only the arc BETWEEN the buckets. Sweeping across them made the
    # robot detect its own buckets as targets and try to pick them up.
    scan_lo, scan_hi = SCAN_MIN_RAW, SCAN_MAX_RAW
    if buckets_mode:
        _bb = sorted((BUCKET_GREEN['base'], BUCKET_RED['base']))
        scan_lo = max(scan_lo, _bb[0] + BUCKET_KEEPOUT_RAW)
        scan_hi = min(scan_hi, _bb[1] - BUCKET_KEEPOUT_RAW)
        if scan_hi - scan_lo < SCAN_STEP_RAW:
            # Buckets too close together to leave a usable arc — don't sweep.
            scan_lo = scan_hi = BASE_FORWARD_RAW
            print(f"SCAN disabled: no clear arc between buckets "
                  f"{_bb[0]} and {_bb[1]}")
        else:
            print(f"SCAN arc {scan_lo}..{scan_hi} raw "
                  f"(buckets {_bb[0]}/{_bb[1]} excluded)")

    def near_bucket(raw):
        """True when the base points at (or near) a bucket — never lock there."""
        if not buckets_mode or raw is None:
            return False
        return any(abs(raw - b['base']) < BUCKET_KEEPOUT_RAW
                   for b in (BUCKET_GREEN, BUCKET_RED))

    def go_centre():
        """Start the grab: open gripper, first base move, enter CENTRE."""
        nonlocal goal_raw, centre_ok, phase, phase_start, assoc_base, colour_votes
        nonlocal orient_votes, locked_lying
        colour_votes = {'ryobi': 0, 'pink': 0}
        orient_votes = {'lying': 0, 'upright': 0}
        locked_lying = False
        track_reset()
        actuator.grip_open()
        err = locked_point[0] - centre_x
        present_raw = actuator.read_position()
        pos_hist.clear()
        pos_hist.append((time.monotonic(), present_raw))
        assoc_base = present_raw
        goal_raw = actuator.move_to(
            present_raw + BASE_DIRECTION_SIGN * err * RAW_PER_PIXEL * CLOSED_LOOP_GAIN)
        centre_ok = 0
        phase, phase_start = 'CENTRE', time.monotonic()
        metrics.attempt()
        log(f"GO — centring on target, goal={goal_raw}")

    def joints_arrived(goals, tol=ARM_ARRIVE_TOL_RAW):
        try:
            return all(abs(actuator.read_position(s) - g) <= tol
                       for s, g in goals.items())
        except ActuatorError:
            return False

    # ---- Base tracker: smoothed reference + rate-limited commands ----------
    _trk = {'es': None, 'cmd_t': 0.0}

    def track_reset():
        _trk['es'], _trk['cmd_t'] = None, 0.0

    def track_goal(present_raw, err, now):
        """Return the base goal for pixel error `err`, or None to HOLD.

        The pixel error is noisy, so it is EMA-smoothed into a stable
        reference before use, and new goals are rate-limited so the servo
        can actually complete each move. Together these remove the
        side-to-side twitch without slowing the initial lock-on."""
        es = float(err) if _trk['es'] is None else (
            TRACK_ERR_SMOOTH * _trk['es'] + (1.0 - TRACK_ERR_SMOOTH) * err)
        _trk['es'] = es
        # Centred (on the smoothed error) — hold, no churn.
        if abs(es) <= TRACK_SETTLE_DEADBAND_PX:
            return None
        # Rate limit: let the previous move execute before commanding again.
        if now - _trk['cmd_t'] < TRACK_MIN_CMD_INTERVAL_S:
            return None
        _trk['cmd_t'] = now
        step = BASE_DIRECTION_SIGN * CLOSED_LOOP_GAIN * es * RAW_PER_PIXEL
        step = max(-TRACK_MAX_STEP_RAW, min(TRACK_MAX_STEP_RAW, step))
        return present_raw + step

    DELIVERY_TOL = 60   # delivery moves don't need grab precision — a loose
                        # tolerance means no timeout stalls in the auto loop

    try:
        while pipeline.isRunning():
            now = time.monotonic()

            # Watchdog: a non-idle phase running far too long has hung. Recover
            # to rest and resume rather than freezing mid-run.
            if watchdog.phase_hung(phase, phase_start, now):
                log(f"WATCHDOG — phase {phase} hung "
                    f"{now - phase_start:.0f}s, recovering to rest")
                try:
                    actuator.grip_open()
                    actuator.set_arm_profile(ARM_MOVE_SPEED, ARM_MOVE_ACC)
                    actuator.go_rest()
                except Exception:
                    pass
                reset_cycle()
                auto_mode = False
                continue

            group = frame_queue.tryGet()

            if det_queue is not None:
                msg = det_queue.tryGet()
                while msg is not None:
                    last_dets = msg.detections
                    last_det_time = now
                    msg = det_queue.tryGet()

            if group is None:
                supervisor.is_stable()
                if supervisor.stop_latched:
                    actuator.emergency_stop()
            else:
                supervisor.observe_frame()
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0

                # PERFORMANCE: only run detection in the phases that consume
                # it. During carry/place/return nothing reads `candidates`,
                # and the contour + depth work was costing a third of the
                # frame budget for nothing. This is the single biggest
                # source of the lag during a cycle.
                detect_now = phase in DETECT_PHASES
                # Run BOTH detectors and FUSE: an object seen by YOLO and the
                # depth detector at once is the strongest signal.
                cand_yolo = []
                if (detect_now and USE_YOLO
                        and now - last_det_time <= 0.5 and last_dets):
                    cand_yolo = candidates_from_yolo(
                        last_dets, depth, CAMERA_WIDTH, CAMERA_HEIGHT, labels, stats)
                cand_cls = (detect_candidates(image, depth, stats)
                            if (detect_now and CLASSICAL_FALLBACK) else [])
                candidates = list(cand_yolo)
                for c in cand_cls:
                    dup = next((y for y in candidates
                                if abs(y['u'] - c['u']) <= MERGE_RADIUS_PX
                                and abs(y['v'] - c['v']) <= MERGE_RADIUS_PX), None)
                    if dup is None:
                        candidates.append(c)
                    else:
                        dup['source'] = 'fused'
                candidates.sort(key=lambda c: c['z'])   # CLOSEST FIRST
                # Temporal stability: must also have been seen last frame —
                # background flicker never survives this.
                for c in candidates:
                    c['stable'] = any(abs(c['u'] - pu) <= STABLE_RADIUS_PX
                                      and abs(c['v'] - pv) <= STABLE_RADIUS_PX
                                      for pu, pv in prev_pts)
                prev_pts = [(c['u'], c['v']) for c in candidates]
                targets = [c for c in candidates if c['kind'] == 'target']
                boxes = [c for c in candidates if c['kind'] == 'box']
                for c in targets:
                    # Orientation. Signal order: a two-class YOLO label if one
                    # exists, else the top-edge residual (top_m), else aspect.
                    # The old 'spread' path is dead — it measured the desk.
                    spread = c.get('spread_m')
                    proud = c.get('proud_m')
                    lbl = (c.get('label') or '').lower()
                    if any(s in lbl for s in YOLO_LYING_LABELS):
                        # A two-class custom model told us outright.
                        c['lying'] = True
                        c['_orient_src'] = 'yolo'
                    elif any(s in lbl for s in YOLO_UPRIGHT_LABELS):
                        c['lying'] = False
                        c['_orient_src'] = 'yolo'
                    elif c.get('top_m') is not None:
                        # PRIMARY: how far the object's top edge stands out of
                        # the desk at the same image rows. Standing batteries
                        # read 60-115mm, lying ones ~20-30mm.
                        c['lying'] = c['top_m'] < UPRIGHT_MIN_TOP_M
                        c['_orient_src'] = 'top'
                    elif spread is not None:
                        # Legacy depth fallback — measured the desk, not the
                        # object. Kept only so nothing crashes if re-enabled.
                        c['lying'] = spread < UPRIGHT_MIN_SPREAD_M
                        c['_orient_src'] = 'depth'
                    else:
                        # No depth: only a clearly TALL silhouette proves
                        # upright, otherwise assume lying.
                        c['lying'] = not (c['height_m']
                                          > c['width_m'] * UPRIGHT_MIN_TALL)
                        c['_orient_src'] = 'aspect'
                    _tm = c.get('top_m')
                    c['_orient_dbg'] = (
                        f"src={c['_orient_src']}"
                        + (f":{lbl}" if c['_orient_src'] == 'yolo' else '')
                        + f" top={'%.0fmm' % (_tm * 1000) if _tm is not None else '?'}"
                        f" h/w={c['height_m'] / max(1e-6, c['width_m']):.2f}")
                if buckets_mode:
                    for c in targets:
                        col = classify_colour(image, c['bbox'])
                        # Pink batteries are ~30% smaller: anything longer
                        # than PINK_MAX_LEN_M cannot be pink.
                        if (col == 'pink'
                                and max(c['width_m'], c['height_m']) > PINK_MAX_LEN_M):
                            col = 'ryobi'
                        c['colour'] = col

                if DEBUG and now - last_stats_log >= DEBUG_STATS_PERIOD_S:
                    log(f"DETECT {stats} targets={len(targets)} boxes={len(boxes)} "
                        f"phase={phase}")
                    stats = {}
                    last_stats_log = now

                cv2.line(image, (centre_x, 0), (centre_x, CAMERA_HEIGHT), (255, 255, 0), 1)
                status, status_colour = '', (0, 255, 255)

                if phase in ('CENTRE', 'VERIFY', 'PLACE_CENTRE'):
                    try:
                        pos_hist.append((now, actuator.read_position()))
                    except ActuatorError:
                        pass

                # PREDICTIVE re-association: from how far the base has
                # rotated since the last sighting (and the measured raw/px
                # mapping) we know where the ORIGINAL target must be now.
                # Only detections in a tight window around that predicted
                # spot are accepted — a lookalike sitting elsewhere cannot
                # steal the lock.
                match = None
                if phase in ('CENTRE', 'VERIFY') and candidates and locked_point:
                    present_lat = (present_at(now - CAMERA_LATENCY_S)
                                   or (pos_hist[-1][1] if pos_hist else None))
                    pred_u = locked_point[0]
                    if assoc_base is not None and present_lat is not None:
                        pred_u = locked_point[0] - (BASE_DIRECTION_SIGN
                                                    * (present_lat - assoc_base)
                                                    / RAW_PER_PIXEL)
                    radius = ASSOC_RADIUS_PX + min(60.0, base_speed() * 0.05)
                    near = [c for c in candidates
                            if abs(c['z'] - locked_z) <= ASSOC_Z_TOL_M]
                    if near:
                        match = min(near, key=lambda c:
                                    (c['u'] - pred_u) ** 2
                                    + (c['v'] - locked_point[1]) ** 2)
                        dist = ((match['u'] - pred_u) ** 2
                                + (match['v'] - locked_point[1]) ** 2) ** 0.5
                        if dist > radius:
                            match = None
                    cv2.drawMarker(image, (int(pred_u), int(locked_point[1])),
                                   (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
                    if match is not None:
                        locked_point = (int(match['u']), int(match['v']))
                        locked_z = match['z']
                        z_hist.append(match['z'])
                        assoc_base = present_lat
                        draw_candidate(image, match, (0, 255, 0))
                        # Colour VOTING while centring: many frames decide
                        # the bucket, not one — mis-sorts need a majority.
                        if buckets_mode and phase == 'CENTRE':
                            colv = classify_colour(image, match['bbox'])
                            _len = max(match['width_m'], match['height_m'])
                            # SIZE beats colour both ways — it is immune to
                            # the green mat, lighting and which face shows —
                            # but ONLY for a plausibly battery-sized blob.
                            if (USE_SIZE_OVERRIDE
                                    and PLAUSIBLE_LEN_M[0] <= _len
                                    <= PLAUSIBLE_LEN_M[1]):
                                if colv == 'pink' and _len > PINK_MAX_LEN_M:
                                    colv = 'ryobi'   # too long to be pink
                                elif colv == 'ryobi' and _len < RYOBI_MIN_LEN_M:
                                    colv = 'pink'    # too short to be a Ryobi
                            if colv != 'unknown':
                                colour_votes[colv] += 1
                        if phase == 'CENTRE':
                            # Orientation vote: lying batteries get the taught
                            # vertical top-down grab instead of the side grab.
                            # A YOLO class label is trustworthy, so it votes
                            # UNBIASED. The depth guess needs LYING_VOTE_BIAS
                            # to offset how often it reads a lying battery as
                            # upright — and is off by default anyway.
                            bias = (1 if match.get('_orient_src') == 'yolo'
                                    else LYING_VOTE_BIAS)
                            if match.get('lying'):
                                orient_votes['lying'] += bias
                            else:
                                orient_votes['upright'] += 1
                            orient_dbg = match.get('_orient_dbg', '')
                if locked_point and phase not in ('IDLE', 'SEL_BOX'):
                    cv2.drawMarker(image, locked_point, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

                if supervisor.stop_latched and phase != 'IDLE':
                    status, status_colour = 'STOP LATCHED — RESTART TO RE-RUN', (0, 0, 255)

                elif phase == 'IDLE':
                    # Priority: stable, classified-as-a-battery, fused,
                    # then NEAREST first.
                    ordered = sorted(targets, key=lambda c: (
                        not c.get('stable', False),
                        c.get('colour') == 'unknown',
                        c['source'] != 'fused',
                        c['z']))
                    # Sticky highlight: keep the same target while it's still
                    # around, instead of flickering between candidates.
                    if ordered:
                        if current is not None:
                            near = min(ordered, key=lambda c:
                                       (c['u'] - current['u']) ** 2
                                       + (c['v'] - current['v']) ** 2)
                            d = ((near['u'] - current['u']) ** 2
                                 + (near['v'] - current['v']) ** 2) ** 0.5
                            current = near if d <= 60 else ordered[0]
                        else:
                            current = ordered[0]
                        auto_last_seen = now
                    else:
                        current = None

                    # AUTO SORT: lock the nearest battery and go, repeatedly.
                    if auto_mode and buckets_mode:
                        # Only lock a STABLE candidate, and only once the
                        # base has settled from the previous cycle — locking
                        # mid-swing produced phantom targets.
                        if (current is not None
                                and not current.get('stable', False)):
                            current = None
                        if now - cycle_end_t < AUTO_RELOCK_DELAY_S:
                            current = None
                        # Never auto-lock while aimed at a bucket — the buckets
                        # are battery-shaped and were being picked up.
                        if current is not None:
                            try:
                                # Only refuse if the base is CURRENTLY aimed at
                                # a bucket. The old "where would centring take
                                # us" test also rejected real batteries sitting
                                # in front of a bucket — a battery at u=470
                                # implied a swing to 2366, inside the green
                                # bucket's keep-out, so it was never picked up.
                                # The colour-evidence gate below is what stops
                                # the bucket ITSELF being locked.
                                if near_bucket(actuator.read_position()):
                                    current = None
                            except ActuatorError:
                                pass
                        # Skip a spot that just failed to grab too many times,
                        # so a stuck/ungraspable object can't loop forever.
                        if current is not None and grabber.cooling(
                                current['u'], current['v'], now):
                            current = None
                        if current is not None and scanning:
                            # Found one mid-sweep: stop scanning, hand the base
                            # back to the fast tracking profile before we chase.
                            scanning = False
                            scan_target = None
                            scan_sweeps = 0
                            actuator.set_base_profile(GOAL_MOVE_SPEED,
                                                      GOAL_MOVE_ACC)
                            log("SCAN — target found, resuming sort")
                        if current is not None:
                            locked_point = (int(current['u']), int(current['v']))
                            locked_z = current['z']
                            col, rc, pc = classify_colour(
                                image, current['bbox'], with_counts=True)
                            locked_colour = (col if col != 'unknown'
                                             else current.get('colour', 'unknown'))
                        # No colour evidence at all = not a battery. These were
                        # the phantom locks near the buckets. Done as its own
                        # step so the lock code below is skipped entirely —
                        # clearing `current` mid-block left the rest of the
                        # body dereferencing None.
                        if (current is not None and locked_colour == 'unknown'):
                            log(f"SKIP phantom — no colour evidence "
                                f"(rc={rc} pc={pc}) at z={locked_z:.3f}")
                            locked_point, locked_z = None, None
                            locked_colour = None
                            current = None

                        if current is not None:
                            # SIZE OVERRIDE — mat-independent, beats colour.
                            _len = max(current['width_m'], current['height_m'])
                            _sz, _size_ruled = '', False
                            if (USE_SIZE_OVERRIDE
                                    and PLAUSIBLE_LEN_M[0] <= _len
                                    <= PLAUSIBLE_LEN_M[1]):
                                if locked_colour == 'pink' and _len > PINK_MAX_LEN_M:
                                    locked_colour, _sz = 'ryobi', ' SIZE->ryobi'
                                    _size_ruled = True
                                elif (locked_colour != 'pink'
                                      and _len < RYOBI_MIN_LEN_M):
                                    locked_colour, _sz = 'pink', ' SIZE->pink'
                                    _size_ruled = True
                            elif USE_SIZE_OVERRIDE:
                                _sz = ' (len implausible — size ignored)'
                            z_hist.clear()
                            z_hist.append(locked_z)
                            log(f"AUTO TARGET ({locked_colour}) z={locked_z:.3f} "
                                f"len={_len * 1000:.0f}mm{_sz} "
                                f"dark_px={rc - pc if rc > pc else rc} "
                                f"ryobi_px={rc} pink_px={pc} "
                                f"src={current['source']}")
                            go_centre()
                            # Seed the votes with strong lock-time evidence —
                            # a ryobi_px=2390 lock once lost to 6 weak
                            # mid-swing pink votes.
                            #
                            # Seed only the WINNER. Seeding both on absolute
                            # thresholds made rc=1443/pc=332 an 8-8 tie, and
                            # the tie sent a Ryobi to the RED bucket.
                            # Seed whatever the lock concluded (colour AND the
                            # size override). A size verdict is measurement,
                            # not appearance, so it seeds heavily enough that
                            # per-frame colour votes cannot overturn it.
                            if _size_ruled and locked_colour in ('pink', 'ryobi'):
                                colour_votes[locked_colour] += 40
                            elif rc >= 800 or pc >= 300:
                                if pc > rc * PINK_VS_RYOBI_W:
                                    colour_votes['pink'] += 8
                                else:
                                    colour_votes['ryobi'] += 8
                            if current.get('lying'):
                                orient_votes['lying'] += 2
                        elif now - auto_last_seen >= SCAN_START_S and SCAN_ENABLED:
                            # Nothing in view. Rather than give up, sweep the
                            # base — the camera only covers ~69 deg, so a
                            # battery just outside it is invisible, not absent.
                            if not scanning:
                                scanning = True
                                scan_sweeps = 0
                                scan_next_t = 0.0
                                # Always set off to the RIGHT. Without this the
                                # direction carried over from the previous
                                # sweep and it often started leftwards.
                                scan_dir = SCAN_FIRST_DIR
                                scan_target = None
                                actuator.set_base_profile(SCAN_SPEED, SCAN_ACC)
                                log("SCAN — sweeping RIGHT first for targets "
                                    "outside the view")
                            if now >= scan_next_t:
                                try:
                                    here = actuator.read_position()
                                except ActuatorError:
                                    here = scan_target or BASE_FORWARD_RAW
                                base_now = (scan_target if scan_target is not None
                                            else here)
                                # Clamp into the bucket-free arc first, in case
                                # the last cycle left us parked at a bucket.
                                base_now = max(scan_lo, min(scan_hi, base_now))
                                nxt = base_now + scan_dir * SCAN_STEP_RAW
                                if nxt >= scan_hi:
                                    nxt, scan_dir = scan_hi, -1
                                    scan_sweeps += 1
                                elif nxt <= scan_lo:
                                    nxt, scan_dir = scan_lo, 1
                                    scan_sweeps += 1
                                scan_target = nxt
                                actuator.move_to(nxt)
                                # Wait for the move AND for the image to settle,
                                # otherwise we detect on motion-blurred frames.
                                scan_next_t = now + SCAN_SETTLE_S + abs(
                                    SCAN_STEP_RAW) / max(1, SCAN_SPEED)
                            if scan_sweeps >= SCAN_MAX_SWEEPS:
                                scanning = False
                                scan_target = None
                                auto_mode = False
                                actuator.set_base_profile(GOAL_MOVE_SPEED,
                                                          GOAL_MOVE_ACC)
                                actuator.move_to(BASE_FORWARD_RAW)
                                log("AUTO SORT COMPLETE — swept the full arc, "
                                    "no batteries found")
                        elif now - auto_last_seen >= AUTO_IDLE_TIMEOUT_S:
                            auto_mode = False
                            log("AUTO SORT COMPLETE — no more batteries in view")
                    for c in targets:
                        if buckets_mode:
                            col = (_OK if c.get('colour') == 'ryobi'
                                   else _HOT if c.get('colour') == 'pink'
                                   else _WARN)
                        else:
                            col = _WARN if c is current else _DIM
                        draw_candidate(image, c, col, 3 if c is current else 1)
                    for c in boxes:
                        draw_candidate(image, c, (255, 0, 255))
                    if auto_mode and buckets_mode and scanning:
                        status = 'Scanning for batteries outside the view'
                    elif auto_mode and buckets_mode:
                        status = 'Seeking next battery — closest first'
                    elif buckets_mode:
                        status = 'ENTER to auto-sort everything, or click one battery'
                    else:
                        status = ('Click a battery, or ENTER for the highlighted one'
                                  if current else 'Click a battery to begin')
                    status_colour = _WARN

                elif phase == 'SEL_BOX':
                    # Target frozen (drawn as cross); pick the box (sticky).
                    cv2.drawMarker(image, locked_point, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                    if boxes:
                        if box_pick is not None:
                            near = min(boxes, key=lambda c:
                                       (c['u'] - box_pick['u']) ** 2
                                       + (c['v'] - box_pick['v']) ** 2)
                            d = ((near['u'] - box_pick['u']) ** 2
                                 + (near['v'] - box_pick['v']) ** 2) ** 0.5
                            box_pick = near if d <= 100 else max(boxes, key=lambda c: c['area'])
                        else:
                            box_pick = max(boxes, key=lambda c: c['area'])
                    else:
                        box_pick = None
                    for c in boxes:
                        draw_candidate(image, c, (255, 0, 255),
                                       3 if c is box_pick else 1)
                    status = ('CLICK THE BOX (ENTER = highlighted, X = grab only)'
                              if box_pick else 'CLICK THE BOX (X = grab only)')
                    status_colour = (255, 0, 255)

                elif phase == 'CENTRE':
                    def _stage(z_val, tag):
                        nonlocal full_pose, stage_goals, phase, phase_start
                        nonlocal locked_colour, locked_lying
                        if buckets_mode:
                            # Colour priority: VOTES > lock-time colour >
                            # blind patch. The blind patch samples a stale
                            # point and must never override real data (it
                            # once flipped a pink_px=2996 lock to ryobi).
                            rv, pv = colour_votes['ryobi'], colour_votes['pink']
                            if rv or pv:
                                if rv != pv:
                                    locked_colour = 'pink' if pv > rv else 'ryobi'
                                else:
                                    # TIE -> ryobi, always. Pink is the class
                                    # that historically false-positives (white
                                    # text, warm highlights), so it has to WIN
                                    # outright, not inherit a tie.
                                    locked_colour = 'ryobi'
                                log(f"COLOUR VOTE ryobi={rv} pink={pv} "
                                    f"-> {locked_colour}")
                            elif locked_colour in (None, 'unknown'):
                                px, py = locked_point
                                locked_colour, rc, pc = classify_colour(
                                    image, (px - 30, py - 40, 60, 80),
                                    with_counts=True)
                                log(f"COLOUR (blind) ryobi_px={rc} pink_px={pc} "
                                    f"-> {locked_colour}")
                            else:
                                log(f"COLOUR kept from lock -> {locked_colour}")
                        # Orientation. The depth-based vote is UNRELIABLE (see
                        # AUTO_ORIENT in config.py) so by default only the
                        # operator's L key selects the top-down grab.
                        # LYING must win by a MARGIN, not a hair. A wrong
                        # top-down grab drives the gripper into a standing
                        # battery; a wrong side-grab just misses.
                        auto_lying = (AUTO_ORIENT
                                      and orient_votes['lying']
                                      >= orient_votes['upright']
                                      + LYING_VOTE_MARGIN)
                        locked_lying = (LYING_POINTS is not None
                                        and (force_lying or auto_lying))
                        if force_lying and LYING_POINTS is not None:
                            log("ORIENTATION forced LYING by operator (L)")
                        if orient_votes['lying'] or orient_votes['upright']:
                            note = ''
                            if locked_lying and LYING_POINTS is None:
                                note = ' (no lying calibration — run calibrate_lying.py!)'
                            elif not AUTO_ORIENT and not force_lying:
                                note = ' (auto-orient OFF — press L for top-down)'
                            elif AUTO_ORIENT and not force_lying:
                                note = (f" (lying needs +{LYING_VOTE_MARGIN};"
                                        f" L forces top-down)")
                            log(f"ORIENTATION lying={orient_votes['lying']} "
                                f"upright={orient_votes['upright']} "
                                f"[{orient_dbg}] -> "
                                f"{'LYING top-down' if locked_lying else 'upright'}{note}")
                        t, pose = actuator.reach_pose(
                            z_val, LYING_POINTS if locked_lying else None)
                        # GRAB_POSE_FRAC < 1 stops the descent early: the
                        # gripper closes higher up the battery (its top).
                        full_pose = {
                            sid: int(round(actuator.rest_pose[sid]
                                           + GRAB_POSE_FRAC
                                           * (pose[sid] - actuator.rest_pose[sid])))
                            for sid in pose}
                        if not locked_lying:
                            # Level the gripper: enter PARALLEL to the floor.
                            full_pose[WRIST_ID] += UPRIGHT_WRIST_TRIM_RAW
                        stage_goals = {
                            sid: int(round(actuator.rest_pose[sid]
                                           + REACH_STAGE_FRAC
                                           * (full_pose[sid] - actuator.rest_pose[sid])))
                            for sid in full_pose}
                        actuator.move_joints(stage_goals)
                        phase, phase_start = 'STAGE1', now
                        log(f"STAGE1 ({tag}) t={t:.2f} goals={stage_goals} z={z_val:.3f}")
                    if match is not None:
                        err = match['u'] - centre_x
                        present_raw = (present_at(now - CAMERA_LATENCY_S)
                                       or actuator.read_position())
                        if abs(err) <= IMAGE_DEADBAND_PX and base_speed() < 60:
                            centre_ok += 1
                            status = f'CONFIRMING CENTRE {centre_ok}/{CENTRE_STABLE_FRAMES}'
                            if centre_ok >= CENTRE_STABLE_FRAMES:
                                z_med = (sorted(z_hist)[len(z_hist) // 2]
                                         if z_hist else locked_z)
                                _stage(z_med, 'confirmed')
                        else:
                            centre_ok = 0
                            new_goal = track_goal(present_raw, err, now)
                            if new_goal is not None and (
                                    goal_raw is None
                                    or abs(new_goal - goal_raw) > GOAL_UPDATE_MIN_RAW):
                                goal_raw = actuator.move_to(new_goal)
                                log(f"TRACK err={err:+d}px goal={goal_raw}")
                            status = f'TRACKING — err {err:+d}px'
                    else:
                        centre_ok = 0
                        status, status_colour = 'TARGET LOST — HOLDING', (0, 165, 255)
                        # Clicked point never re-detected: once the base has
                        # reached its goal and sat there, proceed blind.
                        try:
                            at_goal = (goal_raw is not None
                                       and abs(actuator.read_position() - goal_raw) <= 10)
                        except ActuatorError:
                            at_goal = False
                        if at_goal and now - phase_start >= CENTRE_BLIND_S:
                            _stage(locked_z, 'blind — click point')

                elif phase == 'STAGE1':
                    status, status_colour = 'ARM MOVING (STAGE 1 OF 2)', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if joints_arrived(stage_goals) or now - phase_start >= STAGE_TIMEOUT_S:
                            phase, phase_start = 'VERIFY', now
                            last_corr = 0.0
                            verify_ok = 0
                            log("VERIFY — re-checking target position")

                elif phase == 'VERIFY':
                    status, status_colour = 'VERIFYING AIM', (0, 255, 0)
                    aligned = False
                    if match is not None:
                        err = match['u'] - centre_x
                        # Slightly wider acceptance than the grab deadband —
                        # the log showed 5px errors looping to timeout here.
                        if abs(err) <= IMAGE_DEADBAND_PX + 2:
                            verify_ok += 1
                            aligned = verify_ok >= VERIFY_ALIGN_FRAMES
                            status = f'VERIFYING {verify_ok}/{VERIFY_ALIGN_FRAMES}'
                        elif now - last_corr >= VERIFY_CORR_INTERVAL_S:
                            verify_ok = 0
                            last_corr = now
                            present_raw = (present_at(now - CAMERA_LATENCY_S)
                                           or actuator.read_position())
                            goal_raw = actuator.move_to(
                                present_raw + BASE_DIRECTION_SIGN * err
                                * RAW_PER_PIXEL * CLOSED_LOOP_GAIN)
                            log(f"VERIFY_CORR err={err:+d}px goal={goal_raw}")
                    if aligned or now - phase_start >= VERIFY_S:
                        if z_hist:
                            z_med = sorted(z_hist)[len(z_hist) // 2]
                            _, pose = actuator.reach_pose(
                                z_med, LYING_POINTS if locked_lying else None)
                            full_pose = {
                                sid: int(round(actuator.rest_pose[sid]
                                               + GRAB_POSE_FRAC
                                               * (pose[sid] - actuator.rest_pose[sid])))
                                for sid in pose}
                            if not locked_lying:
                                full_pose[WRIST_ID] += UPRIGHT_WRIST_TRIM_RAW
                        actuator.move_joints(full_pose)
                        phase, phase_start = 'FULL', now
                        arrived_time = None
                        log(f"FULL goals={full_pose} "
                            f"({'aligned' if aligned else 'verify timeout'})")

                elif phase == 'FULL':
                    status, status_colour = 'ARM COMPLETING REACH', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if arrived_time is None and joints_arrived(full_pose):
                            arrived_time = now
                            log("ARM ARRIVED at reach pose")
                        settled = arrived_time and now - arrived_time >= GRIP_DELAY_S
                        timed_out = now - phase_start >= GRIP_TIMEOUT_S
                        if settled or timed_out:
                            if timed_out and not settled:
                                log("GRIP timeout — closing without full arrival")
                            gripped_at = actuator.grip_close()
                            grip_time = now
                            phase = 'GRIPPED'
                            log(f"GRIP closed at raw {gripped_at}")
                            # Empty grip? Don't carry air to a bucket — reopen,
                            # return to rest, and let auto-sort re-attempt.
                            gv = grabber.verdict(gripped_at, locked_point, now)
                            if gv != 'proceed':
                                metrics.empty()
                                actuator.grip_open()
                                actuator.wait_released(timeout=2.0)
                                actuator.set_arm_profile(ARM_MOVE_SPEED,
                                                         ARM_MOVE_ACC)
                                actuator.go_rest()
                                log(f"GRAB {gv} — recovered to rest, "
                                    "re-attempting")
                                reset_cycle()

                elif phase == 'GRIPPED':
                    if buckets_mode:
                        b = BUCKET_RED if locked_colour == 'pink' else BUCKET_GREEN
                        bname = 'RED' if locked_colour == 'pink' else 'GREEN'
                        status = f'GRIPPED — {locked_colour or "?"} -> {bname} BUCKET'
                        status_colour = (0, 255, 0)
                        if grip_time and now - grip_time >= 0.4:
                            seq_base = b['base']
                            seq_hover = b['pose']
                            # PLACE, don't drop: lower into the bucket mouth
                            # before releasing.
                            seq_insert = dict(b['pose'])
                            seq_insert[SHOULDER_ID] = (b['pose'][SHOULDER_ID]
                                                       + BUCKET_DIP_RAW)
                            actuator.move_joints(PLACE_CARRY)
                            phase, phase_start = 'P_CARRY', now
                            log(f"P_CARRY — {locked_colour} battery to the "
                                f"{bname} bucket")
                    elif slots_mode:
                        nxt = next((i for i, f in enumerate(slot_full)
                                    if not f and (i + 1) in USABLE_SLOTS), None)
                        if nxt is None:
                            status = 'HOLSTER FULL — 0 = reset, ENTER = release'
                            status_colour = (0, 0, 255)
                        else:
                            status, status_colour = f'GRIPPED — TO SLOT {nxt + 1}', (0, 255, 0)
                            if grip_time and now - grip_time >= 0.4:
                                cur_slot = nxt
                                seq_base = SLOTS[nxt]['base']
                                seq_hover = SLOTS[nxt]['hover']
                                seq_insert = SLOTS[nxt]['insert']
                                actuator.move_joints(PLACE_CARRY)
                                phase, phase_start = 'P_CARRY', now
                                log(f"P_CARRY — delivering to slot {nxt + 1}")
                    elif place_mode:
                        status, status_colour = 'GRIPPED — PLACING IN SLEEVE', (0, 255, 0)
                        if grip_time and now - grip_time >= 0.4:
                            seq_base = PLACE_BASE_RAW
                            seq_hover = PLACE_HOVER
                            seq_insert = PLACE_INSERT
                            actuator.move_joints(PLACE_CARRY)
                            phase, phase_start = 'P_CARRY', now
                            log("P_CARRY — lifting to upright carry pose")
                    elif box_sel is not None:
                        status, status_colour = 'GRIPPED — DELIVERING TO BOX', (0, 255, 0)
                        if grip_time and now - grip_time >= 0.4:
                            actuator.arm_home()      # lift, gripper closed
                            carry_step = 0
                            phase, phase_start = 'CARRY', now
                            log("CARRY — lifting object, then swinging to box")
                    else:
                        status = 'GRIPPED — B = BRING HOME, ENTER = RELEASE'
                        status_colour = (0, 255, 0)

                elif phase == 'CARRY':
                    status, status_colour = 'CARRYING TO BOX', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if carry_step == 0:
                            if (joints_arrived(actuator.rest_pose)
                                    or now - phase_start >= CARRY_TIMEOUT_S):
                                goal_raw = actuator.move_to(box_sel['base_goal'])
                                carry_step = 1
                                phase_start = now
                                log(f"CARRY — base swinging to box goal {goal_raw}")
                        else:
                            try:
                                near = abs(actuator.read_position() - goal_raw) <= 10
                            except ActuatorError:
                                near = False
                            if near or now - phase_start >= CARRY_TIMEOUT_S:
                                phase, phase_start = 'PLACE_CENTRE', now
                                z_hist.clear()
                                pos_hist.clear()
                                goal_raw = None
                                track_reset()
                                log("PLACE_CENTRE — re-centring on the box")

                elif phase == 'PLACE_CENTRE':
                    status, status_colour = 'CENTRING ON BOX', (255, 0, 255)
                    box_now = min(boxes, key=lambda c: abs(c['u'] - centre_x)) if boxes else None
                    if box_now is not None:
                        draw_candidate(image, box_now, (255, 0, 255), 3)
                        z_hist.append(box_now['z'])
                        err = box_now['u'] - centre_x
                        present_raw = (present_at(now - CAMERA_LATENCY_S)
                                       or actuator.read_position())
                        if abs(err) <= IMAGE_DEADBAND_PX * 3 and base_speed() < 60:
                            z_med = (sorted(z_hist)[len(z_hist) // 2]
                                     if z_hist else box_sel['z'])
                            _, pose = actuator.reach_pose(z_med)
                            drop_pose = {
                                sid: int(round(actuator.rest_pose[sid]
                                               + DROP_POSE_FRAC
                                               * (pose[sid] - actuator.rest_pose[sid])))
                                for sid in pose}
                            low_pose = {
                                sid: int(round(actuator.rest_pose[sid]
                                               + PLACE_LOW_FRAC
                                               * (pose[sid] - actuator.rest_pose[sid])))
                                for sid in pose}
                            actuator.move_joints(drop_pose)
                            phase, phase_start = 'PLACE_REACH', now
                            log(f"PLACE_REACH hover={drop_pose} z={z_med:.3f}")
                        else:
                            new_goal = track_goal(present_raw, err, now)
                            if new_goal is not None and (
                                    goal_raw is None
                                    or abs(new_goal - goal_raw) > GOAL_UPDATE_MIN_RAW):
                                goal_raw = actuator.move_to(new_goal)
                                log(f"PLACE_TRACK err={err:+d}px goal={goal_raw}")
                    elif now - phase_start >= PLACE_BLIND_S:
                        # Spot never re-detected: place from the stored estimate.
                        _, pose = actuator.reach_pose(box_sel['z'])
                        drop_pose = {
                            sid: int(round(actuator.rest_pose[sid]
                                           + DROP_POSE_FRAC
                                           * (pose[sid] - actuator.rest_pose[sid])))
                            for sid in pose}
                        low_pose = {
                            sid: int(round(actuator.rest_pose[sid]
                                           + PLACE_LOW_FRAC
                                           * (pose[sid] - actuator.rest_pose[sid])))
                            for sid in pose}
                        actuator.move_joints(drop_pose)
                        phase, phase_start = 'PLACE_REACH', now
                        log(f"PLACE_REACH (BLIND) hover={drop_pose} z={box_sel['z']:.3f}")

                elif phase == 'PLACE_REACH':
                    status, status_colour = 'HOVERING OVER SPOT', (255, 0, 255)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(drop_pose)
                                or now - phase_start >= STAGE_TIMEOUT_S):
                            actuator.set_arm_profile(INSERT_SPEED, INSERT_ACC)
                            actuator.move_joints(low_pose)
                            phase, phase_start = 'PLACE_LOWER', now
                            log(f"PLACE_LOWER (slow) goals={low_pose}")

                elif phase == 'PLACE_LOWER':
                    status, status_colour = 'LOWERING (SLOW + GENTLE)', (255, 0, 255)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(low_pose)
                                or now - phase_start >= 6.0):
                            actuator.grip_open()
                            drop_time = None       # set once physically open
                            phase, phase_start = 'DROP', now
                            log("DROP — opening gripper at the spot")

                elif phase == 'DROP':
                    # Stay put until the fingers have PHYSICALLY opened past
                    # the release point — the gripper is slow, and retracting
                    # early carries the object away from the box.
                    if now - last_check >= 0.10:
                        last_check = now
                        if drop_time is None:
                            if actuator.grip_released():
                                drop_time = now
                                log("RELEASED over the box")
                            elif now - phase_start >= 6.0:
                                drop_time = now
                                log("RELEASE TIMEOUT — retracting anyway")
                        elif now - drop_time >= DROP_PAUSE_S:
                            actuator.move_joints(drop_pose)   # retreat UP first
                            phase, phase_start = 'PLACE_UP', now
                            log("PLACE_UP — rising clear, object placed")
                    if phase == 'DROP':
                        if drop_time is None:
                            status, status_colour = 'OPENING GRIPPER AT SPOT', (255, 0, 255)
                        else:
                            status, status_colour = 'PLACED — RISING', (0, 255, 0)

                elif phase == 'PLACE_UP':
                    status, status_colour = 'RETREATING UP', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(drop_pose)
                                or now - phase_start >= STAGE_TIMEOUT_S):
                            actuator.set_arm_profile(ARM_MOVE_SPEED, ARM_MOVE_ACC)
                            actuator.go_rest()
                            log("CYCLE COMPLETE — returned to rest")
                            reset_cycle()

                elif phase == 'P_CARRY':
                    status, status_colour = 'LIFTING — BATTERY UPRIGHT', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(PLACE_CARRY, DELIVERY_TOL)
                                or now - phase_start >= CARRY_TIMEOUT_S):
                            goal_raw = actuator.move_to(seq_base)
                            phase, phase_start = 'P_SWING', now
                            log(f"P_SWING — base to drop point at {goal_raw}")

                elif phase == 'P_SWING':
                    status, status_colour = 'SWINGING TO SLEEVE', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        try:
                            near = abs(actuator.read_position() - goal_raw) <= 15
                        except ActuatorError:
                            near = False
                        if near or now - phase_start >= CARRY_TIMEOUT_S:
                            actuator.move_joints(seq_hover)
                            phase, phase_start = 'P_HOVER', now
                            log("P_HOVER — positioning above the drop point")

                elif phase == 'P_HOVER':
                    status, status_colour = 'ABOVE THE SLOT', (255, 0, 255)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(seq_hover, DELIVERY_TOL)
                                or now - phase_start >= STAGE_TIMEOUT_S):
                            actuator.set_arm_profile(INSERT_SPEED, INSERT_ACC)
                            actuator.move_joints(seq_insert)
                            arrived_time = None
                            phase, phase_start = 'P_INSERT', now
                            log("P_INSERT — final approach")

                elif phase == 'P_INSERT':
                    status, status_colour = 'INSERTING (SLOW + PRECISE)', (255, 0, 255)
                    if now - last_check >= 0.10:
                        last_check = now
                        if arrived_time is None and joints_arrived(seq_insert, DELIVERY_TOL):
                            arrived_time = now
                            log("AT DROP POINT — settling")
                        settled = (arrived_time
                                   and now - arrived_time >= PLACE_SETTLE_S)
                        if settled or now - phase_start >= 3.0:
                            actuator.grip_open()
                            phase, phase_start = 'P_RELEASE', now
                            log("P_RELEASE — opening in the sleeve")

                elif phase == 'P_RELEASE':
                    status, status_colour = 'RELEASING IN SLEEVE', (255, 0, 255)
                    if now - last_check >= 0.10:
                        last_check = now
                        if actuator.grip_released() or now - phase_start >= 5.0:
                            actuator.move_joints(seq_hover)
                            phase, phase_start = 'P_RETREAT', now
                            log("P_RETREAT — rising clear of the slot")

                elif phase == 'P_RETREAT':
                    status, status_colour = 'RETREATING', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(seq_hover, DELIVERY_TOL)
                                or now - phase_start >= STAGE_TIMEOUT_S):
                            if cur_slot is not None:
                                slot_full[cur_slot] = True
                                log(f"SLOT {cur_slot + 1} FILLED "
                                    f"({sum(slot_full)}/{len(slot_full)})")
                                cur_slot = None
                            # SMOOTH return: fold the arm FIRST (base still),
                            # only then swing the base — no whip.
                            actuator.set_arm_profile(RETURN_SPEED, RETURN_ACC)
                            actuator.arm_home()
                            phase, phase_start = 'P_HOME', now
                            log("P_HOME — folding arm before the base swings")

                elif phase == 'P_HOME':
                    status, status_colour = 'RETURNING (SMOOTH)', (0, 255, 0)
                    if now - last_check >= 0.10:
                        last_check = now
                        if (joints_arrived(actuator.rest_pose, DELIVERY_TOL)
                                or now - phase_start >= 3.0):
                            actuator.set_arm_profile(ARM_MOVE_SPEED, ARM_MOVE_ACC)
                            actuator.move_to(BASE_FORWARD_RAW)
                            if locked_colour == 'pink':
                                n_pink += 1
                            elif locked_colour == 'ryobi':
                                n_ryobi += 1
                            metrics.placed_ok(locked_colour)
                            log(f"PLACED — cycle complete "
                                f"(ryobi {n_ryobi}, pink {n_pink})")
                            reset_cycle()

                elif phase == 'HOME':
                    status = 'OBJECT AT BASE — ENTER = RELEASE'
                    status_colour = (0, 255, 0)

                # ---- Mouse selection (works even where detection fails) ----
                if mouse_click['pt'] is not None:
                    mx, my = mouse_click['pt']
                    mouse_click['pt'] = None
                    if phase == 'IDLE' and not supervisor.stop_latched:
                        pick = None
                        if candidates:
                            pick = min(candidates, key=lambda c:
                                       (c['u'] - mx) ** 2 + (c['v'] - my) ** 2)
                            if ((pick['u'] - mx) ** 2
                                    + (pick['v'] - my) ** 2) ** 0.5 > CLICK_RADIUS_PX:
                                pick = None
                        if pick is not None:
                            locked_point = (int(pick['u']), int(pick['v']))
                            locked_z = pick['z']
                            locked_colour = (pick.get('colour')
                                             or classify_colour(image, pick['bbox']))
                            log(f"TARGET LOCKED (click->detection) {locked_point} "
                                f"z={locked_z:.3f} colour={locked_colour} "
                                f"src={pick['source']}")
                        else:
                            z_click = _measure_depth(depth, mx, my)
                            if z_click is None:
                                log(f"CLICK ignored — no reliable depth at ({mx},{my})")
                                locked_point = None
                            elif not (REACH_NEAR_M - 0.05 <= z_click
                                      <= DETECT_Z_MAX_M + 0.05):
                                # A z=0.857 click once sent the arm on a wild
                                # far-reach — clicks outside the arm's range
                                # are background, not targets.
                                log(f"CLICK ignored — z={z_click:.3f}m is outside "
                                    f"the arm's working range")
                                locked_point = None
                            else:
                                locked_point = (mx, my)
                                locked_z = z_click
                                locked_colour = classify_colour(
                                    image, (mx - 20, my - 30, 40, 60))
                                log(f"TARGET LOCKED (click->raw point) {locked_point} "
                                    f"z={locked_z:.3f} colour={locked_colour}")
                        if locked_point is not None:
                            z_hist.clear()
                            z_hist.append(locked_z)
                            centre_ok = 0
                            if place_mode or slots_mode or buckets_mode:
                                go_centre()   # straight to the grab
                            else:
                                phase = 'SEL_BOX'
                    elif phase == 'SEL_BOX' and not supervisor.stop_latched:
                        bx, z_box = mx, None
                        if boxes:
                            nb = min(boxes, key=lambda c:
                                     (c['u'] - mx) ** 2 + (c['v'] - my) ** 2)
                            if ((nb['u'] - mx) ** 2
                                    + (nb['v'] - my) ** 2) ** 0.5 <= CLICK_RADIUS_PX * 2:
                                bx, z_box = nb['u'], nb['z']
                        if z_box is None:
                            z_box = _measure_depth(depth, mx, my)
                        if z_box is None:
                            log(f"CLICK ignored — no reliable depth at ({mx},{my})")
                        else:
                            present_raw = actuator.read_position()
                            box_err = bx - centre_x
                            box_sel = {
                                'base_goal': int(round(
                                    present_raw + BASE_DIRECTION_SIGN * box_err * RAW_PER_PIXEL)),
                                'z': z_box,
                            }
                            log(f"BOX LOCKED (click) err={box_err:+d}px z={z_box:.3f} "
                                f"base_goal={box_sel['base_goal']}")
                            actuator.grip_open()
                            err = locked_point[0] - centre_x
                            pos_hist.clear()
                            pos_hist.append((now, present_raw))
                            assoc_base = present_raw
                            goal_raw = actuator.move_to(
                                present_raw + BASE_DIRECTION_SIGN * err
                                * RAW_PER_PIXEL * CLOSED_LOOP_GAIN)
                            centre_ok = 0
                            phase, phase_start = 'CENTRE', now
                            log(f"GO — centring on target, goal={goal_raw}")

                if slots_mode:
                    slots_txt = 'SLOTS  ' + '  '.join(
                        f"{i + 1}:{'X' if f else ('_' if (i + 1) in USABLE_SLOTS else '-')}"
                        for i, f in enumerate(slot_full))
                    _text(image, slots_txt, 310, CAMERA_HEIGHT - 20, 0.44, _INK, 1)

                # Smoothed frame rate — a single spike shouldn't make the
                # read-out jitter during a demo.
                _dt = now - last_frame_t
                last_frame_t = now
                if 0.0 < _dt < 1.0:
                    fps_ema = (_dt if fps_ema is None
                               else fps_ema * 0.9 + _dt * 0.1)
                draw_hud(image, phase, status, status_colour,
                         grab_top=(force_lying and LYING_POINTS is not None),
                         auto_on=auto_mode and buckets_mode,
                         scanning=scanning,
                         sweep=f'{scan_sweeps + 1}/{SCAN_MAX_SWEEPS}',
                         n_ryobi=n_ryobi, n_pink=n_pink,
                         fps=(1.0 / fps_ema) if fps_ema else 0.0,
                         target=current if phase == 'IDLE' else None,
                         stopped=supervisor.stop_latched)
                cv2.imshow(window, image)

            # ---- single key poll per loop -----------------------------------
            key = cv2.waitKey(1) & 255
            if key in (13, 10) and now - last_enter >= ENTER_DEBOUNCE_S:
                last_enter = now
                if phase == 'IDLE' and buckets_mode and not supervisor.stop_latched:
                    auto_mode = True
                    auto_last_seen = now
                    log("AUTO SORT started — ENTER again (mid-cycle) aborts")
                elif phase == 'IDLE' and current is not None and not supervisor.stop_latched:
                    locked_point = (int(current['u']), int(current['v']))
                    locked_z = current['z']
                    locked_colour = current.get('colour')
                    z_hist.clear()
                    z_hist.append(current['z'])
                    log(f"TARGET LOCKED z={locked_z:.3f} "
                        f"size={current['width_m']*1000:.0f}x{current['height_m']*1000:.0f}mm "
                        f"src={current['source']}")
                    if place_mode or slots_mode:
                        go_centre()     # delivery point known: no box selection
                    else:
                        phase = 'SEL_BOX'
                elif phase == 'SEL_BOX' and not supervisor.stop_latched:
                    if box_pick is not None:
                        present_raw = actuator.read_position()
                        box_err = box_pick['u'] - centre_x
                        box_sel = {
                            'base_goal': int(round(
                                present_raw + BASE_DIRECTION_SIGN * box_err * RAW_PER_PIXEL)),
                            'z': box_pick['z'],
                        }
                        log(f"BOX CONFIRMED err={box_err:+d}px z={box_pick['z']:.3f} "
                            f"base_goal={box_sel['base_goal']}")
                        # start the grab
                        actuator.grip_open()
                        err = locked_point[0] - centre_x
                        pos_hist.clear()
                        pos_hist.append((now, present_raw))
                        assoc_base = present_raw
                        goal_raw = actuator.move_to(
                            present_raw + BASE_DIRECTION_SIGN * err
                            * RAW_PER_PIXEL * CLOSED_LOOP_GAIN)
                        centre_ok = 0
                        phase, phase_start = 'CENTRE', now
                        log(f"GO — centring on target, goal={goal_raw}")
                elif phase not in ('IDLE', 'SEL_BOX') and not supervisor.stop_latched:
                    auto_mode = False                     # abort auto-sort too
                    actuator.grip_open()
                    actuator.wait_released(timeout=3.0)   # don't drag the object
                    actuator.set_arm_profile(ARM_MOVE_SPEED, ARM_MOVE_ACC)
                    actuator.go_rest()
                    log(f"RELEASE from phase {phase} — returning to rest")
                    reset_cycle()
            if key in (ord('x'), ord('X')) and phase == 'SEL_BOX' and not supervisor.stop_latched:
                box_sel = None
                actuator.grip_open()
                present_raw = actuator.read_position()
                err = locked_point[0] - centre_x
                pos_hist.clear()
                pos_hist.append((now, present_raw))
                assoc_base = present_raw
                goal_raw = actuator.move_to(
                    present_raw + BASE_DIRECTION_SIGN * err * RAW_PER_PIXEL * CLOSED_LOOP_GAIN)
                centre_ok = 0
                phase, phase_start = 'CENTRE', now
                log("NO BOX — grab-only cycle")
            if key in (ord('b'), ord('B')) and phase == 'GRIPPED' and box_sel is None \
                    and not supervisor.stop_latched:
                actuator.go_rest()             # gripper stays CLOSED
                phase = 'HOME'
                log("BRING HOME — returning to base with object held")
            if key in (ord('l'), ord('L')):
                force_lying = not force_lying
                log(f"FORCE LYING GRAB {'ON' if force_lying else 'OFF'}")
            if slots_mode and ord('0') <= key <= ord('5'):
                if key == ord('0'):
                    slot_full = [False] * len(slot_full)
                    log("SLOTS reset — all marked empty")
                else:
                    idx = key - ord('1')
                    if idx < len(slot_full):
                        slot_full[idx] = not slot_full[idx]
                        log(f"SLOT {idx + 1} manually marked "
                            f"{'FULL' if slot_full[idx] else 'EMPTY'}")
            if key in (27, ord('e')):
                supervisor.latch_stop('operator software stop')
                actuator.emergency_stop()
                log("E-STOP latched by operator")
            if key == ord('q'):
                break
    finally:
        if not actuator.stop_latched and actuator.torque_on:
            try:
                actuator.go_rest()
                time.sleep(1.5)
            except Exception:
                pass
        actuator.close()
        cv2.destroyAllWindows()
        metrics.summary()
        log("SHUTDOWN")
        try:
            pipeline.stop()
        except Exception:
            pass

if __name__ == '__main__':
    main()
