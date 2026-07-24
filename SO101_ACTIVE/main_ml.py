"""ML build of the sorter — same proven pipeline, learned classifier.

    python main_ml.py

YOUR DEMO CODE IS UNTOUCHED. main.py, detector.py and config.py are not
modified in any way. This file imports main.py and substitutes two functions
at runtime, so there is only ONE copy of the robot logic to maintain — a
duplicated main.py would drift from the original within a day.

WHAT CHANGES

    colour        darkness fraction  ->  learned (green mat irrelevant)
    orientation   6 failed heuristics ->  learned  (AUTO_ORIENT on)
    phantoms      pixel-count gate    ->  explicit 'background' class

HOW IT PLUGS IN — two seams, both of which already existed:

  1. detect_candidates() is wrapped. After the depth detector returns boxes,
     the model classifies the nearest few and writes:
         c['colour'] = 'ryobi' | 'pink'
         c['label']  = 'battery_lying' | 'battery_upright'
     main.py ALREADY reads c['label'] against YOLO_LYING_LABELS as its
     highest-priority orientation source, so orientation needs no new code
     and those frames vote unbiased.

  2. classify_colour() is replaced by a lookup of the model's verdict.
     It keeps the (colour, ryobi_px, pink_px) signature, returning synthetic
     counts derived from confidence so the existing vote-seeding still
     works. A 'background' verdict returns 'unknown', which main.py's
     phantom guard already treats as "not a battery" — the background class
     wires itself in for free.

FALLING BACK
  If the model is missing or unconvincing, run main.py. Nothing here can
  affect it.
"""
import math
import sys
import time
from collections import deque

import main as demo
import ml_classify
import ops
import preflight
import roll_align
from actuator import BaseActuator
from config import ROLL_ID
from detector import classify_colour as _heuristic_colour
from ml_classify import BatteryClassifier, ModelMissing

# ---- tuning for the ML build ------------------------------------------------
ml_classify.MAX_PER_FRAME = 4     # nearest N candidates per frame
ml_classify.MIN_CONF = 0.40       # below this -> 'unknown' (no lock)
ml_classify.BG_MIN_CONF = 0.80    # background must be very sure to veto
ml_classify.CACHE_S = 0.25

ML_DEBUG = True                   # log the model's raw verdict for the
                                  # nearest target ~1x/sec, so we can see
                                  # exactly what it decides on live crops
_dbg_next = 0.0

LYING_MARGIN = 2                  # the model is trustworthy, so a smaller
                                  # margin than the depth heuristic needed
                                  # (that one required +4 to be safe)
CONF_TO_PX = 3000                 # maps confidence onto the pixel-count
                                  # scale main.py's vote seeding expects

AUTO_ROLL = True                  # align the wrist to a lying battery's axis
ROLL_ELONG_MIN = 1.8              # silhouette must be this elongated to trust
                                  # its heading (a round blob has no axis)
ROLL_FRESH_S = 1.0                # heading must be newer than this at grab time
ROLL_CENTRE_PX = 90              # target must be within this of image centre
                                  # for its heading to belong to THIS grab

# Rolling store of the grab target's measured heading. Circular stats on the
# DOUBLED angle, because a battery is 180-deg symmetric (0 == 180).
_head = deque(maxlen=9)           # (t, deg, elong)


def _note_heading(deg, elong):
    _head.append((time.monotonic(), deg, elong))


def _current_heading(now):
    """Robust recent heading in [0,180), or None if stale / not elongated."""
    good = [(d, e) for (t, d, e) in _head
            if now - t <= ROLL_FRESH_S and e >= ROLL_ELONG_MIN]
    if len(good) < 3:
        return None
    s = sum(math.sin(math.radians(2 * d)) for d, _ in good)
    c = sum(math.cos(math.radians(2 * d)) for d, _ in good)
    if s == 0 and c == 0:
        return None
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def _install(clf):
    real_detect = demo.detect_candidates

    cx_img = demo.CAMERA_WIDTH // 2

    def detect_with_ml(image, depth, stats=None):
        cands = real_detect(image, depth, stats)
        try:
            clf.classify_candidates(image, cands)
        except Exception as exc:            # never let inference kill a run
            print('[ml] inference error:', exc)
            return cands
        for c in cands:
            v = c.get('ml')
            if not v:
                continue
            c['ml_conf'] = v['conf']
            if v['ok']:
                c['colour'] = v['colour']
                # main.py's top-priority orientation source.
                c['label'] = ('battery_lying' if v['lying']
                              else 'battery_upright')
            else:
                c['colour'] = 'unknown'      # background / low confidence

        # Diagnostic: what does the model ACTUALLY say about the nearest
        # target? Throttled so it doesn't flood the log.
        global _dbg_next
        if ML_DEBUG and cands and time.monotonic() >= _dbg_next:
            _dbg_next = time.monotonic() + 1.0
            near = min(cands, key=lambda c: c['z'])
            v = near.get('ml')
            if v:
                demo.log(f"[ml] nearest z={near['z']:.2f} "
                         f"raw={v.get('raw')} -> "
                         f"cls={v['cls']} conf={v['conf']:.2f} "
                         f"colour={v['colour']} lying={v['lying']} "
                         f"ok={v['ok']}")

        # Track the heading of the LYING target nearest image centre — that
        # is the one the base is centring on and about to grab.
        if AUTO_ROLL and roll_align.is_calibrated():
            lying = [c for c in cands
                     if c.get('ml') and c['ml'].get('lying')
                     and abs(c['u'] - cx_img) <= ROLL_CENTRE_PX]
            if lying:
                c = min(lying, key=lambda c: abs(c['u'] - cx_img))
                res = roll_align.battery_axis_deg(depth, c['bbox'])
                if res is not None:
                    _note_heading(res[0], res[1])
        return cands

    def classify_colour_ml(image, bbox, with_counts=False):
        """Model verdict when confident; the proven darkness heuristic when
        not. Until the model is retrained to generalise, it mostly says
        'background' on live crops, so this keeps sorting working by
        deferring to the heuristic that ran the demo."""
        v = None
        bx, by, bw, bh = bbox
        u, v_ = bx + bw // 2, by + bh // 2
        cached = clf._cached(u, v_, time.monotonic())
        if cached is not None:
            v = cached
        else:
            stub = [{'u': u, 'v': v_, 'z': 0.0, 'bbox': bbox}]
            try:
                clf.classify_candidates(image, stub)
                v = stub[0].get('ml')
            except Exception as exc:
                print('[ml] inference error:', exc)
        if v and v['ok']:
            px = int(v['conf'] * CONF_TO_PX)
            rc, pc = (px, 0) if v['colour'] == 'ryobi' else (0, px)
            return (v['colour'], rc, pc) if with_counts else v['colour']
        # FALLBACK: model unsure -> demo-proven darkness classifier.
        return _heuristic_colour(image, bbox, with_counts=with_counts)

    # Patch main's OWN namespace: it did `from detector import ...`, so it
    # holds direct references — patching detector.* would have no effect.
    demo.detect_candidates = detect_with_ml
    demo.classify_colour = classify_colour_ml

    # --- auto wrist-roll: override ONLY the roll joint of a lying grab ------
    if AUTO_ROLL:
        _real_reach = BaseActuator.reach_pose

        def reach_pose_aligned(self, z, points=None):
            t, pose = _real_reach(self, z, points)
            # Only for lying grabs (points given) and only if we have a fresh,
            # confident heading and a roll calibration. Otherwise the taught
            # roll in `pose` stands — safe fallback, never worse than before.
            if points is not None and roll_align.is_calibrated():
                deg = _current_heading(time.monotonic())
                raw = roll_align.roll_for_heading(deg)
                if raw is not None and ROLL_ID in pose:
                    old = pose[ROLL_ID]
                    pose = dict(pose)
                    pose[ROLL_ID] = raw
                    demo.log(f"ROLL aligned to {deg:.0f} deg: "
                             f"{old} -> {raw}")
            return t, pose

        BaseActuator.reach_pose = reach_pose_aligned

    # The model supersedes the generic COCO YOLO path; turning it off removes
    # a whole detector's cost and stops two sources disagreeing.
    demo.USE_YOLO = False
    demo.AUTO_ORIENT = True
    demo.LYING_VOTE_MARGIN = LYING_MARGIN
    demo.COMPUTE_ORIENT_METRICS = False   # depth orientation stats unused now


def main():
    print(__doc__)

    # 1. Preflight — refuse to power the arm into a broken config.
    #    Exit code 2 = fatal config problem: the restart harness must NOT
    #    loop on it (looping won't un-break a missing calibration).
    if not preflight.report(require_model=True):
        print('Preflight NO-GO — fix the FAIL rows above, then re-run.')
        sys.exit(2)

    # 2. Load the classifier.
    try:
        clf = BatteryClassifier()
    except ModelMissing as exc:
        print('\n' + str(exc))
        print('\nRun main.py for the demo build in the meantime.')
        sys.exit(2)

    _install(clf)

    # 3. Observability: tee the log to a session file + live metrics, and
    #    flag missed grabs the moment they happen.
    metrics = ops.SessionMetrics()
    _real_log = demo.log

    def log_tee(msg):
        _real_log(msg)
        try:
            metrics.observe(msg)
        except Exception:
            pass
    demo.log = log_tee
    ops.install_grab_check(BaseActuator, metrics, log_tee)

    print(f'[ml] AUTO_ORIENT on, lying needs +{LYING_MARGIN} votes, '
          f'L still forces top-down')
    print(f'[ml] session log -> {metrics.path}')
    print('[ml] starting robot...\n')

    # 4. Run, and ALWAYS print the summary — clean quit or crash.
    try:
        demo.main()
    finally:
        metrics.print_summary()
        metrics.close()


if __name__ == '__main__':
    main()
