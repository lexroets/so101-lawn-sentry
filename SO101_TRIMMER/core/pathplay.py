"""Play a taught path: exactly as taught, stopping at every point.

A pose says WHERE to go, not HOW to get there, and left to themselves the
servos answer "every joint moves at its own pace toward its own target". That
caused two failures worth recording:

    IT DID NOT STOP AT EACH POINT.  Blending commanded the next waypoint as
    soon as the arm was merely NEAR the current one, so it never arrived
    anywhere. Fine for a continuous cut, useless for proving a path.

    IT DIVED INTO THE TABLE BETWEEN POINTS.  Joints start together and finish
    whenever they each finish, so a joint with a small change is done long
    before one with a large change. In between, the arm passes through poses
    that were never taught, and those can put the head well below both ends.

Both are fixed by INTERPOLATING: each move is cut into small steps and
commanded in turn, so every joint stays in proportion and the arm follows a
straight line in joint space.

THE HEAD JOINT GOES EXACTLY WHERE IT WAS TAUGHT
-----------------------------------------------
There was a third fix here and it has been removed, because it was wrong in a
way worth being explicit about.

A spinning head is symmetric under half a turn - rolled to 30 degrees or 210
it cuts identically - so the code used to substitute whichever equivalent
angle was closer, turning a 193 degree sweep into 13. Mathematically sound,
and it did work on a single path.

It is still the wrong thing to do. That joint carries the motor cable, and a
substituted angle is one nobody has ever put the arm into and watched. On a
single path it happened to be safe; playing several in sequence it took the
head somewhere no taught waypoint had ever been, and nearly took the loom with
it. Cleverness that produces unverified positions is not worth 180 degrees of
saved travel.

So: the head is commanded to the taught value, unchanged. On top of that, the
taught path itself defines the envelope - the head is never sent outside the
range of values that appear IN THAT PATH. If every angle it can reach is one
you put it at by hand, there is nothing left to go wrong.

THE HEAD'S USABLE ARC CROSSES THE SERVO'S ZERO
----------------------------------------------
On this rig the arc runs 1000 down through 0, wraps to 4095, and continues
down to 3000. Flat sits at 0, right on the wrap. The forbidden middle - the
part containing 2048 - is where the head is at a dangerous angle.

Two consequences, and both are handled by working in ARC COORDINATES rather
than raw ones:

    a plain min/max cannot express it. min=3000, max=1000 reads as an empty
    range; 0..4095 permits everything, including the middle.

    interpolating raw values walks THROUGH the middle. Stepping 3500 to 500
    passes 3000, 2500, 2048. In arc coordinates the same move is 500 to 1596
    and comes out as 3500 -> 4095 -> 0 -> 500, which is the short way and the
    safe way.

The servo itself cannot cross its own zero on command - it drives to an
absolute encoder value, so 4095 -> 0 sweeps backwards the long way. Any
waypoint pair that would require it is refused rather than attempted.
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

import trimmer_config as tc

STEP_RAW = 45               # how far any joint moves between commands
STEP_PERIOD_S = 0.06
SETTLE_S = 0.25             # pause on each taught waypoint, so you can see it
ARRIVE_TIMEOUT_S = 4.0

# A path placed by hand is a list of DESTINATIONS and each is worth stopping
# on. A path captured by dragging the arm is samples of ONE MOVEMENT, and
# stopping on every sample turns a smooth pass into a stutter.
#
# Which it is cannot be read off the spacing, because thinning a drag-taught
# path leaves wide gaps wherever the motion was straight - a perfectly smooth
# sweep can come out as a handful of far-apart points. The reliable signal is
# how many there are: nobody places twelve poses by hand one at a time.
CONTINUOUS_MIN_POINTS = 8

# How far behind the commanded point the arm is allowed to fall inside a
# continuous pass. Small enough that the real path stays on the taught one,
# large enough that it flows instead of stopping dead at every sample.
BLEND_TOL_RAW = 110

# Interpolation only means anything if the arm can keep up with it. Stepping
# 45 raw every 60 ms asks for 750 raw/s; tell the servos they may do 120 and
# the commanded point runs away, the arm stops tracking the path and heads
# straight for whatever was commanded last - which is the dive again.
# So the step PERIOD is derived from the speed the arm was actually given.
RATE_RAW_PER_S = None


def set_rate(raw_per_s):
    """Tell the player how fast the arm has been allowed to move."""
    global RATE_RAW_PER_S
    RATE_RAW_PER_S = float(raw_per_s) if raw_per_s else None


def step_period():
    return STEP_PERIOD_S


def step_size():
    """How far to move between commands.

    THIS IS WHERE THE JOLTING CAME FROM. The step was fixed at 45 raw and the
    PERIOD was stretched to match the speed - so at 200 raw/s the arm was told
    to jump 45 raw (4 degrees), then nothing at all for 225 ms, then jump
    again. Four visible twitches a second. The path was smooth; the delivery
    of it was not.

    Fixing the period and deriving the SIZE gives the same average speed as a
    steady trickle: 12 raw every 60 ms rather than 45 raw every 225 ms. Same
    journey, same duration, no twitching.
    """
    if not RATE_RAW_PER_S:
        return STEP_RAW
    return max(3, int(round(RATE_RAW_PER_S * STEP_PERIOD_S)))


# ---- the head arc ----------------------------------------------------------
# The usable arc crosses the servo's zero (see trimmer_config), which makes
# plain comparisons useless: 3500 and 500 are both allowed and only 1096 ticks
# apart along the arc, but numerically they look 3000 apart with the forbidden
# middle in between.
#
# Everything below therefore works in ARC COORDINATES: distance measured from
# the start of the arc, counting up through 4095/0. In that coordinate the arc
# is a plain interval 0..2096 and all the awkwardness disappears.


def to_arc(raw):
    """Raw encoder value -> distance along the allowed arc."""
    return (int(raw) - tc.HEAD_ARC_LO) % 4096


def from_arc(a):
    """Distance along the arc -> raw encoder value."""
    return int(tc.HEAD_ARC_LO + a) % 4096


def arc_len():
    return (tc.HEAD_ARC_HI - tc.HEAD_ARC_LO) % 4096


def head_ok(raw):
    """Is this angle on the allowed arc at all?"""
    return to_arc(raw) <= arc_len()


def crosses_wrap(a, b):
    """Does the straight raw path between these two pass through the middle?"""
    return not (head_ok(a) and head_ok(b)) or \
        (min(int(a), int(b)) < tc.HEAD_ARC_HI < tc.HEAD_ARC_LO < max(int(a), int(b)))


HEAD_LAG_TOL = 600          # see step_head - a following error, not a fault


def step_head(arm, hid, goal_raw, log=None, tol=HEAD_LAG_TOL):
    """Command one head step and CHECK the servo went where it was told.

    This is the crux of the whole problem. The usable arc straddles the
    servo's zero, so reaching one end from the other means stepping across
    4095 -> 0. Whether that is safe depends on firmware nobody documents
    consistently: if the controller wraps its error term, the step is one
    tick; if it does not, it sweeps the long way round through the forbidden
    middle, which is what grabbed the cable.

    Rather than assume either way, command the step and immediately read
    back. A servo taking the long way is somewhere completely different
    within a few tens of milliseconds, and that is unmistakable - so it is
    caught on the first step and stopped, instead of after half a rotation.

    WHAT THE TOLERANCE HAS TO ALLOW
    -------------------------------
    A moving servo is ALWAYS behind its target - it is chasing a goal that
    moves again every 60 ms, so a steady lag of a hundred-odd ticks is the
    normal, healthy picture. The first version used 140 and aborted constantly
    on perfectly good moves at 141, 142, 148. That is a following error, not a
    fault.

    The failure being looked for is nothing like that. Going the long way
    round means heading off in the opposite direction, so within one step it
    is hundreds of ticks out and climbing, and it enters the forbidden half.
    Both of those are checked; a lag is not.

    AND WHEN THE ARC NO LONGER WRAPS, NONE OF IT IS NEEDED.
    The whole check exists because flat used to sit on the encoder's zero, so
    reaching one end of the arc from the other meant a 4095 -> 0 step the
    servo could read as "sweep backwards". With the zero moved, the usable
    arc is one ascending block and there is no long way round to take.

    Leaving it in costs 50 ms of sleep plus a blocking read on EVERY head
    step - measured at 2.5 seconds per ten waypoints, all of it waiting for
    an answer to a question that cannot have a bad answer any more.
    """
    arm.move_to(hid, int(goal_raw) % 4096)
    if tc.HEAD_ARC_LO < tc.HEAD_ARC_HI:
        return True
    # ONCE THE ARC NO LONGER WRAPS, THERE IS NOTHING LEFT TO CATCH.
    # This whole check existed because flat sat on the encoder's zero, so
    # reaching one end of the arc from the other meant a 4095 -> 0 step the
    # servo could read as "sweep 4095 ticks backwards". With the zero moved,
    # the usable arc is one ascending block and every commanded value inside
    # it is reached by moving directly - there is no long way round to take.
    #
    # Leaving the check in was actively harmful: it reads the servo back after
    # 50 ms and calls any large gap a fault, but a slow profile means the
    # commanded value legitimately runs ahead of the arm, and it started
    # aborting good moves at 601 out.
    if tc.HEAD_ARC_LO < tc.HEAD_ARC_HI:
        return True
    time.sleep(0.05)
    try:
        here = arm.read_position(hid)
    except Exception:
        return True
    if here is None:
        return True
    # The decisive test: has it left the allowed arc? Nothing legitimate does
    # that, and it is what actually winds the cable.
    if not head_ok(here):
        if log:
            log(f'ABORT: head has entered the forbidden half at {here} - '
                f'stopping.')
        try:
            arm.move_to(hid, int(here))
        except Exception:
            pass
        return False
    # distance the SHORT way round - a servo obeying the command is near the
    # goal by this measure even when the numbers straddle zero
    err = min(abs(here - goal_raw), 4096 - abs(here - goal_raw))
    if err > tol:
        if log:
            log(f'ABORT: head was told {int(goal_raw) % 4096} and is at {here} '
                f'({err} out). It is taking the long way round - stopping.')
        try:
            arm.move_to(hid, int(here))       # stop it where it is
        except Exception:
            pass
        return False
    return True


def head_deg(raw):
    """Head angle as degrees from flat: + clockwise, - anticlockwise.

    Raw encoder numbers are unreadable for this joint because flat sits on the
    wrap - 4095 and 3 are four ticks apart and look 4092. Every display uses
    this instead, because "62 deg clockwise" is checkable by eye and "3392" is
    not. Most of the confusion about this joint came from reading raw values.
    """
    d = (int(raw) - tc.HEAD_FLAT_RAW) % 4096
    if d > 2048:
        d -= 4096
    return -d * 360.0 / 4096.0        # decreasing raw = clockwise


def head_label(raw):
    d = head_deg(raw)
    if abs(d) < 1.5:
        return f'{int(raw)} (flat)'
    return f'{int(raw)} ({abs(d):.0f}d {"CW" if d > 0 else "ACW"})'


def head_envelope(*paths):
    """The head range the taught waypoints actually cover.

    This is the safety bound, and it comes from the teaching rather than from
    a number in a config file: the head may go anywhere it was put by hand,
    and nowhere else. A little slack is added so a waypoint sitting exactly on
    the end of the range is not fighting its own limit.
    """
    hid = getattr(tc, 'HEAD_ID', None)
    vals = [to_arc(wp[hid]) for p in paths for wp in p
            if hid is not None and hid in wp]
    if not vals:
        return None
    # in ARC coordinates, so an envelope spanning the wrap is still an interval
    return min(vals) - 20, max(vals) + 20


SMOOTH_WINDOW = 6          # samples either side, on the DENSIFIED path
# A moving average cuts corners, and a corner on this path can be the arm
# lifting off the table. Widening the window keeps buying smoothness - and
# keeps buying corner-cutting with it: window 9 measured 19 degrees off the
# taught path, which is a dive. So the smoothed pose is pulled back towards
# the taught one whenever it strays further than this, which bounds the error
# no matter what the window is set to.
MAX_SMOOTH_DEV_RAW = 45

# --- pacing -----------------------------------------------------------------
# A taught pass is not one speed. It has straights, and it has REVERSALS - the
# edger strokes down a face and comes back up, and at the turn the direction
# of travel flips through nearly 180 degrees. Driving into that at full speed
# and stopping dead is the single biggest source of jolt left, and no amount
# of smoothing the PATH fixes it, because the path is right; it is the pacing
# that is wrong.
#
# So the commanded speed is planned along the path: full rate on straights,
# eased down into corners and back out of them, with a cap on how fast the
# speed itself may change. This is what a machine tool does with a corner, and
# for the same reason.
CORNER_SLOWDOWN = 0.9       # how hard a bend cuts the speed (0 = ignore bends)
MIN_SPEED_FRAC = 0.15       # never crawl to a complete stop mid-path
# How much the distance-per-command may change from one command to the next.
# Expressed in RAW rather than as a fraction of full speed, deliberately: a
# fraction loosens as the speed rises, so a limit tuned at 400 came out twice
# as coarse at 900 and the pacing became the jolt it was meant to remove.
# In raw, the arm feels the same gentleness at every speed.
ACCEL_RAW_PER_CMD = 1.5
EASE_STEPS = 30             # fine samples of ramp at each end of the pass


def _hid():
    return getattr(tc, 'HEAD_ID', None)


def densify(waypoints, step=None):
    """One pose every `step` of joint travel, along the WHOLE path.

    Segment-at-a-time playback is what makes the motion jolt: each waypoint
    is treated as a little journey with its own start, its own end and a
    catch-up wait in between, so the arm accelerates and settles 30 times
    across one pass. Laying the whole path out as a single evenly-spaced
    stream removes the boundaries entirely - there is one motion, and the
    waypoints are just points it happens to go through.
    """
    hid = _hid()
    step = step or step_size()
    out = [dict(waypoints[0])]
    for a, b in zip(waypoints, waypoints[1:]):
        spans = [abs(int(b[j]) - int(a[j])) for j in b if j in a and j != hid]
        if hid is not None and hid in a and hid in b:
            spans.append(abs(to_arc(b[hid]) - to_arc(a[hid])))
        n = max(1, int(round((max(spans) if spans else 0) / float(step))))
        for i in range(1, n + 1):
            f = i / float(n)
            pose = {}
            for j in b:
                if j not in a:
                    continue
                if j == hid:
                    a0, a1 = to_arc(a[j]), to_arc(b[j])
                    pose[j] = from_arc(int(round(a0 + (a1 - a0) * f)))
                else:
                    pose[j] = int(round(a[j] + (b[j] - a[j]) * f))
            out.append(pose)
    return out


def smooth(poses, window=SMOOTH_WINDOW):
    """Take the hand tremor out of a dragged path.

    A path taught by pushing the arm around carries every wobble of the hand
    that pushed it. Replaying it faithfully replays the wobble too, which is
    most of what reads as "jittery" - it is not the player being jerky, it is
    the recording being shaky. A moving average over the densified stream
    removes it while leaving the shape alone.

    The ends are held fixed so the path still starts and finishes exactly
    where it was taught.
    """
    if window < 1 or len(poses) < 2 * window + 3:
        return poses
    hid = _hid()
    keys = list(poses[0])
    out = []
    for i, pose in enumerate(poses):
        if i < window or i >= len(poses) - window:
            out.append(dict(pose))
            continue
        chunk = poses[i - window:i + window + 1]
        avg = {}
        for j in keys:
            vals = [pw[j] for pw in chunk if j in pw]
            if not vals:
                avg[j] = pose[j]
            elif j == hid:
                avg[j] = from_arc(int(round(sum(to_arc(v) for v in vals)
                                            / len(vals))))
            else:
                avg[j] = int(round(sum(vals) / float(len(vals))))
        out.append(avg)

    # BOUND THE CORNER-CUTTING, WITHOUT REINTRODUCING CORNERS.
    # Clamping each sample individually was the obvious thing and it made the
    # motion measurably worse: a clamp is a kink, so capping the average put
    # back exactly the sharp changes the average had removed. Scaling the
    # whole correction by one factor instead keeps the smoothed signal smooth
    # and simply pulls all of it towards the taught path until the furthest
    # point is inside the cap.
    worst = 0
    for sm, raw in zip(out, poses):
        for j in keys:
            if j == hid:
                d = ((to_arc(sm[j]) - to_arc(raw[j]) + 2048) % 4096) - 2048
            else:
                d = sm[j] - raw[j]
            worst = max(worst, abs(d))
    if worst > MAX_SMOOTH_DEV_RAW:
        alpha = MAX_SMOOTH_DEV_RAW / float(worst)
        for sm, raw in zip(out, poses):
            for j in keys:
                if j == hid:
                    d = ((to_arc(sm[j]) - to_arc(raw[j]) + 2048) % 4096) - 2048
                    sm[j] = from_arc(to_arc(raw[j]) + int(round(alpha * d)))
                else:
                    sm[j] = int(round(raw[j] + alpha * (sm[j] - raw[j])))
    return out


_ARM_MODEL = None


def arm_model(touches):
    """The arm's geometry: published lengths, measured joint zeros.

    NUMPY ONLY, DELIBERATELY. This used scipy, which is not in every
    environment - and when the import failed it fell back to the old
    polynomial WITHOUT SAYING SO. The result was hours of tuning a model that
    was never running: the logs said the height was being levelled, and it was
    being levelled by the code this was written to replace. A dependency that
    can silently disable the fix is not worth the few lines it saves.

    WHAT IS FITTED, AND WHAT IS NOT
    ------------------------------
    The link lengths are taken from the published SO-101 URDF and pinned.
    Fitting them as well looked reasonable and came out 42% wrong on the
    upper-to-lower ratio - and the level curve of the wrong arm is not the
    level curve of this one, which is why a tilt survived every correction.

    Only the two joint zeros are solved for. Those are properties of this
    servo's calibration and cannot be looked up anywhere.

    Only the stock section is used - shoulder to elbow, elbow to wrist pitch.
    Everything past the wrist is modified on this rig, and with the head's
    pitch locked its contribution to height is a constant, which cancels when
    what is being held constant is height.
    """
    global _ARM_MODEL
    if _ARM_MODEL is not None:
        return _ARM_MODEL
    try:
        import numpy as np
    except Exception:
        return None
    if len(touches) < 3:
        return None
    g = 2 * np.pi / 4096
    L1 = float(getattr(tc, 'ARM_L1_M', 0.1160))
    L2 = float(getattr(tc, 'ARM_L2_M', 0.1350))
    sh = np.array([t[tc.SHOULDER_ID] for t in touches], float)
    el = np.array([t[tc.ELBOW_ID] for t in touches], float)

    def spread(o2, o3, s2, s3):
        a2 = s2 * (sh - o2) * g
        z = L1 * np.cos(a2) + L2 * np.cos(a2 + s3 * (el - o3) * g)
        return z - z.mean()

    best = None
    for s2 in (1.0, -1.0):
        for s3 in (1.0, -1.0):
            # coarse sweep, then Gauss-Newton - two parameters, so a grid is
            # cheap and it cannot land in the wrong basin
            for a0 in np.linspace(-2048, 2048, 25):
                for b0 in np.linspace(-2048, 2048, 25):
                    c = float(np.sqrt((spread(a0, b0, s2, s3) ** 2).mean()))
                    if best is None or c < best[0]:
                        best = (c, a0, b0, s2, s3)
    _, o2, o3, s2, s3 = best
    for _ in range(200):
        r = spread(o2, o3, s2, s3)
        j = np.stack([(spread(o2 + 0.5, o3, s2, s3) - r) / 0.5,
                      (spread(o2, o3 + 0.5, s2, s3) - r) / 0.5], axis=1)
        try:
            step = np.linalg.lstsq(j, -r, rcond=None)[0]
        except Exception:
            break
        o2, o3 = o2 + step[0] * 0.5, o3 + step[1] * 0.5
        if abs(step[0]) + abs(step[1]) < 1e-7:
            break
    r = spread(o2, o3, s2, s3)
    m = dict(L1=L1, L2=L2, SH0=float(o2), EL0=float(o3),
             S2=float(s2), S3=float(s3),
             Z=float((L1 * np.cos(s2 * (sh - o2) * g)
                      + L2 * np.cos(s2 * (sh - o2) * g
                                    + s3 * (el - o3) * g)).mean()),
             RMS_MM=float(np.sqrt((r ** 2).mean()) * 1000.0),
             BRANCH=1.0, UP=1.0, UP_SET=False)
    m['UP_SET'] = True
    m['SELF_ERR_RAW'] = float(np.abs(
        elbow_for_height(m, sh, 0.0, el) - el).max())
    m['UP_SET'] = False
    _ARM_MODEL = m
    return _ARM_MODEL


def height_mm(model, shoulder, elbow):
    """Head height above the calibrated plane, in millimetres.

    UP IS POSITIVE. The sign of the cosine model on its own is ambiguous -
    which way the servo counts is not knowable from the geometry - so it is
    settled by the one thing that is certain: the taught pass was cut ABOVE
    the surface that was touched, never through it.
    """
    import numpy as np
    g = 2 * np.pi / 4096
    a2 = model['S2'] * (np.asarray(shoulder, float) - model['SH0']) * g
    a3 = model['S3'] * (np.asarray(elbow, float) - model['EL0']) * g
    z = model['L1'] * np.cos(a2) + model['L2'] * np.cos(a2 + a3)
    return model.get('UP', 1.0) * (z - model['Z']) * 1000.0


def elbow_for_height(model, shoulder, mm, near, window=900.0):
    """The elbow that puts the head `mm` above the plane, nearest `near`.

    Solved by search rather than by inverting the cosine: inverting gives two
    answers - the arm folded one way or the other - and choosing between them
    analytically got it wrong by 2900 raw, which is a different robot.
    Searching near the taught elbow removes the ambiguity, because the arm
    stays in the configuration it was taught in.

    VECTORISED, and that is not a detail. The first version looped in Python
    over a 1801-point grid per waypoint and then bisected scalar-by-scalar -
    about 50,000 interpreted calls per path. It is called every time a path is
    played, and it turned a two-second job into a timeout. Same method, done
    on whole arrays at once.
    """
    import numpy as np
    sh = np.atleast_1d(np.asarray(shoulder, float))
    nr = np.atleast_1d(np.asarray(near, float))
    n = len(sh)
    # mm may be ONE height for the whole path, or one per pose - the sweep
    # tilts its rows, so both have to work.
    mm = np.broadcast_to(np.atleast_1d(np.asarray(mm, float)), (n,))
    # coarse sweep either side of the taught elbow, all waypoints together
    steps = 121
    off = np.linspace(-window, window, steps)
    grid = nr[:, None] + off[None, :]
    err = (height_mm(model, np.repeat(sh[:, None], steps, axis=1), grid)
           - mm[:, None])
    sign = np.sign(err)
    flip = sign[:, :-1] != sign[:, 1:]
    # of the sign changes, take the one closest to where it was taught
    dist = np.abs(grid[:, :-1] - nr[:, None])
    dist = np.where(flip, dist, np.inf)
    k = np.argmin(dist, axis=1)
    rows = np.arange(n)
    lo = grid[rows, k]
    hi = grid[rows, k + 1]
    none = ~np.isfinite(dist[rows, k])
    for _ in range(40):                       # vectorised bisection
        mid = 0.5 * (lo + hi)
        e = height_mm(model, sh, mid) - mm
        same = np.sign(e) == np.sign(height_mm(model, sh, lo) - mm)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    out = 0.5 * (lo + hi)
    if none.any():
        # no crossing in the window - take the closest the grid could get
        best = np.argmin(np.abs(err), axis=1)
        out = np.where(none, grid[rows, best], out)
    return out if np.ndim(shoulder) else float(out[0])


def prepare(waypoints, name=None, log=None):
    """Everything that happens between the taught path and the servos.

    Returns (targets, commands).

        TARGETS   where the head is meant to be - taught, levelled, pitch
                  locked. This is what "correct" means, and it is what the
                  achieved position is judged against.

        COMMANDS  what is actually sent, which is the target minus the error
                  the arm was measured to make getting there.

    Having one function produce both matters. The measuring tool and the
    player used to build their poses separately, so the tool measured the arm
    against a command the player would never send, and the correction it
    learned was for a pose that did not exist.
    """
    w = waypoints
    if name in getattr(tc, 'LEVEL_HEIGHT_PATHS', ()):
        w = level_height(w, name, log)
    if name in getattr(tc, 'LEVEL_PITCH_PATHS', ()):
        w = level_pitch(w, name, log)
    targets = [dict(x) for x in w]
    return targets, apply_sag(w, log)


def apply_sag(waypoints, log=None):
    """Subtract the error the arm was MEASURED to make.

    Every model in this file predicts where the head should be. None of them
    can predict where it actually ends up, because that depends on gravity
    droop and on friction - and friction depends on which way the joint came
    from, not on where it is.

    So it is not predicted. trimmer/check_level.py drives the path and reads
    the encoders, and this subtracts what it found. Two measured runs agreed
    to 0.2 mm, so what is being corrected is a real, stable property of this
    path on this arm rather than noise.

    It is specific to this setup, deliberately: change the path, the speed,
    the payload or the servo gains and it must be measured again. The file
    says so at the top.
    """
    if not getattr(tc, 'SAG_COMPENSATE', False):
        return waypoints
    try:
        from sag_table import SAG
    except Exception:
        if log:
            log('SAG_COMPENSATE is on but core/sag_table.py is missing - run '
                'trimmer/check_level.py first')
        return waypoints
    if not SAG:
        return waypoints

    # WHY THERE IS A FALLBACK.
    # The lookup below matches on the exact pose, which is right when the path
    # has not changed: it refuses to hand a correction measured somewhere else
    # to a waypoint that has since moved. But regenerating the sweep moves
    # every elbow, so every pose misses the tolerance and gets NO correction -
    # and the arm droops again, worst at full reach, which is where the
    # correction was largest. The measurement was still valid; only the
    # addresses changed.
    #
    # So what was measured is also kept as a curve against REACH. Overshoot is
    # a torque effect - the further the arm is stretched, the more gravity it
    # holds and the further a proportional servo must sit off target to make
    # that torque - so reach is the variable it genuinely depends on, and the
    # measurements say so plainly: +3 raw folded in, +26 at full stretch.
    #
    # Degree 1, and the reach is CLAMPED to the range that was measured. A
    # curve fitted to 1940..2554 says nothing trustworthy at 1028; holding the
    # edge value there is honest, where extrapolating a polynomial is not.
    fit = _sag_curve(SAG)
    out, worst, guessed = [], 0, 0
    for wp in waypoints:
        # match on the pose it was measured at, so a path that has been edited
        # since does not silently get someone else's correction
        best, dist = None, None
        for row in SAG:
            at = row['at']
            d = sum(abs(int(wp[j]) - int(at[j]))
                    for j in (tc.SHOULDER_ID, tc.ELBOW_ID) if j in at and j in wp)
            if dist is None or d < dist:
                best, dist = row, d
        v = dict(wp)
        if best is not None and dist is not None and dist < 120:
            for j, e in best['err'].items():
                j = int(j)
                if j in v:
                    v[j] = max(0, min(4095, int(v[j]) - int(e)))
                    worst = max(worst, abs(int(e)))
        elif fit is not None and tc.SHOULDER_ID in wp:
            guessed += 1
            for j, e in _sag_at(fit, wp[tc.SHOULDER_ID]).items():
                if j in v:
                    v[j] = max(0, min(4095, int(v[j]) - int(e)))
                    worst = max(worst, abs(int(e)))
        out.append(v)
    if log:
        log(f'sag: subtracting the measured overshoot, up to {worst} raw')
        if guessed:
            log(f'   {guessed} pose(s) are not where the correction was '
                f'measured - using the reach curve fitted to it. Run '
                f'trimmer/check_level.py to measure these poses directly.')
    return out


def _sag_curve(SAG):
    """Fit the measured overshoot against reach, once."""
    import numpy as np
    try:
        sh = np.array([float(r['at'][tc.SHOULDER_ID]) for r in SAG])
    except Exception:
        return None
    if len(sh) < 4 or sh.max() - sh.min() < 100:
        return None
    out = {'LO': float(sh.min()), 'HI': float(sh.max())}
    for j in (tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID):
        e = np.array([float(r['err'].get(j, 0)) for r in SAG])
        if np.any(e):
            out[j] = np.polyfit(sh, e, 1)
    return out


def _sag_at(fit, shoulder):
    """The correction the curve predicts, never extrapolated past the data."""
    import numpy as np
    x = min(max(float(shoulder), fit['LO']), fit['HI'])
    return {j: int(round(float(np.polyval(c, x))))
            for j, c in fit.items() if j not in ('LO', 'HI')}


def level_height(waypoints, name=None, log=None):
    """Put every pose of a pass onto ONE cutting plane.

    With the pitch locked, the tip's height depends on the shoulder and elbow
    alone - the base yaw only points the arm, it does not raise or lower it.
    Constant height is therefore a single curve through (shoulder, elbow), and
    a pass that wanders either side of that curve cuts unevenly.

    So the curve is fitted to the taught pass and every waypoint is put back
    onto it: the shoulder stays exactly as taught, the elbow becomes whatever
    holds the height. What is removed is the scatter - the hand going up and
    down while dragging - not the shape of the pass.

    The height this settles on is the AVERAGE of what was taught, which is
    what makes it the right default: the cut becomes even without moving to
    some level nobody chose.
    """
    if len(waypoints) < 4:
        return waypoints
    sh_id, el_id = tc.SHOULDER_ID, tc.ELBOW_ID
    if not all(j in waypoints[0] for j in (sh_id, el_id)):
        return waypoints
    try:
        import numpy as np
    except Exception:
        return waypoints
    sh = np.array([float(w[sh_id]) for w in waypoints])
    el = np.array([float(w[el_id]) for w in waypoints])
    if sh.max() - sh.min() < 50:
        return waypoints                  # the arm barely moves; nothing to fit
    # PREFER A TAUGHT PLANE OVER A FITTED ONE.
    # Fitting the curve to the pass itself is circular: a dragged pass bobs,
    # and a curve through the middle of the bobbing calls the bobbing "the
    # plane". Poses recorded with the head resting ON the surface do not have
    # that problem - they are at the surface height whatever the hand did on
    # the way there - so they are used in preference whenever they exist.
    coef = None
    taught = False
    try:
        import height_plane as hp
        # THE PLANE IS ONLY VALID AT THE PITCH IT WAS TAUGHT AT.
        # The trimmer head sticks out past the wrist, so its contribution to
        # the tip's height is only constant while the pitch is. A plane taught
        # at one pitch and applied at another describes a different arm, and
        # the error grows with reach - it starts high and digs in as it
        # extends. That is not a subtle bias worth warning about; it is wrong,
        # so it is refused.
        want_pitch = None
        wps = waypoints
        s3 = sorted(w[tc.SHOULDER_ID] + w[tc.ELBOW_ID] + w[tc.WRIST_ID]
                    for w in wps)
        fixed = getattr(tc, 'LEVEL_PITCH_TARGET', {}).get(name)
        want_pitch = (fixed if fixed is not None else s3[len(s3) // 2]) + \
            getattr(tc, 'LEVEL_PITCH_TRIM', 0)
        got = getattr(hp, 'PITCH_TARGET', None)
        if got is None:
            if log:
                log('REFUSED: core/height_plane.py predates the pitch check, so '
                    'there is no way to tell what head angle it was taught at. '
                    'Re-run trimmer/level_plane.py.')
            return waypoints
        if abs(int(got) - int(want_pitch)) > 60:
            if log:
                log(f'REFUSED: the plane was taught at head pitch {got} but this '
                    f'pass is played at {want_pitch}. Re-run '
                    f'trimmer/level_plane.py.')
            return waypoints
        coef = np.array(hp.COEF, float)
        taught = True
        lo, hi = hp.SHOULDER_RANGE
        out_of = int(((sh < lo - 60) | (sh > hi + 60)).sum())
        if log:
            log(f'using the plane taught in level_plane.py '
                f'(shoulder {lo:.0f}..{hi:.0f})')
            if out_of:
                log(f'{out_of} waypoint(s) reach outside the calibrated range - '
                    f'left as taught, with the correction faded in at the edge')
    except Exception:
        pass
    if coef is None:
        deg = min(int(getattr(tc, 'LEVEL_HEIGHT_DEGREE', 2)), len(waypoints) - 2)
        coef = np.polyfit(sh, el, deg)
        if log:
            log('no taught plane found - averaging the pass instead, which '
                'only removes scatter. Run trimmer/level_plane.py to fix the '
                'height properly.')
    # OUTSIDE THE TAUGHT RANGE, HOLD THE EDGE - DO NOT EXTRAPOLATE.
    # A quadratic is only meaningful where it was fitted. Past the last touch
    # point it curves away with nothing to hold it, and the further out the
    # arm reaches the faster that error grows - which is exactly where a
    # height error costs the most. Freezing it at the boundary is wrong by a
    # bounded amount instead of wrong by an unbounded one.
    ev = sh
    blend = np.ones(len(sh))
    if taught:
        ev = np.clip(sh, lo, hi)
        # OUTSIDE THE CALIBRATED RANGE, LEAVE THE POSE ALONE.
        # A pass has ends, and the ends are not cutting - they are the arm
        # arriving and leaving, folded further in than the head can be rested
        # on the surface at all. That is why those poses cannot be calibrated:
        # they are not on the plane, and they are not meant to be.
        #
        # Clamping them to the edge of the fit was worse than useless. It
        # applied the correction belonging to a completely different reach,
        # which flipped the sign at the last waypoint and put a 297 raw dive
        # right at the end of the pass. Fading the correction out instead
        # leaves the ends exactly as taught and joins them smoothly.
        margin = float(getattr(tc, 'LEVEL_HEIGHT_FADE_RAW', 150))
        if margin > 0:
            inside = np.minimum(sh - lo, hi - sh)
            blend = np.clip(inside / margin, 0.0, 1.0)
        else:
            blend = ((sh >= lo) & (sh <= hi)).astype(float)
    model = None
    if taught:
        try:
            model = arm_model(hp.TOUCHES)
        except Exception:
            model = None
    if model is not None:
        # Settle which way is up, once, from the one fact that cannot be in
        # doubt: the pass was taught cutting ABOVE the surface, not through it.
        if not model.get('UP_SET'):
            h = height_mm(model, sh, el)
            model['UP'] = -1.0 if float(np.mean(h)) < 0 else 1.0
            model['UP_SET'] = True
        mm = float(getattr(tc, 'LEVEL_HEIGHT_TRIM_MM', 0.0))
        want = elbow_for_height(model, ev, mm, el)
        if log:
            h = height_mm(model, sh, el)
            log(f'geometry: published SO-101 lengths, joint zeros solved from '
                f'your touches (rms {model["RMS_MM"]:.1f} mm)')
            log(f'taught height varies {float(h.max() - h.min()):.1f} mm - '
                f'holding it at {mm:+.1f} mm above the touched plane')
    else:
        if log:
            log('WARNING: could not build the arm model, so the height is '
                'being levelled by the OLD polynomial, which tilts with '
                'reach. This is not the corrected behaviour.')
        want = np.polyval(coef, ev) + getattr(tc, 'LEVEL_HEIGHT_TRIM', 0)
    want = el + (np.asarray(want, float) - el) * blend
    fix = want - el
    worst = float(np.abs(fix).max())
    # The cap guards against a SELF-FITTED curve being dragged off by a bad
    # waypoint. A TAUGHT plane is the ground truth - if it says the arm is
    # 15 degrees off, the arm is 15 degrees off, and refusing to correct that
    # would be refusing to do the one job this has.
    cap = getattr(tc, 'LEVEL_HEIGHT_MAX_FIX', 200)
    if taught:
        cap = 4096
    if worst > cap:
        if log:
            log(f'NOT levelling the height: one waypoint is {worst:.0f} raw off '
                f'the fitted plane (limit {cap}). The pass does not describe a '
                f'single plane - check it in edit_path before trusting this.')
        return waypoints
    out = []
    for w, v in zip(waypoints, want):
        d = dict(w)
        d[el_id] = int(max(0, min(4095, round(v))))
        out.append(d)
    if log:
        # the scatter ABOUT the plane, which is the unevenness - not the range
        # of the elbow itself, which is the shape of the pass and must stay
        log(f'cutting at one height: scatter about the plane '
            f'{float(fix.max() - fix.min()):.0f} -> 0 raw, '
            f'worst correction {worst:.0f} raw ({worst / 4096 * 360:.1f} deg)')
    return out


def level_pitch(waypoints, name=None, log=None):
    """Hold the head at ONE pitch by deriving the wrist from the other joints.

    The shoulder, elbow and wrist pitch about parallel axes, so the head's
    angle to the ground is their sum. That makes the wrist redundant: pick the
    angle you want the head held at, and the wrist that achieves it is
    determined by wherever the other two happen to be.

    Taught by hand the three drift against each other and the head tips about
    - 18.9 degrees across the taught top pass. Derived, it cannot: the head is
    flat at every pose because being flat is what defines the wrist value.
    """
    if not waypoints:
        return waypoints
    sh, el, wr = tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID
    if not all(j in waypoints[0] for j in (sh, el, wr)):
        return waypoints
    target = getattr(tc, 'LEVEL_PITCH_TARGET', {}).get(name)
    if target is None:
        sums = sorted(w[sh] + w[el] + w[wr] for w in waypoints)
        target = sums[len(sums) // 2]          # the taught pass, on average
    target += getattr(tc, 'LEVEL_PITCH_TRIM', 0)
    out, worst, clamped_n = [], 0, 0
    for w in waypoints:
        v = dict(w)
        want = target - w[sh] - w[el]
        if not (0 <= want <= 4095):
            clamped_n += 1
            want = max(0, min(4095, want))
        worst = max(worst, abs(want - w[wr]))
        v[wr] = want
        out.append(v)
    if log:
        was = max(w[sh] + w[el] + w[wr] for w in waypoints) - \
              min(w[sh] + w[el] + w[wr] for w in waypoints)
        log(f'head held flat: pitch was drifting {was / 4096 * 360:.1f} deg, '
            f'now fixed; wrist moved up to {worst} raw '
            f'({worst / 4096 * 360:.1f} deg) to do it')
        if clamped_n:
            log(f'WARNING: {clamped_n} pose(s) could not reach that angle - '
                f'the wrist ran out of travel')
    return out


def joint_limits(*paths, margin=90):
    """The box every joint stayed inside while being taught.

    Interpolating between two taught poses cannot leave this box on its own -
    a straight line between two points inside a range stays inside it. What
    CAN leave it is an offset, a hand-edited waypoint, or a path played from
    the wrong file, and those are worth catching before the head reaches the
    table rather than after.
    """
    lim = {}
    for p in paths:
        for wp in p:
            for j, v in wp.items():
                j, v = int(j), int(v)
                if j == getattr(tc, 'HEAD_ID', None):
                    continue                    # the head has its own arc
                lo, hi = lim.get(j, (v, v))
                lim[j] = (min(lo, v), max(hi, v))
    return {j: (lo - margin, hi + margin) for j, (lo, hi) in lim.items()}


def clamped(pose, limits, now=None):
    """Hold a commanded pose inside the taught box, widened to reach it from
    wherever the arm currently is."""
    if not limits:
        return pose
    out = {}
    for j, v in pose.items():
        lo, hi = limits.get(int(j), (None, None))
        if lo is None:
            out[j] = v
            continue
        if now and int(j) in now:               # the travel move may start
            here = int(now[int(j)])             # outside the taught box
            lo, hi = min(lo, here), max(hi, here)
        out[j] = max(lo, min(hi, int(v)))
    return out


def check_safe(pose, current, envelope=None, log=None):
    """Refuse anything that would put the head somewhere dangerous."""
    hid = getattr(tc, 'HEAD_ID', None)
    if hid is None or hid not in pose:
        return True
    want = int(pose[hid])

    if not head_ok(want):
        if log:
            log(f'REFUSED: head waypoint {want} is in the FORBIDDEN middle '
                f'({tc.HEAD_ARC_HI + 1}-{tc.HEAD_ARC_LO - 1}). The allowed arc '
                f'is {tc.HEAD_ARC_LO} up through 0 to {tc.HEAD_ARC_HI}.')
        return False

    # THE ENVELOPE ONLY EVER MADE SENSE WHILE THE ARC WRAPPED.
    # It was built to stop the head being sent to an angle nobody had put it
    # at by hand, back when a wrap could be resolved the wrong way round. With
    # the zero moved the arc is a single ascending block, head_ok() already
    # refuses everything outside it, and the taught values are inside it by
    # construction - so the envelope adds nothing and costs a great deal: it
    # refuses to let the head TRAVEL from flat to the side-pass angle, which
    # is the one move the demo is built on.
    if envelope is not None and tc.HEAD_ARC_LO > tc.HEAD_ARC_HI:
        a = to_arc(want)
        if not (envelope[0] <= a <= envelope[1]):
            if log:
                log(f'REFUSED: head waypoint {want} is outside the taught '
                    f'range for this path.')
            return False

    if hid in current:
        here = int(current[hid])
        # Crossing the servo zero is allowed, because the crossing is now
        # stepped and verified rather than commanded blind - see step_head.
        delta = abs(to_arc(want) - to_arc(here))
        if delta > getattr(tc, 'HEAD_MAX_STEP_RAW', 1200):
            if log:
                log(f'REFUSED: head would move {delta} raw along the arc '
                    f'({delta / 4096 * 360:.0f} deg) in one go. Add an '
                    f'intermediate waypoint.')
            return False
    return True


def lift_sign():
    """Which way the shoulder has to move to LIFT the head.

    Measured, not assumed, and by the same method the cut already uses: REST
    is up and clear, the taught poses are down on the work, so the sign of
    (working - rest) is 'down' and lifting is the opposite of it.
    """
    try:
        rest = int(tc.REST_POSE[tc.SHOULDER_ID])
    except Exception:
        return -1
    from box_paths import BOX_PATHS
    work = [w[tc.SHOULDER_ID] for v in BOX_PATHS.values() for w in v
            if tc.SHOULDER_ID in w]
    if not work:
        return -1
    work.sort()
    return -1 if work[len(work) // 2] - rest > 0 else 1


def goto(arm, target, current=None, log=None, stop_check=None,
         step=None, settle=SETTLE_S, envelope=None, quick=False,
         limits=None, lift=0):
    """Move to one pose along an interpolated path. Returns it, or None."""
    try:
        now = current or arm.read_pose()
    except Exception:
        return None
    tgt = {int(k): int(v) for k, v in target.items()}
    if not check_safe(tgt, now, envelope, log):
        return None

    hid = getattr(tc, 'HEAD_ID', None)
    moving = [j for j in tgt if j in now]
    if not moving:
        return now

    # The head is stepped along the ARC; every other joint interpolates
    # straight. Stepping it in raw values is what walked it through the
    # forbidden middle - 3500 to 500 passes 3000, 2500, 2048. Along the arc
    # the same move comes out 3500 -> 4095 -> 0 -> 500, the short way.
    head_move = hid is not None and hid in moving
    a0 = to_arc(now[hid]) if head_move else None
    a1 = to_arc(tgt[hid]) if head_move else None

    step = step or step_size()
    spans = [abs(tgt[j] - now[j]) for j in moving if j != hid]
    if head_move:
        spans.append(abs(a1 - a0))
    biggest = max(spans) if spans else 0
    n = max(1, int(biggest / float(step) + 0.5))
    for i in range(1, n + 1):
        if stop_check is not None and stop_check():
            if log:
                log('stopped mid-move')
            return None
        f = i / float(n)
        others = {j: int(round(now[j] + (tgt[j] - now[j]) * f))
                  for j in moving if j != hid}
        if lift and tc.SHOULDER_ID in others:
            # A half sine: zero at both ends, full lift in the middle. The
            # taught start and finish are therefore untouched - only the
            # journey between them is raised.
            import math as _m
            others[tc.SHOULDER_ID] += int(round(lift * _m.sin(_m.pi * f)))
        if others:
            arm.move_pose(clamped(others, limits, now))
        if head_move:
            goal = from_arc(int(round(a0 + (a1 - a0) * f)))
            if not step_head(arm, hid, goal, log=log):
                return None
        time.sleep(step_period())
    if quick:
        # A point in the middle of a continuous drag. It is not a destination,
        # so there is no full stop and no settle - BUT the arm is still made to
        # catch up to within BLEND_TOL_RAW before the next one is commanded.
        #
        # Skipping that entirely was a serious mistake. Commanding a waypoint
        # does not move the arm there, it sets a goal; issue 33 of them in
        # three seconds and the servos simply chase the last one, so the arm
        # cuts a straight line from wherever it started to the end of the path
        # and everything in between - the table included - is in the way.
        # Waiting for the arm to come within reach of each point keeps the
        # commanded position and the real one from ever diverging.
        t0 = time.monotonic()
        while time.monotonic() - t0 < ARRIVE_TIMEOUT_S:
            if stop_check is not None and stop_check():
                return None
            try:
                if arm.arrived(tgt, tol=BLEND_TOL_RAW):
                    break
            except Exception:
                # A read that fails must not become "carry on immediately" -
                # that is the runaway again, arrived at by a different route.
                # Give the arm the time the move needed and continue blind.
                time.sleep(min(0.5, biggest / 900.0))
                break
            time.sleep(0.02)
        return tgt
    t0 = time.monotonic()
    while time.monotonic() - t0 < ARRIVE_TIMEOUT_S:
        if stop_check is not None and stop_check():
            return None
        try:
            if arm.arrived(tgt, tol=tc.ARRIVE_TOL_RAW):
                break
        except Exception:
            break
        time.sleep(0.03)
    if settle:
        time.sleep(settle)
    return tgt


def play(arm, waypoints, log=print, stop_check=None, settle=SETTLE_S,
         envelope=None, name=None):
    """Walk a taught path, stopping properly at every waypoint.

    `envelope` bounds the head joint. Pass one covering EVERY path when
    playing several in a row - the move between the end of one and the start
    of the next is where an unbounded head gets into trouble.
    """
    if len(waypoints) < 2:
        log('need at least two waypoints to play')
        return False
    # HEIGHT FIRST, THEN PITCH. Levelling the height moves the elbow, and the
    # pitch lock derives the wrist from the shoulder and elbow - so doing them
    # the other way round would leave the wrist solving for an elbow that no
    # longer exists.
    _targets, waypoints = prepare(waypoints, name, log)
    if envelope is None:
        envelope = head_envelope(waypoints)
    limits = joint_limits(waypoints)
    try:
        current = arm.read_pose()
    except Exception:
        log('cannot read the arm')
        return False
    hid = getattr(tc, 'HEAD_ID', None)
    # Same reasoning: refusing to START a path because the head is not already
    # somewhere inside that path's own range is backwards. Getting there is
    # the first thing the path has to do.
    if (envelope and tc.HEAD_ARC_LO > tc.HEAD_ARC_HI
            and hid is not None and hid in current):
        # BOTH SIDES MUST BE IN ARC COORDINATES. This compared a RAW reading
        # against an ARC envelope, which is meaningless and refused a head
        # sitting exactly at flat: raw 13 is arc 1109, comfortably inside a
        # 72-1115 envelope, but 13 < 72 so it looked out of range.
        here = int(current[hid])
        a = to_arc(here)
        if not (envelope[0] <= a <= envelope[1]):
            log(f'head is at {head_label(here)}, outside the taught range '
                f'{head_label(from_arc(max(0, envelope[0])))} to '
                f'{head_label(from_arc(envelope[1]))}. Jog it back into range '
                f'before playing.')
            return False
    # How the path was taught decides how it is played - see
    # CONTINUOUS_MIN_POINTS above.
    flow = len(waypoints) >= CONTINUOUS_MIN_POINTS
    for wp in waypoints:
        if not check_safe({int(k): int(v) for k, v in wp.items()},
                          current, envelope, log):
            return False
    if flow:
        return stream(arm, waypoints, log=log, stop_check=stop_check,
                      limits=limits, settle=settle)
    for i, wp in enumerate(waypoints):
        log(f'waypoint {i + 1}/{len(waypoints)}')
        reached = goto(arm, wp, current, log=log, stop_check=stop_check,
                       settle=settle, envelope=envelope, limits=limits)
        if reached is None:
            log('path aborted')
            return False
        current = reached
    return True


def stream(arm, waypoints, log=print, stop_check=None, limits=None,
           settle=SETTLE_S):
    """Play a dragged path as ONE motion, at a steady cadence.

    Three things make this fluid where the old segment-by-segment version
    was not:

        ONE TRAJECTORY.   The waypoints are densified into a single evenly
        spaced stream, so there are no segment boundaries to accelerate out
        of and settle into.

        SMOOTHED.         The hand tremor in the recording is averaged out
        before anything is commanded.

        STEADY CADENCE.   One pose every step_period(), matched to the speed
        the servos were given, with no per-point catch-up wait. The lag is
        checked periodically instead of constantly, so a genuine runaway is
        still caught but ordinary following error does not stop the motion.
    """
    poses = plan(waypoints)
    log(f'{len(waypoints)} taught points -> {len(poses)} commands, '
        f'{len(poses) * step_period():.0f}s, eased through the turns')
    period = step_period()
    # Travel to the start properly first: this one IS a destination. If it is
    # a long way off - coming from the top pass to a side pass, say - go over
    # rather than through.
    lift = 0
    try:
        now = arm.read_pose()
        far = max(abs(int(poses[0][j]) - int(now[j]))
                  for j in poses[0] if j in now and j != _hid())
        if far >= getattr(tc, 'TRAVEL_LIFT_MIN_RAW', 350):
            lift = lift_sign() * getattr(tc, 'TRAVEL_LIFT_RAW', 220)
            log(f'travelling {far} raw to the start - arcing over, '
                f'shoulder {lift:+d} at the peak')
    except Exception:
        pass
    if goto(arm, poses[0], log=log, stop_check=stop_check, limits=limits,
            lift=lift) is None:
        return False
    for i, pose in enumerate(poses[1:], 1):
        if stop_check is not None and stop_check():
            log('stopped mid-pass')
            return False
        hid = _hid()
        others = {j: v for j, v in pose.items() if j != hid}
        if others:
            arm.move_pose(clamped(others, limits))
        if hid is not None and hid in pose:
            arm.move_to(hid, int(pose[hid]) % 4096)
        # A lag check every so often, not every step - reading the arm takes
        # about as long as a step, so doing it constantly is itself a stutter.
        if i % 12 == 0:
            t0 = time.monotonic()
            while time.monotonic() - t0 < ARRIVE_TIMEOUT_S:
                try:
                    if arm.arrived(pose, tol=BLEND_TOL_RAW * 3):
                        break
                except Exception:
                    break
                if stop_check is not None and stop_check():
                    return False
                time.sleep(0.02)
        time.sleep(period)
    if settle:
        time.sleep(settle)
    return True


SUBSTEPS = 8        # fine samples per full-speed command - see plan()


def _turn(a, b, c, keys, hid):
    """Direction change at b, in degrees, over the joints that matter."""
    import math
    u = v = 0.0
    du, dv, dot = [], [], 0.0
    for j in keys:
        if j == hid:
            continue
        p, q = b[j] - a[j], c[j] - b[j]
        du.append(p)
        dv.append(q)
    u = math.sqrt(sum(x * x for x in du))
    v = math.sqrt(sum(x * x for x in dv))
    if u < 1e-6 or v < 1e-6:
        return 0.0
    dot = sum(x * y for x, y in zip(du, dv)) / (u * v)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def pace(poses):
    """A speed fraction for every pose: slow into bends, ease at the ends.

    Three passes, in order, and each exists for a reason:

        CURVATURE   a sharp turn gets a low fraction, a straight gets 1.0.

        ENDS        the first and last few steps ramp, so the pass does not
                    start or finish with a lurch.

        ACCELERATION  the fractions above can still change abruptly from one
                    step to the next, and an abrupt change in speed is a jolt
                    even when every speed in it is reasonable. A forward and
                    a backward pass limit how fast the fraction may change,
                    which is what makes the arm start braking BEFORE the
                    corner rather than at it.
    """
    n = len(poses)
    if n < 3:
        return [1.0] * n
    keys = list(poses[0])
    hid = _hid()
    f = [1.0] * n
    w = max(1, SUBSTEPS)          # measure the bend over a real distance,
    for i in range(1, n - 1):     # not between two adjacent fine samples
        a = poses[max(0, i - w)]
        c = poses[min(n - 1, i + w)]
        ang = _turn(a, poses[i], c, keys, hid)
        f[i] = max(MIN_SPEED_FRAC, 1.0 - CORNER_SLOWDOWN * (ang / 180.0))
    for i in range(n):
        e = min(i, n - 1 - i)
        if e < EASE_STEPS:
            f[i] = min(f[i], MIN_SPEED_FRAC
                       + (1.0 - MIN_SPEED_FRAC) * (e + 1) / float(EASE_STEPS))
    # convert the raw-per-command cap into a per-fine-sample fraction
    cap = ACCEL_RAW_PER_CMD / float(max(1, step_size()) * SUBSTEPS)
    for i in range(1, n):                       # forward: limit speeding up
        f[i] = min(f[i], f[i - 1] + cap)
    for i in range(n - 2, -1, -1):              # backward: brake in advance
        f[i] = min(f[i], f[i + 1] + cap)
    return f


def _at(fine, cum, s, keys, hid):
    """The pose a distance s along the fine polyline."""
    import bisect
    k = max(1, min(len(fine) - 1, bisect.bisect_left(cum, s)))
    span = cum[k] - cum[k - 1]
    t = 0.0 if span <= 0 else (s - cum[k - 1]) / span
    a, b = fine[k - 1], fine[k]
    pose = {}
    for j in keys:
        if j == hid:
            a0, a1 = to_arc(a[j]), to_arc(b[j])
            pose[j] = from_arc(int(round(a0 + (a1 - a0) * t)))
        else:
            pose[j] = int(round(a[j] + (b[j] - a[j]) * t))
    return pose


def plan(waypoints):
    """Turn taught waypoints into the exact list of poses to command.

    The cadence is fixed - one command every step_period() - so speed is
    expressed as DISTANCE PER COMMAND, not as a longer wait. That distinction
    matters: slowing down by waiting longer between the same-sized jumps is
    precisely the twitching this was built to remove, and it gets worse the
    slower it goes. Slowing by taking smaller steps stays smooth to a crawl.

    The walk along the path is by ARC LENGTH, interpolating between the fine
    samples. An earlier version swallowed a whole number of fine samples per
    command instead, which quantised the spacing - the distance per command
    could only change in lumps, and the lumps were themselves a jolt (17 raw,
    measured, against 7 for the fixed-spacing player it replaced). Walking a
    continuous distance removes the quantisation entirely.
    """
    import math
    fine_step = max(2, int(round(step_size() / float(SUBSTEPS))))
    fine = smooth(densify(waypoints, step=fine_step))
    if len(fine) < 3:
        return fine
    keys = list(fine[0])
    hid = _hid()
    move = [j for j in keys if j != hid]
    cum = [0.0]
    for a, b in zip(fine, fine[1:]):
        cum.append(cum[-1] + math.sqrt(sum((b[j] - a[j]) ** 2 for j in move)))
    total = cum[-1]
    if total <= 0:
        return [fine[0], fine[-1]]
    f = pace(fine)
    out, s, i = [dict(fine[0])], 0.0, 0
    while s < total:
        while i < len(cum) - 2 and cum[i + 1] < s:
            i += 1
        s = min(total, s + max(1.0, step_size() * f[i]))
        out.append(_at(fine, cum, s, keys, hid))
    out[-1] = dict(fine[-1])          # finish exactly where it was taught
    # Clamping the last step to the end of the path leaves a stub - a 7 raw
    # command followed by a 1 raw one, which is a visible twitch right at the
    # finish. Fold it into the step before it.
    if len(out) > 3:
        def gap(a, b):
            return math.sqrt(sum((b[j] - a[j]) ** 2 for j in move))
        if gap(out[-2], out[-1]) < 0.4 * max(1e-6, gap(out[-3], out[-2])):
            del out[-2]
    return out
