"""Point the camera: manual nudges and a slow survey sweep.

The camera rides on the SO-101, so "look over there" is a servo move. This
wraps that in something safe enough to expose to a web button:

* every target is clamped to the limits in trimmer_config before it is sent
* speeds are deliberately low - a survey wants smooth, not quick, and a slow
  pan also keeps the detector's frames sharp
* one lock serialises commands, so a flurry of button presses cannot interleave
  two half-written positions on the bus
* an idle sweep stops on its own if nothing keeps it alive

Nothing here runs unless something asks it to. `scan_lawn` creates one of these
only when the servo actually answers.
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
import json
import threading
import time

import trimmer_config as tc

# --- how fast the survey pans ----------------------------------------------
# THIS is the number to change, and it is in degrees per second, which is the
# only unit anyone can reason about while watching the arm.
#
# It used to be a servo speed register value, set once at the start of the
# sweep, with the far end of the arc as the target - so the actual pan rate
# was whatever the servo decided that register meant. The Feetech datasheets
# disagree about the unit (steps/s in some, tenths of an RPM in others), and
# the difference between those readings is a factor of fifty. Guessing wrong
# gives a sweep that races across the lawn, which is exactly what happened.
#
# So the rate is no longer delegated. The loop below steps the goal position
# by a small amount on a timer and works out that amount from the clock, which
# makes the pan rate arithmetic we control rather than a register we hope we
# understand.
SWEEP_DEG_PER_S = 2.5       # a slow, deliberate survey
SWEEP_TICK_S = 0.12         # how often the goal is nudged along
SWEEP_MIN_STEP_DEG = 0.05   # never command a step too small to move the servo

# The servo profile during a sweep only has to be quick enough to reach each
# small step before the next one arrives; the STEP SIZE sets the speed, not
# this. Too slow here and the motion stutters, too fast and each step snaps.
SURVEY_FOLLOW_SPEED = 300
SURVEY_ACC = 8              # gentle ramp, no jerk at the turn-round

NUDGE_SPEED = 900        # was 260 - a 4 deg nudge took a noticeable moment
NUDGE_ACC = 45

# How far one button press moves each joint.
NUDGE_DEG = 6.0
TICKS_PER_DEG = 4096.0 / 360.0

# Stop sweeping if nothing has checked in for this long, so a closed browser
# does not leave the arm creeping.
SWEEP_IDLE_TIMEOUT_S = 25.0

# --- passing the origin is a free chance to cancel drift --------------------
# The lawn plane was measured at ONE camera pose, and visual tracking carries
# it to every other pose by chaining frame-to-frame homographies. Chaining
# accumulates: each estimate is nearly right, and "nearly" compounds, so the
# outline creeps off the grass over a few minutes.
#
# But a sweep returns through the pose the plane was measured at, twice a
# cycle. At that instant the calibration is exactly true again, so the
# accumulated chain can simply be thrown away rather than corrected. Drift
# never gets more than half a sweep to build up.
#
# The tolerance is a compromise: too tight and a sweep step can jump straight
# over the window between ticks; too loose and we re-anchor while genuinely
# off-centre, which INTRODUCES the error we are trying to remove. At 2.5 deg/s
# and a 0.12 s tick the arm moves 0.3 deg per step, so a 1 degree window is
# always caught and is worth at most 1 degree of error.
AT_HOME_TOL_DEG = 1.0
AT_HOME_MIN_GAP_S = 4.0     # do not re-anchor more often than this


def deg_to_raw(deg):
    return int(round(deg * TICKS_PER_DEG))


def load_map():
    """Which joint moves the camera which way, for THIS mounting.

    Written by calibrate_joints.py. Falls back to the layout that suited the
    original mounting, which is a guess and says so - a button that moves the
    wrong joint is worse than one that admits it does not know.
    """
    from paths import DATA
    p = DATA / 'arm_map.json'
    try:
        j = json.loads(p.read_text())
        roles = j.get('roles') or {}
        if roles:
            # the wizard MEASURES which way the map should turn by nudging the
            # arm and watching the picture, rather than anyone reasoning about
            # signs - which is what got it backwards before
            if j.get('map_pan_sign') is not None:
                roles = dict(roles)
                roles['_map_sign'] = int(j['map_pan_sign'])
            return roles, True
    except (OSError, ValueError):
        pass
    return ({'pan_right': {'joint': 'base', 'sign': -1},
             'tilt_down': {'joint': 'wrist', 'sign': -1},
             'lift_up': {'joint': 'shoulder', 'sign': 1}}, False)


class ArmControl:
    """Manual jogging plus a slow survey sweep. Safe to call from any thread."""

    def __init__(self, arm=None, enable=True):
        self.lock = threading.RLock()
        # If a connection is handed in, we BORROW it - opening a second one on
        # the same port makes both readers fail intermittently.
        self._owned = arm is None
        if arm is None:
            from trimmer_arm import TrimmerArm
            arm = TrimmerArm()
        self.arm = arm
        self.ok = True
        self.sweeping = False
        self.sweep_lo = None
        self.sweep_hi = None
        self._dir = 1
        self._last_ping = time.monotonic()
        self._thread = None
        self._stop = threading.Event()
        self.error = None
        self._target = {}          # where we last commanded each joint
        # Called whenever the camera is back at the pose the lawn plane was
        # measured at. Whoever owns the ground tracker can use that moment to
        # throw away accumulated drift - see AT_HOME_TOL_DEG below.
        self.on_at_home = None
        self._last_home_call = 0.0
        self.roles, self.mapped = load_map()
        if not self.mapped:
            print('[arm] no joint map yet - the direction buttons are GUESSING.')
            print('      Fix in a minute:  python lawn_sentry/calibrate/calibrate_joints.py')
        try:
            self.home = self.arm.read_pose()
        except Exception as e:
            self.ok = False
            self.error = str(e)
            self.home = {}
        if self.ok and enable:
            try:
                self.arm.clamp_enabled = False   # camera needs the full range
                self.arm.enable()
                self.arm.set_profile(NUDGE_SPEED, NUDGE_ACC)   # once, not per press
                try:
                    self._target = dict(self.home)
                except Exception:
                    self._target = {}
            except Exception as e:
                self.ok = False
                self.error = str(e)

    # -- state ---------------------------------------------------------------
    def pose(self):
        with self.lock:
            try:
                return self.arm.read_pose()
            except Exception:
                return {}

    def base_deg(self):
        """Pan angle relative to WHERE THE ARM STARTED, not a config constant.

        BASE_FORWARD_RAW was measured on the old mounting; this arm rests near
        raw 2966, which made the readout claim ~85 degrees while pointing
        straight at the lawn - and the map cone drew sideways at rest. The pose
        at startup IS the aimed-at-the-lawn pose, so it is the natural zero.
        """
        sid = self.pan_joint()
        with self.lock:
            try:
                r = self.arm.read_position(sid)
            except Exception:
                return None
        if r is None:
            return None
        zero = self.home.get(sid, tc.BASE_FORWARD_RAW)
        # CONVENTION: positive = the camera has panned RIGHT.
        # calibrate_joints recorded which servo direction pans right, so that
        # sign converts raw ticks into this convention directly. It was being
        # negated here AND again when the map turned it into a heading, so the
        # cone swung opposite to the arm.
        sgn = int(self.roles.get('_map_sign',
                                 (self.roles.get('pan_right') or {}).get('sign', 1)))
        return (r - zero) * 360.0 / 4096.0 * sgn

    _cache = (0.0, None)

    def status(self):
        # base_deg() talks to the servo, and this is published several times a
        # second: every one of those reads is a round trip on a bus the scan
        # loop and the sweep also want.
        #
        # Almost always we already KNOW the answer, because we commanded it -
        # nudges and the sweep both record their target. So read the servo only
        # when nothing has been commanded yet, or once in a while to catch the
        # arm being moved by hand. Between those, answer from the cache.
        now = time.monotonic()
        stale = now - self._cache[0]
        known = self._target.get(self.pan_joint()) is not None
        if stale > (3.0 if (known and not self.sweeping) else 0.5):
            ArmControl._cache = (now, self.base_deg())
        elif self._cache[1] is None and known:
            zero = self.home.get(self.pan_joint(), 0)
            sgn = int(self.roles.get('_map_sign',
                                     (self.roles.get('pan_right') or {}).get('sign', 1)))
            ArmControl._cache = (
                now, (self._target[self.pan_joint()] - zero) * 360.0 / 4096.0 * sgn)
        return {'ok': self.ok, 'sweeping': self.sweeping,
                'base_deg': self._cache[1], 'error': self.error,
                'mapped': self.mapped, 'roles': self.roles,
                'sweep_lo': self.sweep_lo, 'sweep_hi': self.sweep_hi}

    # -- manual jogging ------------------------------------------------------
    # What the buttons mean, in terms of what the PICTURE does. Each maps to a
    # role, and the joint map says which servo delivers it on this mounting.
    ACTIONS = {'right': ('pan_right', 1), 'left': ('pan_right', -1),
               'down': ('tilt_down', 1), 'up': ('tilt_down', -1),
               'tilt_down': ('tilt_down', 1), 'tilt_up': ('tilt_down', -1),
               'raise': ('lift_up', 1), 'lower': ('lift_up', -1)}

    def act(self, action, degrees=NUDGE_DEG):
        """Move the camera the way the user asked, whatever joint that takes.

        Also refreshes the cached angle from the value we just commanded, so
        the map turns immediately instead of waiting for the next servo read.
        """
        spec = self.ACTIONS.get(action)
        if spec is None:
            return False
        role, direction = spec
        r = self.roles.get(role)
        if not r:
            return False
        ok = self.nudge(r['joint'], degrees * direction * int(r.get('sign', 1)))
        if ok and role == 'pan_right':
            sid = self.pan_joint()
            tgt = self._target.get(sid)
            if tgt is not None:
                zero = self.home.get(sid, tgt)
                sgn = int(self.roles.get('_map_sign',
                                         (self.roles.get('pan_right') or {}).get('sign', 1)))
                ArmControl._cache = (time.monotonic(),
                                     (tgt - zero) * 360.0 / 4096.0 * sgn)
        return ok

    def pan_joint(self):
        """The servo that pans, for the survey sweep."""
        r = self.roles.get('pan_right') or {}
        return {'base': tc.BASE_ID, 'shoulder': tc.SHOULDER_ID,
                'elbow': tc.ELBOW_ID, 'wrist': tc.WRIST_ID}.get(
                    r.get('joint', 'base'), tc.BASE_ID)

    def nudge(self, joint, degrees):
        """Move one joint by a small amount.

        Kept to ONE bus write. It used to read the servo's position and set the
        motion profile on every press - three extra transactions, one of them a
        blocking round trip with an 80 ms timeout. That is what made the
        controls feel a beat behind: the delay was on the serial bus, not in
        the browser. The profile is set once at startup, and the target is
        tracked locally, so a press is now a single write.
        """
        if not self.ok:
            return False
        sid = {'base': tc.BASE_ID, 'shoulder': tc.SHOULDER_ID,
               'elbow': tc.ELBOW_ID, 'wrist': tc.WRIST_ID}.get(joint)
        if sid is None:
            return False
        with self.lock:
            try:
                cur = self._target.get(sid)
                if cur is None:                     # first press: ask once
                    cur = self.arm.read_position(sid)
                    if cur is None:
                        return False
                tgt = max(0, min(4095, int(cur) + deg_to_raw(degrees)))
                self.arm.move_to(sid, tgt)
                self._target[sid] = tgt
                return True
            except Exception as e:
                self.error = str(e)
                return False

    def _maybe_at_home(self, sid, pos, why):
        """Fire the callback when the pan joint is back at its calibrated pose."""
        cb = self.on_at_home
        if cb is None:
            return
        home = self.home.get(sid)
        if home is None:
            return
        if abs(float(pos) - float(home)) > AT_HOME_TOL_DEG * TICKS_PER_DEG:
            return
        now = time.monotonic()
        if now - self._last_home_call < AT_HOME_MIN_GAP_S:
            return
        self._last_home_call = now
        try:
            cb(why)
        except Exception as e:
            self.error = str(e)

    def go_home(self):
        if not self.ok or not self.home:
            return False
        self.stop_sweep()          # a sweep would drag it straight back off
        with self.lock:
            try:
                self.arm.set_profile(NUDGE_SPEED, NUDGE_ACC)
                self.arm.move_pose(self.home)
                self._target = dict(self.home)
            except Exception as e:
                self.error = str(e)
                return False
        # Tell the tracker only once the arm has ACTUALLY arrived. This used to
        # be a fixed guess of a few seconds made by the caller; if the arm was
        # slower than the guess, the plane was re-anchored to a view taken
        # mid-swing, which is worse than not re-anchoring at all. Watching for
        # arrival costs one short-lived thread and removes the guess.
        def _watch():
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10.0:
                with self.lock:
                    try:
                        done = self.arm.arrived(self.home, tol=25)
                    except Exception:
                        done = False
                if done:
                    time.sleep(0.4)           # let the last wobble settle
                    self._last_home_call = 0.0        # a HOME press always counts
                    self._maybe_at_home(self.pan_joint(),
                                        self.home.get(self.pan_joint(), 0),
                                        'HOME pressed - arm back at rest')
                    return
                time.sleep(0.2)
        threading.Thread(target=_watch, daemon=True).start()
        return True

    # -- survey sweep --------------------------------------------------------
    def start_sweep(self, span_deg=None, centre_deg=None):
        """Pan slowly back and forth so the camera surveys the whole lawn."""
        if not self.ok or self.sweeping:
            return False
        with self.lock:
            try:
                cur = self.arm.read_position(self.pan_joint())
            except Exception:
                return False
        if cur is None:
            return False
        centre = cur if centre_deg is None else \
            tc.BASE_FORWARD_RAW + deg_to_raw(centre_deg)
        half = deg_to_raw((span_deg or 50.0) / 2.0)
        lo = max(0, int(centre) - half)
        hi = min(4095, int(centre) + half)
        if hi - lo < deg_to_raw(4):
            return False
        self.sweep_lo, self.sweep_hi = lo, hi
        self._stop.clear()
        self._last_ping = time.monotonic()
        self.sweeping = True
        self._thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._thread.start()
        return True

    def stop_sweep(self):
        """Stop where it is. No read needed - the sweep tracked its own goal."""
        self._stop.set()
        self.sweeping = False
        sid = self.pan_joint()
        with self.lock:
            try:                      # hold position rather than drifting on
                tgt = self._target.get(sid)
                if tgt is None:
                    tgt = self.arm.read_position(sid)
                if tgt is not None:
                    self.arm.move_to(sid, int(tgt))
                    self._target[sid] = int(tgt)
            except Exception:
                pass
        return True

    def ping(self):
        """Tell the sweep someone is still watching."""
        self._last_ping = time.monotonic()

    def _sweep_loop(self):
        """Walk the goal position along at a known number of degrees a second.

        Two things changed here, and both matter.

        The RATE is now ours. Instead of commanding the far end of the arc and
        letting the servo's speed register decide how fast to get there, the
        goal is nudged along by (rate x elapsed time) each tick. Degrees per
        second means degrees per second, whatever the register's units turn
        out to be.

        And there is NO SERVO READ in here. The old loop did a blocking
        read_position every tick while holding the bus lock, and a read is a
        round trip with an 80 ms timeout. The dashboard shares that same lock
        to publish arm status, so every sweep tick was stalling the web page -
        which is why panning felt like it was fighting the interface. The loop
        knows where it commanded the arm to go, so it does not need to ask.
        """
        sid = self.pan_joint()
        try:
            with self.lock:
                self.arm.set_profile(SURVEY_FOLLOW_SPEED, SURVEY_ACC, ids=[sid])
                try:
                    pos = float(self.arm.read_position(sid))   # once, at the start
                except Exception:
                    pos = float(self._target.get(sid, (self.sweep_lo + self.sweep_hi) / 2))
            lo, hi = float(self.sweep_lo), float(self.sweep_hi)
            pos = min(max(pos, lo), hi)
            direction = 1.0 if pos < (lo + hi) / 2 else -1.0
            last = time.monotonic()
            while not self._stop.is_set():
                if time.monotonic() - self._last_ping > SWEEP_IDLE_TIMEOUT_S:
                    break
                now = time.monotonic()
                dt, last = now - last, now
                step = max(SWEEP_MIN_STEP_DEG,
                           SWEEP_DEG_PER_S * min(dt, 0.5)) * TICKS_PER_DEG
                pos += direction * step
                if pos >= hi:
                    pos, direction = hi, -1.0
                elif pos <= lo:
                    pos, direction = lo, 1.0
                with self.lock:
                    try:
                        self.arm.move_to(sid, int(round(pos)))
                    except Exception as e:
                        self.error = str(e)
                        break
                # Keep the cached angle honest from what we COMMANDED, so the
                # map's view cone follows the sweep without anyone reading the
                # servo. This is the same trick act() uses after a nudge.
                self._target[sid] = int(round(pos))
                self._maybe_at_home(sid, pos, 'sweep passed the origin')
                zero = self.home.get(sid, self._target[sid])
                sgn = int(self.roles.get('_map_sign',
                                         (self.roles.get('pan_right') or {}).get('sign', 1)))
                ArmControl._cache = (now, (pos - zero) * 360.0 / 4096.0 * sgn)
                self._stop.wait(SWEEP_TICK_S)
        finally:
            self.sweeping = False
            with self.lock:
                try:
                    self.arm.set_profile(NUDGE_SPEED, NUDGE_ACC, ids=[sid])
                except Exception:
                    pass

    # -- shutdown ------------------------------------------------------------
    def close(self, relax=False):
        self.stop_sweep()
        if self._owned and self.arm is not None:
            try:
                if relax:
                    self.arm.torque_off()
                self.arm.close()
            except Exception:
                pass
