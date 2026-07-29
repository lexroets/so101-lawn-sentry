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

# A survey should look deliberate and keep the picture sharp. These are much
# slower than the trimmer's own moves.
SURVEY_SPEED = 45           # steps/s - about 4 deg/s, a deliberate crawl
SURVEY_ACC = 4              # very gentle ramp, no jerk at the turn-round
NUDGE_SPEED = 900        # was 260 - a 4 deg nudge took a noticeable moment
NUDGE_ACC = 45

# How far one button press moves each joint.
NUDGE_DEG = 6.0
TICKS_PER_DEG = 4096.0 / 360.0

# Stop sweeping if nothing has checked in for this long, so a closed browser
# does not leave the arm creeping.
SWEEP_IDLE_TIMEOUT_S = 25.0


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
        # base_deg() talks to the servo. This is published several times a
        # second, and every one of those reads competes with the scan loop for
        # the same serial bus. Three a second is plenty for a readout.
        now = time.monotonic()
        if now - self._cache[0] > 0.5:
            ArmControl._cache = (now, self.base_deg())
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

    def go_home(self):
        if not self.ok or not self.home:
            return False
        with self.lock:
            try:
                self.arm.set_profile(NUDGE_SPEED, NUDGE_ACC)
                self.arm.move_pose(self.home)
                self._target = dict(self.home)
                return True
            except Exception as e:
                self.error = str(e)
                return False

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
        self._stop.set()
        self.sweeping = False
        self._target = {}          # the sweep moved things; re-read on next press
        with self.lock:
            try:                      # hold position rather than drifting on
                cur = self.arm.read_position(self.pan_joint())
                if cur is not None:
                    self.arm.move_to(self.pan_joint(), int(cur))
            except Exception:
                pass
        return True

    def ping(self):
        """Tell the sweep someone is still watching."""
        self._last_ping = time.monotonic()

    def _sweep_loop(self):
        try:
            with self.lock:
                self.arm.set_profile(SURVEY_SPEED, SURVEY_ACC, ids=[self.pan_joint()])
            target = self.sweep_hi
            while not self._stop.is_set():
                if time.monotonic() - self._last_ping > SWEEP_IDLE_TIMEOUT_S:
                    break
                with self.lock:
                    try:
                        self.arm.move_to(self.pan_joint(), int(target))
                        cur = self.arm.read_position(self.pan_joint())
                    except Exception as e:
                        self.error = str(e)
                        break
                if cur is not None and abs(int(cur) - int(target)) < 25:
                    target = self.sweep_lo if target == self.sweep_hi else self.sweep_hi
                self._stop.wait(0.15)
        finally:
            self.sweeping = False
            with self.lock:
                try:
                    self.arm.set_profile(NUDGE_SPEED, NUDGE_ACC, ids=[self.pan_joint()])
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
