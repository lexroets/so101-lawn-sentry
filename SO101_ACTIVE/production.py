"""Production reliability helpers for the SO-101 sorter.

Four small, independent pieces, each gated by a config flag (all default to
the safe choice). main.py calls into these at a handful of hook points; the
logic lives here so main.py stays close to the demo build and every feature
can be turned off from config.py.

    GrabVerifier   — detect an empty grip and retry / cool down the spot
    Watchdog       — recover from hung phases and transient serial errors
    SelfCheck      — verify camera + servos + calibrations before moving
    SessionMetrics — count attempts/placed/empties and write a run summary
"""
import os
import time

from config import (
    ARM_JOINT_IDS, BASE_SERVO_ID, GRIPPER_ID, GRIPPER_CLOSED_RAW,
    GRAB_RETRY_ENABLED, GRAB_EMPTY_MARGIN, GRAB_MAX_RETRIES,
    GRAB_COOLDOWN_S, GRAB_COOLDOWN_RADIUS_PX,
    WATCHDOG_ENABLED, WATCHDOG_PHASE_MAX_S, WATCHDOG_MAX_CONSEC_ERRORS,
    METRICS_ENABLED,
)


# ---------------------------------------------------------------------------
class GrabVerifier:
    """Was there actually a battery in the gripper after it closed?

    The gripper closes to ~GRIPPER_CLOSED_RAW on empty air; a held battery
    stops the fingers well above that. So a settled position at/near fully
    closed means the grab missed."""

    def __init__(self, log):
        self.log = log
        self.retries = 0
        self._cooldown = []          # (u, v, expiry_time)

    def is_empty(self, gripped_at):
        if gripped_at is None or GRIPPER_CLOSED_RAW is None:
            return False
        return gripped_at <= GRIPPER_CLOSED_RAW + GRAB_EMPTY_MARGIN

    def verdict(self, gripped_at, point, now):
        """'proceed' (battery held) · 'retry' (missed, try again) ·
        'skip' (missed too often — cool the spot and move on)."""
        if not GRAB_RETRY_ENABLED:
            return 'proceed'
        if not self.is_empty(gripped_at):
            self.retries = 0
            return 'proceed'
        self.retries += 1
        self.log(f"EMPTY GRIP (raw {gripped_at} <= closed+"
                 f"{GRAB_EMPTY_MARGIN}) — miss {self.retries}/"
                 f"{GRAB_MAX_RETRIES}")
        if self.retries <= GRAB_MAX_RETRIES:
            return 'retry'
        self.retries = 0
        if point is not None:
            self._cooldown.append((point[0], point[1],
                                   now + GRAB_COOLDOWN_S))
            self.log("EMPTY GRIP — giving up on this spot, cooling it down")
        return 'skip'

    def cooling(self, u, v, now):
        """True if this location recently failed too many times."""
        self._cooldown = [c for c in self._cooldown if c[2] > now]
        return any(abs(cu - u) <= GRAB_COOLDOWN_RADIUS_PX
                   and abs(cv - v) <= GRAB_COOLDOWN_RADIUS_PX
                   for cu, cv, _ in self._cooldown)


# ---------------------------------------------------------------------------
class Watchdog:
    """Backstop for unattended running: recover from hung phases and ride out
    transient serial/camera errors instead of crashing."""

    # Phases where waiting indefinitely is normal — never time these out.
    SAFE_PHASES = frozenset(('IDLE', 'SEL_BOX', 'HOME', 'GRIPPED'))

    def __init__(self, log):
        self.log = log
        self.consec_errors = 0

    def phase_hung(self, phase, phase_start, now):
        if not WATCHDOG_ENABLED or phase in self.SAFE_PHASES:
            return False
        return (now - phase_start) >= WATCHDOG_PHASE_MAX_S

    def note_ok(self):
        self.consec_errors = 0

    def note_error(self, exc):
        self.consec_errors += 1
        self.log(f"RECOVERABLE ERROR ({self.consec_errors}/"
                 f"{WATCHDOG_MAX_CONSEC_ERRORS}): {exc}")
        return self.consec_errors < WATCHDOG_MAX_CONSEC_ERRORS


# ---------------------------------------------------------------------------
class SelfCheck:
    """Verify the rig before the arm is allowed to move."""

    def __init__(self, log):
        self.log = log
        self.problems = []
        self.warnings = []

    def run(self, actuator, first_frame_ok):
        # Camera: did a frame actually arrive?
        if not first_frame_ok:
            self.problems.append('camera: no frame received at startup')
        # Servos: every arm joint + base + gripper must answer.
        for sid in (BASE_SERVO_ID, *ARM_JOINT_IDS, GRIPPER_ID):
            try:
                if actuator.read_position(sid) is None:
                    self.problems.append(f'servo {sid}: no response')
            except Exception as exc:
                self.problems.append(f'servo {sid}: {exc}')
        # Calibrations: missing REQUIRED ones fail; optional ones warn.
        here = os.path.dirname(os.path.abspath(__file__))

        def present(name):
            return os.path.isfile(os.path.join(here, name))

        if not present('reach_calibration.py'):
            self.problems.append('reach_calibration.py missing (arm cannot '
                                 'reach targets) — run calibrate_full.py')
        if not present('gripper_calibration.py'):
            self.warnings.append('gripper_calibration.py missing — grips '
                                 'will be skipped')
        for opt, why in (('buckets_calibration.py', 'bucket sorting'),
                         ('lying_calibration.py', 'top-down lying grabs'),
                         ('roll_calibration.py', 'auto wrist-roll')):
            if not present(opt):
                self.warnings.append(f'{opt} missing — {why} disabled')

        for w in self.warnings:
            self.log('SELFCHECK warn: ' + w)
        for p in self.problems:
            self.log('SELFCHECK FAIL: ' + p)
        ok = not self.problems
        self.log('SELFCHECK ' + ('PASSED' if ok else 'FAILED'))
        return ok


# ---------------------------------------------------------------------------
class SessionMetrics:
    """Count what happened and leave a per-run record."""

    def __init__(self, log, path=None):
        self.log = log
        self.t0 = time.monotonic()
        self.attempts = 0
        self.placed = 0
        self.empties = 0
        self.ryobi = 0
        self.pink = 0
        self.path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'session_metrics.txt')

    def attempt(self):
        if METRICS_ENABLED:
            self.attempts += 1

    def empty(self):
        if METRICS_ENABLED:
            self.empties += 1

    def placed_ok(self, colour):
        if not METRICS_ENABLED:
            return
        self.placed += 1
        if colour == 'ryobi':
            self.ryobi += 1
        elif colour == 'pink':
            self.pink += 1

    def summary(self):
        if not METRICS_ENABLED:
            return
        mins = (time.monotonic() - self.t0) / 60.0
        rate = self.placed / max(1, self.attempts)
        line = (f"placed={self.placed} attempts={self.attempts} "
                f"({rate:.0%}) empty_grips={self.empties} "
                f"ryobi={self.ryobi} pink={self.pink} "
                f"runtime={mins:.1f}min")
        self.log('SESSION ' + line)
        try:
            with open(self.path, 'a') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S  ') + line + '\n')
        except OSError:
            pass
