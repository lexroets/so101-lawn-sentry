"""Where is the camera pointed, right now?

Visual tracking answers this by comparing consecutive frames, which means every
frame's small error is added to the last. Measured on this setup, that reaches
8 px of displacement after 5 degrees of pan and 537 px after 45 - it grows
without bound, and the usual accuracy checks cannot see it because a slow shift
does not distort anything.

An encoder does not have that problem. It reports an ABSOLUTE angle, so the
error after 45 degrees is the same as after 5. Nothing is chained, so nothing
accumulates.

Sources, all behind the same `.angle_deg()`:

    ArmPan()        the SO-101 base servo
    FixedPan()      a camera that does not move (always 0)
    ManualPan(f)    anything else - supply a callable

Most PTZ security cameras report their pan position over ONVIF, so the same
approach extends to them; add a class here and the calibration works unchanged.
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

# Feetech STS servos report 0..4095 over a full turn.
TICKS_PER_TURN = 4096.0
DEG_PER_TICK = 360.0 / TICKS_PER_TURN


class PanSource:
    """Base class. `angle_deg()` returns degrees, or None if unavailable."""
    name = 'none'

    def angle_deg(self):
        raise NotImplementedError

    def ok(self):
        return self.angle_deg() is not None

    def close(self):
        pass


class FixedPan(PanSource):
    """A camera that does not move."""
    name = 'fixed'

    def angle_deg(self):
        return 0.0


class ManualPan(PanSource):
    name = 'manual'

    def __init__(self, fn):
        self._fn = fn

    def angle_deg(self):
        try:
            return float(self._fn())
        except Exception:
            return None


class ArmPan(PanSource):
    """Pan angle from the SO-101 base joint.

    READ ONLY. This never enables torque and never commands a movement - you
    pan the arm, this just records where it is. A calibration tool has no
    business moving the hardware.
    """
    name = 'so101-base'

    def __init__(self, arm=None, zero_raw=None, sign=None):
        import trimmer_config as tc
        self.tc = tc
        self.zero_raw = tc.BASE_FORWARD_RAW if zero_raw is None else zero_raw
        self.sign = tc.BASE_DIRECTION_SIGN if sign is None else sign
        self.base_id = tc.BASE_ID
        self._owned = arm is None
        self.arm = arm
        self.last_raw = None
        if self.arm is None:
            from trimmer_arm import TrimmerArm
            self.arm = TrimmerArm()

    def raw(self):
        try:
            r = self.arm.read_position(self.base_id)
        except Exception:
            return None
        # a disconnected bus tends to return 0 or a wild value; reject those
        if r is None or not (0 <= r <= 4095):
            return None
        self.last_raw = int(r)
        return int(r)

    def angle_deg(self):
        r = self.raw()
        if r is None:
            return None
        return float(self.sign) * (r - self.zero_raw) * DEG_PER_TICK

    def close(self):
        if self._owned and self.arm is not None:
            try:
                self.arm.close()
            except Exception:
                pass


def open_pan(kind='auto', arm=None):
    """Best available pan source, borrowing `arm` if one is supplied.

    Never raises - falls back to a fixed source so a missing arm degrades
    instead of stopping the scan.
    """
    if kind in ('arm', 'so101', 'auto'):
        try:
            p = ArmPan(arm=arm)
            if p.angle_deg() is not None:
                return p
            p.close()
        except Exception as e:
            if kind != 'auto':
                print(f'[pan] could not read the SO-101 base: {e}')
    if kind in ('fixed', 'auto', 'none'):
        return FixedPan()
    return FixedPan()
