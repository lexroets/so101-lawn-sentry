"""Remember where the trimmer has already cut, so it stops going back.

Without this the loop is: find the tallest patch, cut it, look again, find the
same patch (now shorter but still proud of the deck), cut it again. It looks
like the machine is working and it is actually stuck, and the giveaway is that
it never moves on to the second patch.

The fix needs somewhere to record "done", and the awkward part is what
coordinates to record it in. The trimmer has no world map - it is bolted to a
bench, not driving around - so there is nothing like the lawn plane the poop
scanner uses. But it does not need one. Two numbers place a target completely:

    the BASE ANGLE that centres it   - where it is left to right
    the DEPTH to it                  - how far away it is

Both are already measured for every detection, and both are in the arm's own
frame, so they stay valid however the camera is pointed. A cleared zone is a
circle in that space, and a new detection landing inside one is the patch that
was just cut rather than a new one.

Cleared zones fade. Grass grows back, someone moves the target, or the cut was
not as complete as it looked - so a zone that has not been touched for
FORGET_AFTER_S becomes fair game again rather than being excluded forever.
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

# How close a detection has to be to a cleared zone to count as the same thing.
# In base ticks and metres respectively.
#
# The angular one is the looser of the two on purpose: a patch that has just
# been cut looks smaller, so its centroid shifts, and the base angle that
# centres it moves with it. Being too tight here is what lets the same patch
# come back under a new name.
SAME_SPOT_RAW = 260          # about 23 degrees of base rotation
SAME_SPOT_M = 0.05           # 5 cm in depth

FORGET_AFTER_S = 240.0       # grass grows; nothing is cleared forever


class ClearedZones:
    """Where the trimmer has already been. Cheap to check, easy to reset."""

    def __init__(self, same_raw=SAME_SPOT_RAW, same_m=SAME_SPOT_M,
                 forget_s=FORGET_AFTER_S):
        self.same_raw = same_raw
        self.same_m = same_m
        self.forget_s = forget_s
        self.zones = []          # {'base','z','at','hits'}

    # -- writing -------------------------------------------------------------
    def mark(self, base_raw, z):
        """Record that this spot has been cut.

        Cutting the same place twice refreshes the existing zone rather than
        stacking a second one on top of it, so the list stays short.
        """
        hit = self._find(base_raw, z)
        if hit is not None:
            hit['at'] = time.monotonic()
            hit['hits'] += 1
            return hit
        z0 = {'base': float(base_raw), 'z': float(z),
              'at': time.monotonic(), 'hits': 1}
        self.zones.append(z0)
        return z0

    # -- reading -------------------------------------------------------------
    def _find(self, base_raw, z):
        now = time.monotonic()
        self.zones = [c for c in self.zones if now - c['at'] < self.forget_s]
        for c in self.zones:
            if (abs(c['base'] - base_raw) <= self.same_raw
                    and abs(c['z'] - z) <= self.same_m):
                return c
        return None

    def is_cleared(self, base_raw, z):
        return self._find(base_raw, z) is not None

    def age_of(self, base_raw, z):
        c = self._find(base_raw, z)
        return None if c is None else time.monotonic() - c['at']

    def forget_all(self):
        n = len(self.zones)
        self.zones = []
        return n

    def forget_nearest(self, base_raw, z):
        c = self._find(base_raw, z)
        if c is not None:
            self.zones.remove(c)
        return c is not None

    def __len__(self):
        now = time.monotonic()
        return len([c for c in self.zones if now - c['at'] < self.forget_s])

    def summary(self):
        n = len(self)
        if not n:
            return 'nothing cleared yet'
        return f'{n} area(s) cleared'


def base_for_detection(u, base_raw, centre_x, raw_per_px, direction_sign):
    """The base angle that would centre this detection.

    This is what makes a detection comparable with a cleared zone: the same
    patch seen from a slightly different base position gives the same answer,
    because the pixel offset and the base offset cancel.
    """
    return float(base_raw) + (float(u) - centre_x) * raw_per_px * direction_sign
