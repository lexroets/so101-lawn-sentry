"""The taught reach table, and whether it can be trusted at a given distance.

The table maps a measured distance to an arm pose that was physically taught
at that distance. It is honest inside the range it was taught over and quietly
wrong outside it, and that asymmetry caused a real bug: targets beyond the last
taught point were clamped to the furthest pose, so the arm under-reached by a
fixed amount with nothing in the logs to explain it.

Two failure modes, both checkable without moving anything:

    OUTSIDE THE RANGE   clamped to an end pose. Always short, never long, and
                        by exactly the distance past the edge.

    A REVERSED SEGMENT  reaching further means the shoulder rises and the elbow
                        opens. A captured point that breaks that trend makes
                        the head move BACKWARDS across that stretch, so a
                        target further away is reached less far than a nearer
                        one.

Both are properties of the table alone, so they are checked once at startup
rather than discovered by watching the arm miss.
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
import trimmer_config as tc

SHOULDER, ELBOW, WRIST = tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID


def reversed_segments(table):
    """Stretches where the head retracts as the target gets further away.

    Only flagged when the shoulder drops AND the elbow opens together. Either
    one alone can be legitimate depending on the arm's geometry; both at once
    cannot be, and being cautious here matters - a false alarm on a good table
    would teach the user to ignore the warning.
    """
    pts = sorted(table, key=lambda p: p[0])
    out = []
    for (za, pa), (zb, pb) in zip(pts, pts[1:]):
        if not all(s in pa and s in pb for s in (SHOULDER, ELBOW)):
            continue
        if pb[SHOULDER] < pa[SHOULDER] and pb[ELBOW] > pa[ELBOW]:
            out.append((za, zb))
    return out


def coverage(table):
    """(nearest, furthest) distance the table was taught over."""
    pts = sorted(table, key=lambda p: p[0])
    return pts[0][0], pts[-1][0]


def verdict(z, table):
    """How much to trust the pose for a target at z.

    Returns (ok, short_cm, message). `short_cm` is how far the head will stop
    short - positive means it under-reaches - or None when it is not simply a
    clamping error.
    """
    lo, hi = coverage(table)
    if z > hi:
        return False, (z - hi) * 100.0, (
            f'{(z-hi)*100:.0f} cm beyond the taught table')
    if z < lo:
        return False, None, f'{(lo-z)*100:.0f} cm nearer than the taught table'
    for a, b in reversed_segments(table):
        if a <= z <= b:
            return False, None, 'in a reversed segment of the table'
    return True, 0.0, 'inside the taught range'


def report(table, out=print):
    """Print the audit once at startup. Returns True if the table looks sane."""
    pts = sorted(table, key=lambda p: p[0])
    lo, hi = coverage(table)
    bad = reversed_segments(table)
    out(f'[reach] table taught over {lo:.2f}-{hi:.2f} m, {len(pts)} points')
    if bad:
        for a, b in bad:
            out(f'[reach] WARNING between {a:.3f} and {b:.3f} m the shoulder '
                f'drops while the elbow opens -')
            out(f'        the head moves BACK as the target gets further away. '
                f'That point looks mis-taught.')
        out('[reach] re-teach it: python trimmer/calibrate/calibrate_reach.py')
    out('[reach] targets outside that range are CLAMPED to an end pose - the '
        'arm will stop short.')
    return not bad
