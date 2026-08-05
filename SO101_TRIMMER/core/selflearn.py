"""Let the arm correct itself, every time it runs, without being asked.

WHAT THIS IS
------------
The arm does not go where it is told. It droops under gravity, and it sticks
against friction, and the amount depends on which direction the joint just
came from - so it cannot be predicted from the pose. Measured on this rig the
head landed up to 24 mm below where it was sent.

None of that has to be predicted, because it repeats. Two measured runs of the
same path agreed to 0.2 mm. Something that repeats to 0.2 mm does not need a
model; it needs remembering. So the path is run, the encoders are read, the
difference is folded back into the command, and the next run is closer.

WHY IT REPEATS AND ACCUMULATES
------------------------------
One measurement is not enough. The correction changes the command, and the
error depends on the command - subtract what was measured and the arm makes a
different, smaller error. So each pass ADDS what is left over to what is
already in force, and the leftover shrinks each time. Simulated against a
realistic response it goes 113 -> 46 -> 18 -> 7 -> 3 raw over five passes.

The gain is below one deliberately. Taking the whole leftover error each time
assumes the arm responds exactly as commanded, and overshoots when it does
not.

WHAT IT IS NOT
--------------
It is not a model and does not generalise. It describes THIS path at THIS
speed with THIS head. Change any of them and the old corrections are wrong -
so the table records what it was measured against, and a path that no longer
matches is ignored rather than silently mis-corrected.
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
import importlib
import sys
import time

import trimmer_config as tc
import pathplay
from pathplay import goto

GAIN = 0.7                 # how much of each pass's leftover error to take on
MAX_CORRECTION = 400       # never push a joint further than this off its target
SETTLE_S = 0.9             # let the servo stop creeping before reading it
MATCH_TOL = 120            # a stored correction only applies to the same pose


def _previous(old, at):
    best, dist = None, None
    for row in old:
        d = sum(abs(int(at[j]) - int(row['at'][j]))
                for j in (tc.SHOULDER_ID, tc.ELBOW_ID)
                if j in row['at'] and j in at)
        if dist is None or d < dist:
            best, dist = row, d
    if best is None or dist is None or dist > MATCH_TOL:
        return {}
    return best['err']


def load():
    try:
        from sag_table import SAG
        return list(SAG)
    except Exception:
        return []


def save(merged, log=print):
    from paths import CORE
    p = CORE / 'sag_table.py'
    with p.open('w') as f:
        f.write('# Written by core/selflearn.py ' +
                time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# Measured overshoot, subtracted from the command so the arm\n')
        f.write('# lands where it is meant to. Learned by running the path and\n')
        f.write('# reading the encoders - not modelled.\n')
        f.write('# Describes THIS path at THIS speed with THIS head.\n\n')
        f.write('SAG = [\n')
        for m in merged:
            f.write('    {'
                    + "'at': {" + ', '.join(f'{k}: {v}'
                                            for k, v in sorted(m['at'].items())) + '}, '
                    + "'err': {" + ', '.join(f'{k}: {v}'
                                             for k, v in sorted(m['err'].items())) + '}'
                    + '},\n')
        f.write(']\n')
    # Python decides a .pyc is current from the source's (mtime, size). This
    # file is rewritten constantly with numbers of the same length, so those
    # can easily match and a fresh process would load the previous lesson.
    for c in (CORE / '__pycache__').glob('sag_table.*.pyc'):
        try:
            c.unlink()
        except OSError:
            pass
    sys.modules.pop('sag_table', None)
    importlib.invalidate_caches()


def fold_in(errs, log=print):
    """Add this pass's leftover error to the correction already in force."""
    old = load()
    merged, moved = [], 0
    for at, e in errs:
        was = _previous(old, at)
        now = {}
        for j, v in e.items():
            acc = int(round(was.get(j, 0) + GAIN * v))
            now[j] = max(-MAX_CORRECTION, min(MAX_CORRECTION, acc))
            moved = max(moved, abs(now[j] - was.get(j, 0)))
        merged.append({'at': at, 'err': now})
    save(merged, log)
    return moved


def learn_once(arm, wps, name, log=print, settle=SETTLE_S, quiet=False):
    """Run the path a pose at a time, read where it landed, fold in the error.

    Returns the worst leftover error in raw ticks, or None if it could not.
    """
    from trimmer_arm import TrimmerError
    targets, cmd = pathplay.prepare(wps, name, log=(lambda m: None) if quiet else log)
    errs, worst = [], 0
    for i, w in enumerate(cmd):
        if goto(arm, w, log=lambda m: None) is None:
            return None
        time.sleep(settle)
        try:
            act = arm.read_pose()
        except TrimmerError:
            continue
        t = targets[i]
        e = {j: int(act[j]) - int(t[j])
             for j in (tc.SHOULDER_ID, tc.ELBOW_ID, tc.WRIST_ID)
             if j in act and j in t}
        if not e:
            continue
        worst = max(worst, max(abs(v) for v in e.values()))
        errs.append((dict(t), e))
    if len(errs) < 4:
        return None
    moved = fold_in(errs, log)
    log(f'learned from {len(errs)} poses: worst leftover {worst} raw, '
        f'correction moved up to {moved} raw')
    return worst


def learn(arm, wps, name, cycles=4, log=print, stop_check=None,
          verbose=False):
    """Keep folding in until it stops getting better.

    Stops early on its own. Running more passes than needed is not harmful,
    but it is slow, and a correction that has stopped shrinking is finished.
    """
    history = []
    for c in range(max(1, cycles)):
        if stop_check is not None and stop_check():
            log('learning stopped')
            break
        # the transform chain announces itself on every call; once is
        # informative, once per learning pass is noise
        worst = learn_once(arm, wps, name, log=log,
                           quiet=not (verbose and c == 0))
        if worst is None:
            break
        history.append(worst)
        log(f'  pass {c + 1}: worst error {worst} raw')
        if worst <= 6:
            log('  as close as the encoder can tell - done')
            break
        if len(history) > 1 and history[-1] > history[-2] * 0.9:
            log('  no longer improving - done')
            break
    return history
