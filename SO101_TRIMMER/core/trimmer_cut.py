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
# Written by calibrate_reach.py 2026-07-22 15:30:21
# Reach table taught against real targets (object depth +
# trimmer-head-at-top pose). Replaces any earlier table.
REST_POSE = {1: 1995, 2: 1633, 3: 2320, 4: 2130}
CUT_POINTS = [
    (0.2600, {1: 2057, 2: 2288, 3: 2119, 4: 1691}),
    (0.2860, {1: 2105, 2: 2404, 3: 1970, 4: 1671}),
    (0.3230, {1: 1845, 2: 2759, 3: 1332, 4: 1983}),
    (0.3500, {1: 1992, 2: 2728, 3: 1414, 4: 1922}),
]
