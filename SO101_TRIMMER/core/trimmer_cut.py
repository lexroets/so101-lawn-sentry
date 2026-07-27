# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
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
