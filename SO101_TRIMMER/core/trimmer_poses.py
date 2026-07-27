# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
# Written by calibrate_trimmer.py 2026-07-22 14:49:13
# Trimmer arm taught poses (4 joints: base/shoulder/elbow/wrist).
REST_POSE = {1: 1973, 2: 2056, 3: 1814, 4: 2219}
SWEEP = [
    {1: 986, 2: 2584, 3: 1806, 4: 1790},
    {1: 1409, 2: 2583, 3: 1806, 4: 1787},
    {1: 1933, 2: 2617, 3: 1807, 4: 1673},
    {1: 2486, 2: 2625, 3: 1813, 4: 1674},
    {1: 2917, 2: 2638, 3: 1808, 4: 1672},
    {1: 923, 2: 1610, 3: 2898, 4: 1670},
    {1: 1584, 2: 1614, 3: 2857, 4: 1674},
    {1: 2068, 2: 1613, 3: 2856, 4: 1673},
    {1: 2606, 2: 1613, 3: 2857, 4: 1673},
    {1: 2869, 2: 1614, 3: 2856, 4: 1674},
    {1: 980, 2: 1905, 3: 1502, 4: 2613},
    {1: 1970, 2: 1912, 3: 1501, 4: 2613},
    {1: 2969, 2: 1913, 3: 1502, 4: 2613},
    {1: 957, 2: 1711, 3: 1580, 4: 1874},
    {1: 1934, 2: 1787, 3: 1374, 4: 1874},
    {1: 3078, 2: 1786, 3: 1374, 4: 1874},
]
