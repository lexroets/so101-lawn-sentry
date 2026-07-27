# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
# Seeded from the reach_object.py tuning session (2026-07-22).
# Your live tuning settled here before the run went clean. Verify it lands on
# the battery tops, and press  p  in reach_object.py to re-save if you nudge it.
REACH_TRIM = {2: -60, 3: 0, 4: -10}
