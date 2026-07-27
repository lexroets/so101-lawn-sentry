"""Locked demo box colour (calibrate_box.py)."""

# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
POOP_DEMO_BOX_MODE = True
POOP_BOX_HSV_LO = (9, 0, 36)
POOP_BOX_HSV_HI = (47, 204, 255)
