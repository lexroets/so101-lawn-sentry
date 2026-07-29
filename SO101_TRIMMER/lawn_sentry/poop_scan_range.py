"""Taught base sweep range. Regenerate with calibrate_sweep.py."""

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
SCAN_BASE_MIN_RAW = 981
SCAN_BASE_MAX_RAW = 1218
SCAN_STATIONS = 4
