"""Locked demo box colour (calibrate_box.py)."""

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
POOP_DEMO_BOX_MODE = True
POOP_BOX_HSV_LO = (9, 0, 36)
POOP_BOX_HSV_HI = (47, 204, 255)
