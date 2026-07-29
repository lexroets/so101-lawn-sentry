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
# Seeded from the reach_object.py tuning session (2026-07-22).
# Your live tuning settled here before the run went clean. Verify it lands on
# the battery tops, and press  p  in reach_object.py to re-save if you nudge it.
REACH_TRIM = {2: -60, 3: 0, 4: -10}
