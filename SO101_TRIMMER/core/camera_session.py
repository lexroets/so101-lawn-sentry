"""Crash-resilient OAK-D camera session.

The OAK-D Lite drops off the USB bus under load (power brownout). This wraps
the pipeline so a crash is detected and the device is rebuilt automatically,
instead of freezing the program. For unattended running the caller should
still PARK THE ARM before a restart (the camera is blind during it).

    cam = CameraSession(build_pipeline)
    while True:
        g = cam.get()
        if g is None:
            if cam.crashed():
                arm.park(); cam.restart()   # or break if restart fails
            continue
        ... use g ...
"""

# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import time


class CameraSession:
    def __init__(self, build_fn, name_key='rgb'):
        self._build = build_fn
        self.name_key = name_key
        self.restarts = 0
        self._open()

    def _open(self):
        self.pipeline, self.queue = self._build()
        self.pipeline.start()
        self.last_frame = time.monotonic()

    def running(self):
        try:
            return self.pipeline.isRunning()
        except Exception:
            return False

    def get(self):
        """Return the latest frame group, or None. Updates the liveness clock."""
        try:
            g = self.queue.tryGet()
        except Exception:
            return None
        if g is not None:
            self.last_frame = time.monotonic()
        return g

    def crashed(self, timeout=3.0):
        """True if the device stopped or has delivered no frame for `timeout`."""
        return (not self.running()) or (time.monotonic() - self.last_frame > timeout)

    def restart(self, settle=1.5):
        """Tear down and rebuild the pipeline. Returns True on success."""
        self.restarts += 1
        try:
            self.pipeline.stop()
        except Exception:
            pass
        time.sleep(settle)
        try:
            self._open()
            print(f"[camera] restarted (attempt {self.restarts})")
            return True
        except Exception as exc:
            print(f"[camera] restart FAILED: {exc}")
            return False

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
