"""One way to get frames, whatever the camera is.

The lawn scanner grew up around the OAK-D, but the calibration idea — tell the
system where the ground is, once — applies to any fixed camera. A homeowner's
security camera pointed at the back garden is exactly the same problem, minus
the depth sensor.

So this wraps every source behind `.read()`:

    FrameSource()                       the OAK-D (default)
    FrameSource(0)                      a USB webcam
    FrameSource('rtsp://user:pw@10.0.0.7/stream1')    an IP / security camera
    FrameSource('http://10.0.0.7/snapshot.jpg')       a snapshot URL
    FrameSource('garden.jpg')           a still image (repeats forever)
    FrameSource('clip.mp4')             a video file

Nothing here needs camera intrinsics or depth, which is what lets the same
calibration run on hardware we know nothing about.
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
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


class FrameSource:
    """Uniform frame access. `read()` returns a BGR frame or None."""

    def __init__(self, source=None, width=None, height=None):
        self.source = source
        self.kind = None
        self._cap = None
        self._still = None
        self._oak = None
        self._fail = 0

        if source is None:
            self._open_oak()
        elif isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXT \
                and Path(source).is_file():
            self._still = cv2.imread(source)
            if self._still is None:
                raise RuntimeError(f'could not read image: {source}')
            self.kind = 'image'
        else:
            # int-like strings mean a device index ("0"), everything else is a
            # URL or file path that OpenCV can open
            src = source
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            self._cap = cv2.VideoCapture(src)
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # a network stream can take a moment to produce its first frame
            t0 = time.time()
            while time.time() - t0 < 8.0:
                if self._cap.isOpened() and self._cap.read()[0]:
                    break
                time.sleep(0.2)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f'could not open video source: {source!r}\n'
                    '  webcam?  try 0, or 1 if you have several\n'
                    '  IP cam?  check the RTSP URL and credentials in VLC first')
            self.kind = 'stream'

    # -- OAK-D ---------------------------------------------------------------
    def _open_oak(self):
        try:
            from trim_detector import build_pipeline
            from camera_session import CameraSession
        except ImportError as e:
            raise RuntimeError(
                'no source given and the OAK-D modules are unavailable '
                f'({e}). Pass --source 0 for a webcam, or an RTSP URL.')
        self._oak = CameraSession(build_pipeline)
        self.kind = 'oak-d'

    # -- reading -------------------------------------------------------------
    def read(self):
        if self._still is not None:
            return self._still.copy()
        if self._oak is not None:
            g = self._oak.get()
            if g is not None:
                return g['rgb'].getCvFrame()
            if self._oak.crashed():
                self._oak.restart()
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fail += 1
            # video file that ran out -> loop it, which is handy for testing
            if self.kind == 'stream' and self._fail < 3:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok:
                    self._fail = 0
                    return frame
            return None
        self._fail = 0
        return frame

    def read_blocking(self, timeout=10.0):
        """Wait for a frame rather than returning None on a slow first read."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.read()
            if f is not None:
                return f
            time.sleep(0.03)
        return None

    def describe(self):
        if self.kind == 'oak-d':
            return 'OAK-D Lite'
        if self.kind == 'image':
            return f'still image ({self.source})'
        return f'stream ({self.source})'

    def stop(self):
        if self._cap is not None:
            self._cap.release()
        if self._oak is not None:
            try:
                self._oak.stop()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()
        return False
