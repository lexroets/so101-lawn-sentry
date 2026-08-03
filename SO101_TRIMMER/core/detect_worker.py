"""Run the detector on another thread, so slow inference cannot stall the video.

The scan loop does several jobs in sequence: grab a frame, track the ground,
detect waste, draw, publish. Detection is by far the slowest - with
onnxruntime unavailable it falls back to cv2.dnn, and 3x3 tiling means ten
forward passes, measured at around half a second. Every time that ran inline,
everything behind it waited, including the frame being written for the
dashboard. The result was a live feed that hitched every few frames and a
control panel that felt like it was arguing with the arm.

Running it every Nth frame only spread the stall out; it did not remove it.

Here the detector gets its own thread and its own copy of the newest frame.
The scan loop hands over a frame and moves on immediately, so video is
published at camera rate no matter how long inference takes. Results are
picked up whenever they are ready.

The queue is deliberately ONE frame deep and overwritten. A detector that has
fallen behind should work on the newest picture, not grind through a backlog
of stale ones - old detections are worse than late ones, because the camera
has panned since.
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
import threading
import time

import cv2
import numpy as np


def carry_forward(dets, H_then, H_now):
    """Move boxes from the frame they were found in to the frame on screen now.

    Detection takes a few hundred milliseconds, so its results always describe
    a frame that has already gone. While the camera is still that is invisible.
    While it is SWEEPING it is not: at the survey rate the picture slides on by
    several pixels in that time, so every box trails its object, and a viewer
    reads that as the tracking being loose.

    It is correctable, though, because the ground tracker already knows how the
    view moved. H maps image pixels to lawn metres, so a point in the older
    frame reaches the current one via inv(H_now) . H_then - going out to the
    lawn, which does not move, and coming back in through the current pose.
    Boxes then sit on their objects while panning instead of behind them.
    """
    if not dets or H_then is None or H_now is None:
        return dets
    try:
        M = np.linalg.inv(np.asarray(H_now, np.float64)) @ np.asarray(H_then, np.float64)
    except np.linalg.LinAlgError:
        return dets
    if not np.isfinite(M).all():
        return dets
    M = (M / M[2, 2]).astype(np.float32)
    if abs(M[0, 2]) < 0.5 and abs(M[1, 2]) < 0.5 and np.allclose(M[:2, :2], np.eye(2), atol=1e-3):
        return dets                      # camera was still; nothing to do
    out = []
    for d in dets:
        try:
            x, y, w, h = d['bbox']
            pts = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.float32)
            q = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            if not np.isfinite(q).all():
                out.append(d)
                continue
            nx, ny = q[:, 0].min(), q[:, 1].min()
            nw, nh = q[:, 0].max() - nx, q[:, 1].max() - ny
            e = dict(d)
            e['bbox'] = (int(round(nx)), int(round(ny)),
                         int(round(nw)), int(round(nh)))
            e['bbox_raw'] = d['bbox']
            out.append(e)
        except (KeyError, ValueError, TypeError):
            out.append(d)
    return out


class DetectWorker:
    """Newest-frame-wins detection on a background thread.

        worker = DetectWorker(detector.detect)
        worker.submit(frame, depth, roi_poly=quad)   # returns at once
        dets = worker.results                        # last completed, or []

    `results` never blocks and never returns a half-finished list.
    """

    def __init__(self, detect_fn, name='detect'):
        self._fn = detect_fn
        self._lock = threading.Lock()
        self._pending = None          # (args, kwargs) - only ever the newest
        self._results = []
        self._stats = {}
        # Whatever the caller wants to remember about the frame these results
        # came from - in practice the camera pose at that instant, so the
        # boxes can be brought forward to where the objects are NOW.
        self._tag = None
        self._pending_tag = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._busy = False
        self.frames_done = 0
        self.frames_dropped = 0
        self.last_ms = 0.0
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # -- the scan loop's side -------------------------------------------------
    def submit(self, *args, **kwargs):
        """Offer a frame. Returns immediately, whether or not it gets used.

        `tag=` is carried alongside and handed back with the results, so the
        caller can tell which frame they describe.
        """
        tag = kwargs.pop('tag', None)
        with self._lock:
            if self._pending is not None:
                self.frames_dropped += 1      # superseded before it was started
            self._pending = (args, kwargs)
            self._pending_tag = tag
        self._wake.set()

    @property
    def results(self):
        with self._lock:
            return self._results

    def latest(self):
        """(results, tag) together, so the two can never be mismatched."""
        with self._lock:
            return self._results, self._tag

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    @property
    def busy(self):
        return self._busy

    def health(self):
        """A line worth printing: is the detector keeping up?"""
        return (f'{self.frames_done} detections, {self.frames_dropped} frames '
                f'skipped, last {self.last_ms:.0f} ms')

    # -- the worker's side ----------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    job, self._pending = self._pending, None
                    tag, self._pending_tag = self._pending_tag, None
                if job is None:
                    break
                args, kwargs = job
                self._busy = True
                t0 = time.monotonic()
                try:
                    stats = {}
                    kwargs = dict(kwargs)
                    kwargs['stats'] = stats
                    out = self._fn(*args, **kwargs)
                except Exception as e:
                    out, stats = [], {'error': str(e)}
                self.last_ms = (time.monotonic() - t0) * 1000.0
                self._busy = False
                self.frames_done += 1
                with self._lock:
                    self._results = out or []
                    self._stats = stats
                    self._tag = tag

    def stop(self, timeout=1.0):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)
