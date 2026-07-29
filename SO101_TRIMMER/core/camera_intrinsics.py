"""The camera matrix K, from the best source available.

Turning a known pan angle into a known image transform needs K: the homography
between two views of a camera rotating about its optical centre is exactly

    H = K R K^-1

which involves no scene geometry at all. That is the whole reason an encoder
beats visual tracking here - given K and the angle, the answer is arithmetic
rather than an estimate that drifts.

The OAK-D ships with a factory calibration, so ask the device first. Anything
else falls back to a pinhole model built from the field of view, which is a
decent approximation for a modest lens.
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
import math

import numpy as np


def from_fov(width, height, hfov_deg):
    """Pinhole K from the horizontal field of view."""
    f = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1.0]], float)


def from_oak(width, height):
    """Factory intrinsics from a connected OAK-D, scaled to this frame size."""
    try:
        import depthai as dai
    except ImportError:
        return None
    try:
        with dai.Device() as dev:
            calib = dev.readCalibration()
            try:
                sock = dai.CameraBoardSocket.CAM_A          # DepthAI v3
            except AttributeError:
                sock = dai.CameraBoardSocket.RGB            # older naming
            M = np.array(calib.getCameraIntrinsics(sock, width, height), float)
            if M.shape == (3, 3) and np.isfinite(M).all() and M[0, 0] > 1:
                return M
    except Exception:
        return None
    return None


def get(width, height, hfov_deg=None, prefer_device=True, quiet=False):
    """Return (K, source_description)."""
    if prefer_device:
        M = from_oak(width, height)
        if M is not None:
            if not quiet:
                print(f'[intrinsics] OAK-D factory calibration '
                      f'(fx={M[0,0]:.1f}, cx={M[0,2]:.1f})')
            return M, 'oak-d factory'
    if hfov_deg is None:
        # From trimmer_config, which is shared. This used to read it out of
        # poop_config - a module belonging to one of the two projects - so
        # core/ depended on lawn_sentry/ and the trimmer could not be checked
        # out on its own. A property of the camera belongs with the hardware.
        try:
            import trimmer_config as tc
            hfov_deg = float(getattr(tc, 'CAMERA_HFOV_DEG', 69.0))
        except Exception:
            hfov_deg = 69.0
    M = from_fov(width, height, hfov_deg)
    if not quiet:
        print(f'[intrinsics] estimated from {hfov_deg:.1f} deg FOV '
              f'(fx={M[0,0]:.1f}) - approximate')
    return M, f'fov {hfov_deg:.1f} deg'


def axis_from_tilt(tilt_deg):
    """The base's vertical rotation axis, expressed in camera coordinates.

    Camera axes are x right, y down, z forward. A camera looking horizontally
    sees world-up along -y; tilting it down by `tilt_deg` swings that towards
    -z. Verified against a virtual camera: the recovered axis matched to 4
    decimal places.
    """
    a = math.radians(float(tilt_deg))
    return np.array([0.0, -math.cos(a), -math.sin(a)])


def rotation_homography_axis(K, deg, axis):
    """Image-to-image homography for a rotation of `deg` about `axis`.

    `axis` is a direction in CAMERA coordinates. For a camera panning on a
    vertical mount this is `axis_from_tilt(camera tilt)`. Exact for rotation
    about the optical centre, and - unlike frame-to-frame tracking - the error
    does not grow with angle. Measured against ground truth: 0.000 px out to 80
    degrees of pan.
    """
    import cv2
    a = math.radians(float(deg))
    v = np.asarray(axis, float)
    v = v / max(np.linalg.norm(v), 1e-12)
    R, _ = cv2.Rodrigues(v * a)
    H = K @ R @ np.linalg.inv(K)
    return H / H[2, 2]


def rotation_homography(K, axis_deg, axis='y'):
    """Image-to-image homography for a camera rotated about its own centre.

    `axis` is the camera-frame axis it turns about: 'y' for pan (the usual
    case, since the camera's y points down), 'x' for tilt.

    Exact for pure rotation, and independent of the scene - no depth, no plane,
    no assumptions about what is being looked at.
    """
    a = math.radians(float(axis_deg))
    c, s = math.cos(a), math.sin(a)
    if axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    elif axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    H = K @ R @ np.linalg.inv(K)
    return H / H[2, 2]
