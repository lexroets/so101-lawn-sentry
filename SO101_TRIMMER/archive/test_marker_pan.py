"""Does the marker calibration still work when the camera has to PAN?

A homography describes one camera pose. If the lawn does not fit in the field
of view, the four corners get captured from four DIFFERENT views, and naively
combining those image points produces nonsense.

Ground tracking is supposed to fix that: it estimates the transform from the
current view back to the first, so a corner seen after panning can be written
in the first view's coordinates. This test checks whether that actually holds
up, by panning a virtual camera between corners and comparing the result
against geometry we know exactly.

    python tools/test_marker_pan.py
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

import cv2
import numpy as np

from marker import MarkerFinder, marker_image
from ground_track import GroundTracker
from calibrate_marker import solve, measure_marker, to_world
from test_marker_calibration import VirtualCamera


def textured_ground(seed=0, size=4000, span=12.0):
    """A big top-down texture of grass, so tracking has something to hold on to.

    Real turf is full of detail; flat synthetic green would make the tracker
    look worse than it is and the test meaningless.
    """
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (size // 8, size // 8, 3))
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
    img = np.clip((58, 104, 52) + g * 26, 0, 255).astype(np.uint8)
    fine = rng.normal(0, 12, (size, size, 1))
    img = np.clip(img + fine, 0, 255).astype(np.uint8)
    for _ in range(700):                      # leaves, clippings, wear patches
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(4, 22))
        col = tuple(int(c) for c in rng.integers((30, 60, 25), (90, 150, 80)))
        cv2.circle(img, (int(x), int(y)), r, col, -1)
    return img, size / span                   # image, pixels per metre


GROUND, PPM = textured_ground()


def render(cam, marker_xy=None, marker_side=0.15, origin=(-2.0, -2.0)):
    """Render the ground (and optionally a marker) as this camera sees it."""
    h, w = cam.h, cam.w
    # map ground-texture pixels -> world metres -> image
    corners_tex = np.array([[0, 0], [GROUND.shape[1], 0],
                            [GROUND.shape[1], GROUND.shape[0]],
                            [0, GROUND.shape[0]]], np.float32)
    world = corners_tex / PPM + np.array(origin, np.float32)
    dst = cam.ground(world).astype(np.float32)
    H = cv2.getPerspectiveTransform(corners_tex, dst)
    img = cv2.warpPerspective(GROUND, H, (w, h), borderValue=(40, 55, 38))

    if marker_xy is not None:
        m = cv2.cvtColor(marker_image(600), cv2.COLOR_GRAY2BGR)
        m = cv2.copyMakeBorder(m, 90, 90, 90, 90, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
        half = marker_side * (600 + 180) / 600 / 2
        cx, cy = marker_xy
        d = cam.ground([[cx - half, cy + half], [cx + half, cy + half],
                        [cx + half, cy - half], [cx - half, cy - half]])
        src = np.array([[0, 0], [m.shape[1], 0], [m.shape[1], m.shape[0]],
                        [0, m.shape[0]]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, d.astype(np.float32))
        warped = cv2.warpPerspective(m, Hm, (w, h))
        mask = cv2.warpPerspective(np.full(m.shape[:2], 255, np.uint8), Hm, (w, h))
        img[mask > 128] = warped[mask > 128]
    return img


def cam_at(yaw, W, L, height=1.2, dist=2.2, px=(1280, 720), hfov=70):
    """Camera on a fixed mount, panned to `yaw`, aimed at the lawn centre."""
    c = VirtualCamera(px[0], px[1], hfov, height, dist, 0, (W / 2, L / 2))
    # rotate the view direction about the camera's own position
    a = math.radians(yaw)
    Rz = np.array([[math.cos(a), -math.sin(a), 0],
                   [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    c.R = c.R @ Rz.T
    c.t = -c.R @ c.C
    return c


def yaw_to_see(corner, W, L, limit=60.0):
    """Pan angle that brings `corner` closest to the middle of the frame.

    A user aims the camera at the corner they are about to place the marker on,
    rather than turning by some fixed amount, so the test should do the same.
    """
    best, best_d = 0.0, 1e18
    for y in np.arange(-limit, limit + 0.5, 1.0):
        cam = cam_at(float(y), W, L)
        p = cam.ground([corner])[0]
        if not np.isfinite(p).all():
            continue
        d = float(np.hypot(p[0] - cam.w / 2, p[1] - cam.h * 0.62))
        # must actually be on screen with room for the marker
        if not (60 < p[0] < cam.w - 60 and 60 < p[1] < cam.h - 60):
            continue
        if d < best_d:
            best, best_d = float(y), d
    return best if best_d < 1e17 else None


def run(W, L, side, yaws, label):
    finder = MarkerFinder()
    trk = GroundTracker(np.eye(3))
    corners = [(0, 0), (W, 0), (W, L), (0, L)]
    caps = []
    lost = 0

    for i, (yaw, corner) in enumerate(zip(yaws, corners)):
        # pan smoothly from the previous yaw to this one, tracking all the way
        prev = yaws[i - 1] if i else yaws[0]
        # Step about 1.5 degrees per frame. A real camera at 30 fps panned by
        # hand moves far less than that between frames; stepping in big jumps
        # would be testing a scenario that never happens.
        n = max(2, int(abs(yaw - prev) / 1.5) + 1)
        for t in np.linspace(0, 1, n)[1:] if i else [1.0]:
            y = prev + (yaw - prev) * float(t)
            trk.update(render(cam_at(y, W, L)))
            if trk.status()['lost']:
                lost += 1
        cam = cam_at(yaw, W, L)
        img = render(cam, corner, side)
        trk.update(img)
        c = finder.find(img)
        if c is None:
            return None, f'marker not detected at corner {i+1}'
        ref = cv2.perspectiveTransform(
            c.reshape(1, -1, 2).astype(np.float32), trk.G).reshape(4, 2)
        caps.append({'centre': ref.mean(axis=0), 'corners': ref})

    H, _ = solve(caps, W, L, side)

    # accuracy is judged in the FIRST view, which is what the calibration is
    # tied to and the pose the camera must return to for a real scan
    cam0 = cam_at(yaws[0], W, L)
    gx, gy = np.meshgrid(np.linspace(0.1, W - 0.1, 6), np.linspace(0.1, L - 0.1, 6))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    err = np.linalg.norm(to_world(H, cam0.ground(pts)) - pts, axis=1) * 100
    sq = max(measure_marker(H, c['corners'], side)[2] for c in caps) * 100
    return {'mean': err.mean(), 'worst': err.max(), 'sq': sq, 'lost': lost}, None


def main():
    print('=' * 72)
    print('MARKER CALIBRATION WITH A PANNING CAMERA')
    print('=' * 72)
    print('\nThe lawn does not fit in one view, so the camera pans between')
    print('corners and ground tracking stitches the views together.\n')
    W, L, side = 3.5, 2.1, 0.15
    corners = [(0, 0), (W, 0), (W, L), (0, L)]
    aimed = [yaw_to_see(c, W, L) for c in corners]
    if any(a is None for a in aimed):
        print('  could not frame every corner even with panning'); return 1
    print(f'  panning to {[round(a) for a in aimed]} degrees to reach each corner')
    print(f'  (total sweep {max(aimed)-min(aimed):.0f} degrees)\n')
    cases = [
        ('no pan (control)', [0, 0, 0, 0]),
        ('half the pan', [a * 0.5 for a in aimed]),
        ('aimed at corners', aimed),
        ('over-panned x1.3', [a * 1.3 for a in aimed]),
    ]
    print(f'  {"pan pattern":20s} {"mean":>9s} {"worst":>9s} {"squareness":>12s} {"lost":>6s}')
    print('  ' + '-' * 60)
    results = []
    for label, yaws in cases:
        r, err = run(W, L, side, yaws, label)
        if r is None:
            print(f'  {label:20s} {err}')
            continue
        results.append((label, r))
        print(f'  {label:20s} {r["mean"]:7.1f}cm {r["worst"]:7.1f}cm '
              f'{r["sq"]:10.1f}% {r["lost"]:6d}')
    print()
    if not results:
        print('  NO CASES RAN'); return 1
    worst = max(r['worst'] for _, r in results)
    print(f'  worst position error across all pan patterns: {worst:.1f} cm')
    print()
    ok = worst < 20.0
    print('  VERDICT:', 'PASS - panning is safe with --pan' if ok
          else 'FAIL - panning degrades the calibration too much')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
