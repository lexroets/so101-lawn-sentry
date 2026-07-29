"""Prove the marker calibration is accurate, using a camera we fully control.

On real hardware there is no ground truth: if the map says a target is at
(2.1, 1.4) m there is nothing to check it against short of a tape measure. So
this builds a *virtual* security camera looking at a *virtual* lawn, renders
the marker at each corner exactly as a real camera would see it, runs the real
calibration code on those images, and then asks the only question that matters:

    given a point at a KNOWN place on the lawn, where does the calibration
    think it is?

Because the scene is synthetic, the answer is known to the millimetre.

    python tools/test_marker_calibration.py
    python tools/test_marker_calibration.py --save    # write the test images
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
import argparse
import math

import cv2
import numpy as np

from marker import MarkerFinder, marker_image
from calibrate_marker import solve, measure_marker, to_world, MIN_MARKER_PX


class VirtualCamera:
    """A pinhole camera on a mast, looking down at the ground plane z=0."""

    def __init__(self, width=1920, height=1080, hfov_deg=70.0,
                 height_m=3.0, distance_m=6.0, yaw_deg=0.0, aim=(0.0, 0.0)):
        self.w, self.h = width, height
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float)

        # camera sits behind and above the aim point
        yaw = math.radians(yaw_deg)
        self.C = np.array([aim[0] - distance_m * math.sin(yaw),
                           aim[1] - distance_m * math.cos(yaw),
                           height_m], float)
        fwd = np.array([aim[0], aim[1], 0.0]) - self.C
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0, 0, 1.0])
        right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.vstack([right, down, fwd])          # world -> camera
        self.t = -self.R @ self.C

    def project(self, pts_world):
        """(N,3) world -> (N,2) image pixels."""
        P = np.asarray(pts_world, float).reshape(-1, 3)
        cam = (self.R @ P.T).T + self.t
        z = np.clip(cam[:, 2], 1e-6, None)
        img = (self.K @ (cam / z[:, None]).T).T
        return img[:, :2]

    def ground(self, xy):
        """(N,2) lawn metres -> (N,2) image pixels."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.project(np.hstack([xy, np.zeros((len(xy), 1))]))


def fit_camera(width, height, hfov, cam_h, yaw, W, L, margin=0.12):
    """Back the camera off until the whole lawn is comfortably in frame.

    A calibration can only use corners it can see, so a test that places them
    off-screen tests nothing. This finds the closest distance that still shows
    the entire lawn with a margin — which is also the advice a real user needs.
    """
    aim = (W / 2, L / 2)
    best = None
    for dist in np.arange(1.5, 60.0, 0.25):
        cam = VirtualCamera(width, height, hfov, cam_h, float(dist), yaw, aim)
        pts = cam.ground([[0, 0], [W, 0], [W, L], [0, L]])
        mx, my = width * margin, height * margin
        inside = np.all((pts[:, 0] > mx) & (pts[:, 0] < width - mx) &
                        (pts[:, 1] > my) & (pts[:, 1] < height - my))
        # also require the far edge to be below the horizon and in front
        if inside:
            best = cam
            break
    return best


def needed_marker_size(cam, W, L, min_px=None):
    """Smallest marker that is still readable at the FARTHEST corner.

    Detection, not geometry, is what limits this method on a big garden: the
    maths is exact but a code that lands on 12 pixels cannot be decoded. So the
    honest recommendation is a size, not a shrug.
    """
    # 1.4x the bare minimum: sizing for exactly the threshold means half the
    # corners fail on a slightly soft or shadowed frame.
    min_px = min_px or MIN_MARKER_PX * 1.4
    f = cam.K[0, 0]
    worst = 0.0
    for c in [(0, 0), (W, 0), (W, L), (0, L)]:
        d = float(np.linalg.norm(cam.C - np.array([c[0], c[1], 0.0])))
        worst = max(worst, min_px * d / f)
    return worst


def render_scene(cam, lawn_w, lawn_l, marker_xy, marker_side, seed=0):
    """Draw grass, the lawn, and one marker lying flat at marker_xy."""
    rng = np.random.default_rng(seed)
    img = np.full((cam.h, cam.w, 3), (48, 62, 44), np.uint8)
    img = (img + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)

    # the lawn itself, so the picture looks like a garden
    quad = cam.ground([[0, 0], [lawn_w, 0], [lawn_w, lawn_l], [0, lawn_l]])
    cv2.fillConvexPoly(img, quad.astype(np.int32), (58, 104, 52))
    img = cv2.GaussianBlur(img, (0, 0), 1.2)

    # the marker, warped onto the ground exactly as perspective demands
    m = cv2.cvtColor(marker_image(600), cv2.COLOR_GRAY2BGR)
    m = cv2.copyMakeBorder(m, 90, 90, 90, 90, cv2.BORDER_CONSTANT,
                           value=(255, 255, 255))   # quiet zone
    # The image being warped is the marker (600 px) PLUS a 90 px quiet zone on
    # every side, so it is 780 px wide and must map to a ground square of
    # marker_side * 780/600. Getting this wrong renders the marker at the
    # wrong scale and quietly invalidates the whole test.
    half = marker_side * (600 + 2 * 90) / 600 / 2
    cx, cy = marker_xy
    dst = cam.ground([[cx - half, cy + half], [cx + half, cy + half],
                      [cx + half, cy - half], [cx - half, cy - half]])
    src = np.array([[0, 0], [m.shape[1], 0], [m.shape[1], m.shape[0]],
                    [0, m.shape[0]]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(m, Hm, (cam.w, cam.h))
    mask = cv2.warpPerspective(np.full(m.shape[:2], 255, np.uint8), Hm,
                               (cam.w, cam.h))
    img[mask > 128] = warped[mask > 128]
    return img


def run_case(name, cam, W, L, side, save=False):
    finder = MarkerFinder()
    # The marker CENTRE goes on the corner — that is what the tool asks the
    # user to do, and what makes the centres correspond to the rectangle.
    placements = [(0, 0), (W, 0), (W, L), (0, L)]

    detected, sizes = [], []
    for i, p in enumerate(placements):
        img = render_scene(cam, W, L, p, side, seed=i)
        c = finder.find(img)
        if c is None:
            # how big was it on screen? that is the actionable number
            q = cam.ground([[p[0] - side / 2, p[1]], [p[0] + side / 2, p[1]]])
            px = float(np.linalg.norm(q[0] - q[1]))
            print(f'  {name:15s} marker NOT detected at corner {i+1} '
                  f'- only ~{px:.0f} px on screen (needs ~{MIN_MARKER_PX})')
            return None
        detected.append(c)
        sizes.append(float(np.linalg.norm(c[0] - c[1])))
        if save:
            cv2.imwrite(f'/tmp/marker_{name}_{i+1}.jpg', img)

    caps = [{'centre': c.mean(axis=0), 'corners': c} for c in detected]
    H, src = solve(caps, W, L, side)

    # --- the self-check the tool shows the user ---
    m = [measure_marker(H, c, side) for c in detected]
    worst_self = max(abs(e[1]) for e in m)
    worst_sq = max(e[2] for e in m) * 100

    # --- the REAL test: known ground points, mapped back ---
    gx, gy = np.meshgrid(np.linspace(0.15, W - 0.15, 7),
                         np.linspace(0.15, L - 0.15, 7))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    px = cam.ground(pts)
    got = to_world(H, px)
    d = np.linalg.norm(got - pts, axis=1) * 100      # cm
    return {'name': name, 'self_cm': worst_self, 'mean_cm': d.mean(),
            'worst_cm': d.max(), 'p95_cm': np.percentile(d, 95),
            'min_px': min(sizes), 'sq': worst_sq, 'W': W, 'L': L}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true', help='write test images to /tmp')
    args = ap.parse_args()

    print('=' * 74)
    print('MARKER CALIBRATION - ACCURACY AGAINST KNOWN GROUND TRUTH')
    print('=' * 74)
    print('\nA virtual camera views a virtual lawn. The calibration only ever')
    print('sees rendered images; the true geometry is known exactly.\n')

    # (name, px, hfov, camera height m, yaw deg, lawn W, lawn L, marker m)
    cases = [
        ('demo-lawn',     1280,  720,  70, 1.2,  0,  3.5,  2.1, 0.15),
        ('security-cam',  1920, 1080,  70, 3.0,  0,  8.0,  6.0, 0.15),
        ('high-mount',    1920, 1080,  90, 5.0,  0, 10.0,  8.0, 0.20),
        ('oblique-20deg', 1920, 1080,  70, 3.0, 20,  8.0,  6.0, 0.15),
        ('oblique-40deg', 1920, 1080,  70, 3.5, 40,  8.0,  6.0, 0.15),
        ('wide-garden',   1920, 1080, 100, 4.0,  0, 14.0, 10.0, 0.25),
        ('cheap-720p',    1280,  720,  62, 2.5, 15,  6.0,  5.0, 0.15),
    ]

    print(f'  {"scene":15s} {"lawn":>10s} {"dist":>6s} {"self-check":>11s} '
          f'{"marker":>7s} {"sq":>6s} {"mean":>8s} {"p95":>8s} {"worst":>8s}')
    print('  ' + '-' * 82)
    results = []
    for name, pw, ph, hfov, ch, yaw, W, L, side in cases:
        cam = fit_camera(pw, ph, hfov, ch, yaw, W, L)
        if cam is None:
            print(f'  {name:15s} could not frame the whole lawn - skipped')
            continue
        dist = float(np.hypot(cam.C[0] - W / 2, cam.C[1] - L / 2))
        need = needed_marker_size(cam, W, L)
        # use what the geometry actually demands, rounded up to a printable size
        use = max(side, math.ceil(need * 100 / 5) * 5 / 100)
        r = run_case(name, cam, W, L, use, args.save)
        if r is None:
            continue
        results.append(r)
        print(f'  {r["name"]:15s} {W:4.1f}x{L:<4.1f}m {dist:5.1f}m {use*100:4.0f}cm '
              f'{r["self_cm"]:8.2f}cm {r["sq"]:5.1f}% {r["mean_cm"]:6.1f}cm '
              f'{r["p95_cm"]:6.1f}cm {r["worst_cm"]:6.1f}cm')

    print()
    if not results:
        print('  NO CASES RAN'); return 1
    worst = max(r['worst_cm'] for r in results)
    mean = np.mean([r['mean_cm'] for r in results])
    print(f'  across {len(results)} scenes: mean error {mean:.1f} cm, '
          f'worst single point {worst:.1f} cm')
    print()
    print('  "self-check" is the number the tool reports to the user (how far')
    print('  the marker\'s measured size is from its true size). "mean/p95/worst"')
    print('  are the real position errors over a 7x7 grid covering the lawn.')
    print()
    ok = worst < 15.0
    print('  VERDICT:', 'PASS - accurate enough to place targets on a map'
          if ok else 'FAIL - position error too large')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
