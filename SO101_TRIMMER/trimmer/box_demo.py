"""Outline a box on the table: the top face, and the vertical edges.

    python trimmer/box_demo.py

The point of this for a demo is that a strimmer has two jobs, not one. Mowing
the TOP of a bed is the easy half; edging the SIDE of it is a different motion
entirely, and a machine that only does the first is half a product. A box is
the honest way to show both at once - it has a flat top and four vertical
faces, and unlike grass it has exact dimensions, so the numbers on screen can
be checked against a tape measure in front of whoever is watching.

What gets drawn:

    GREEN   the top face, and the path a flat cut would follow across it
    ORANGE  the four vertical edges, and the path an edge cut would follow
            down them
    the measured size in centimetres, live

The arm is not commanded. This is the perception half; run it alongside
reach_object.py to talk about the motion.

    m   cycle: outline only / + cut paths / the depth view it works from
    f   freeze, so you can point at it while talking
    q   quit
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
import cv2
import numpy as np

import trimmer_config as tc
from trim_detector import build_pipeline
from camera_session import CameraSession
from camera_intrinsics import get as get_K
from box_finder import fit_table, find_box, project
from keys import Keys

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'box outline - top cut and edge cut'
GREEN, ORANGE, GREY = (90, 255, 90), (60, 170, 255), (120, 120, 120)
VIEWS = ('outline', 'outline + cut paths', 'depth')
# How many passes a top cut would take across the face, and how many heights
# an edge cut would step down each side. Presentation only - nothing moves.
TOP_PASSES = 4
EDGE_STEPS = 3


def lerp(a, b, t):
    return a + (b - a) * t


def draw_top_path(img, box, K, n):
    """Parallel passes across the top face, like mowing it."""
    t = box['top_m']
    for i in range(TOP_PASSES):
        f = (i + 0.5) / TOP_PASSES
        a = lerp(t[0], t[3], f)
        b = lerp(t[1], t[2], f)
        pa, pb = project(a, K), project(b, K)
        if pa and pb:
            cv2.line(img, tuple(np.int32(pa)), tuple(np.int32(pb)),
                     GREEN, 1, cv2.LINE_AA)


def draw_edge_path(img, box, K, n):
    """Passes at several heights down each vertical face, like edging it."""
    t, b = box['top_m'], box['base_m']
    for i in range(4):
        for s in range(1, EDGE_STEPS + 1):
            f = s / (EDGE_STEPS + 1.0)
            p = lerp(t[i], b[i], f)
            q = lerp(t[(i + 1) % 4], b[(i + 1) % 4], f)
            pp, qq = project(p, K), project(q, K)
            if pp and qq:
                cv2.line(img, tuple(np.int32(pp)), tuple(np.int32(qq)),
                         ORANGE, 1, cv2.LINE_AA)


def main():
    cam = CameraSession(build_pipeline)
    kb = Keys()
    K = None
    view = 0
    frozen = None
    last = None
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print('\nPut a box on the table in front of the arm.')
    print('The arm is NOT commanded by this - it only looks.\n')
    try:
        while True:
            if frozen is None:
                g = cam.get()
                if g is None:
                    if cam.crashed() and not cam.restart():
                        break
                    kb.pump()
                    if kb.hit(ord('q')):
                        break
                    continue
                image = g['rgb'].getCvFrame()
                depth = g['depth'].getFrame().astype(np.float32) / 1000.0
                if K is None:
                    K, ksrc = get_K(image.shape[1], image.shape[0])
                    print(f'[box] intrinsics from {ksrc}')
                table = fit_table(depth, K, tc.DETECT_Z_MIN_M, tc.DETECT_Z_MAX_M)
                box = (find_box(depth, K, table, tc.DETECT_Z_MIN_M,
                                tc.DETECT_Z_MAX_M) if table else None)
                last = (image, depth, table, box)
            image, depth, table, box = last

            if view == 2:
                d = np.clip((depth - tc.DETECT_Z_MIN_M)
                            / max(1e-6, tc.DETECT_Z_MAX_M - tc.DETECT_Z_MIN_M), 0, 1)
                d[~np.isfinite(depth)] = 0
                disp = cv2.applyColorMap((d * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
            else:
                disp = image.copy()

            lines = []
            if table is None:
                lines.append(('cannot see the table - is anything in range?',
                              (80, 140, 255)))
            elif box is None:
                lines.append(('table found, no box on it', (80, 140, 255)))
            else:
                n, _c = table
                top = np.int32(box['top_px'])
                base = np.int32(box['base_px'])
                # the four vertical edges first, so the top face sits over them
                for i in range(4):
                    cv2.line(disp, tuple(top[i]), tuple(base[i]),
                             ORANGE, 2, cv2.LINE_AA)
                cv2.polylines(disp, [base], True, GREY, 1, cv2.LINE_AA)
                cv2.polylines(disp, [top], True, GREEN, 2, cv2.LINE_AA)
                for p in top:
                    cv2.circle(disp, tuple(p), 4, GREEN, -1)
                if view == 1:
                    draw_top_path(disp, box, K, n)
                    draw_edge_path(disp, box, K, n)
                cx, cy = np.int32(box['centre_px'])
                cv2.putText(disp, 'TOP  flat cut', (top[:, 0].min(),
                                                    max(14, top[:, 1].min() - 8)),
                            FONT, 0.5, GREEN, 1, cv2.LINE_AA)
                cv2.putText(disp, 'SIDE  edge cut', (base[:, 0].min(),
                                                     min(disp.shape[0] - 6,
                                                         base[:, 1].max() + 16)),
                            FONT, 0.5, ORANGE, 1, cv2.LINE_AA)
                # Sorted rather than labelled width/depth: which side the
                # rectangle fit calls "width" depends on the box's rotation,
                # so naming them invites someone checking with a tape measure
                # to think it is wrong when it is not.
                w, d_ = sorted((box['width_m'] * 100, box['depth_m'] * 100),
                               reverse=True)
                h = box['height_m'] * 100
                lines.append((f'box {w:.1f} x {d_:.1f} cm, {h:.1f} cm tall',
                              GREEN))
                lines.append((f'top face {w*d_:.0f} cm2 to mow, '
                              f'{2*(w+d_):.0f} cm of edge to trim, {h:.1f} cm deep',
                              ORANGE))

            lines.append((f'view: {VIEWS[view]}   m = change   '
                          f'f = {"unfreeze" if frozen is not None else "freeze"}'
                          f'   q = quit', (200, 200, 200)))
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 24 * len(lines) + 8),
                          (16, 16, 16), -1)
            for i, (t_, c_) in enumerate(lines):
                cv2.putText(disp, t_, (10, 20 + i * 23), FONT, 0.48, c_, 1,
                            cv2.LINE_AA)
            if frozen is not None:
                cv2.putText(disp, 'FROZEN', (disp.shape[1] - 90, 20), FONT,
                            0.6, (60, 200, 255), 2)
            cv2.imshow(WIN, disp)

            kb.pump(1)
            if kb.hit(ord('q')):
                break
            if kb.hit(ord('m')):
                view = (view + 1) % len(VIEWS)
            if kb.hit(ord('f')):
                frozen = None if frozen is not None else True
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
