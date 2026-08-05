"""Minimal line head: press onto the shaft, thread the line through. Writes STL.

    python cad/trimmer_head.py --gauge     # print this FIRST, 2 minutes
    python cad/trimmer_head.py             # then the head, 2.3 mm shaft
    python cad/trimmer_head.py --clear 0.1 # whatever bore actually fitted

It is a plug with two holes:

    a blind bore down the axis   - squeeze fits onto the motor shaft
    a cross hole near the bottom - the line goes straight through

The cross hole runs right through the diameter, below the bottom of the shaft
bore so the two never meet. That one detail does most of the work: a single
length of line threaded all the way through comes out equal on both sides, so
it is balanced without being adjusted, and centrifugal force pulls on one
continuous piece rather than on any kind of clip or knot. There is nothing to
retain the line because the line retains itself.

The bore is the only dimension that matters, and it depends on your printer
more than on anything here - which is what the gauge is for.
"""
import argparse
import math
import os

import trimesh
from trimesh.creation import cylinder

# ---- millimetres -----------------------------------------------------------
SHAFT_D = 2.3           # measured motor shaft
BORE_CLEAR = 0.0        # bore = shaft + this. 0 is a press fit; use the gauge.
HUB_D = 14.0            # outside diameter
HUB_H = 18.0            # grip length + room for the line hole under the bore
BORE_DEPTH = 12.0       # blind, from the top - grip length on the shaft
LINE_D = 2.2            # trimmer line + a little
LINE_Z = 2.8            # cross hole height. With a 12 mm bore in an 18 mm hub
                        # the bore floor is at 6, so this leaves 2.1 mm of
                        # material between the two holes and 1.7 mm under the
                        # line - thin enough to matter, so it is checked below.
SEG = 96


def _cyl(d, h, x=0.0, y=0.0, z=0.0, axis='z'):
    c = cylinder(radius=d / 2.0, height=h, sections=SEG)
    if axis == 'y':
        c.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))
    c.apply_translation([x, y, z])
    return c


def build(shaft_d=SHAFT_D, clear=BORE_CLEAR):
    bore = shaft_d + clear
    body = _cyl(HUB_D, HUB_H, z=HUB_H / 2.0)

    # Shaft bore, blind from the top. Placed by its FLOOR, not its centre:
    # the first version put the cylinder centre where the floor should have
    # been and produced a 6 mm bore instead of 12, which is exactly the sort
    # of thing that looks fine in a render.
    floor = HUB_H - BORE_DEPTH
    cut_h = BORE_DEPTH + 10.0
    body = body.difference(_cyl(bore, cut_h, z=floor + cut_h / 2.0))
    # lead-in, so it starts square on the shaft instead of catching a corner
    lead = trimesh.creation.cone(radius=bore / 2.0 + 0.8, height=1.4, sections=SEG)
    lead.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    lead.apply_translation([0, 0, HUB_H])
    body = body.difference(lead)

    # the line: straight through the diameter, perpendicular to the shaft
    body = body.difference(_cyl(LINE_D, HUB_D * 2, z=LINE_Z, axis='y'))
    return body


def gauge(shaft_d=SHAFT_D):
    """Four bores around nominal. Find the one that fits, then print the head.

    Printers vary by more than the tolerance that decides a press fit at
    2.3 mm, so this is the difference between one print and four.
    """
    sizes = (-0.1, 0.0, 0.1, 0.2)
    plate = trimesh.creation.box(extents=[46, 16, 6])
    plate.apply_translation([0, 0, 3])
    for i, c in enumerate(sizes):
        plate = plate.difference(
            _cyl(shaft_d + c, 20, x=-16.5 + i * 11, z=3))
    return plate, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shaft', type=float, default=SHAFT_D)
    ap.add_argument('--clear', type=float, default=BORE_CLEAR)
    ap.add_argument('--gauge', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    if args.gauge:
        m, sizes = gauge(args.shaft)
        out = args.out or os.path.join(here, f'bore_gauge_{args.shaft:.1f}mm.stl')
        m.export(out)
        print(f'{out}')
        print('   bores: ' + ', '.join(f'{args.shaft + c:.2f}' for c in sizes)
              + '  (left to right)')
        return

    m = build(args.shaft, args.clear)
    out = args.out or os.path.join(here, f'trimmer_head_{args.shaft:.1f}mm.stl')
    m.export(out)
    b = m.bounds
    print(f'{out}')
    print(f'   watertight : {m.is_watertight}')
    print(f'   size       : {b[1][0]-b[0][0]:.0f} dia x {b[1][2]-b[0][2]:.0f} tall mm')
    print(f'   volume     : {m.volume/1000:.2f} cm3  (~{m.volume/1000*1.24:.1f} g PLA)')
    print(f'   bore       : {args.shaft + args.clear:.2f} mm, {BORE_DEPTH:.0f} mm deep')
    print(f'   line hole  : {LINE_D:.1f} mm, straight through, {LINE_Z:.1f} mm up')


if __name__ == '__main__':
    main()
