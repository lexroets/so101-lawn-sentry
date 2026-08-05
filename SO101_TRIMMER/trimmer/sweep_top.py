"""Sweep the top of the box at the calibrated height - no taught path.

    python trimmer/sweep_top.py            # show the plan, move nothing
    python trimmer/sweep_top.py --run      # sweep it
    python trimmer/sweep_top.py --save     # write it in as the 'top' path

WHY THIS CAN EXIST NOW
----------------------
Until the plane was calibrated there was nothing to generate a path FROM, so
the only way to cut the top was to drag the arm across it by hand and replay
that. A dragged path carries the hand that dragged it: it bobs, it wanders,
and it only covers where the hand happened to go.

Three things are now known independently of any taught path:

    THE HEIGHT.   level_plane.py measured the cutting surface, and the arm
                  model turns "this height" into "this elbow, at this reach".

    THE ATTITUDE. The head's pitch is whatever holds it flat, derived from
                  the shoulder and elbow rather than taught.

    THE REACH.    The published link lengths say where the head is for any
                  pose, so a row can be placed at a chosen distance out.

That is everything needed to describe a cut. The taught path is no longer the
definition of the motion, only of the AREA - how far out the box sits and how
wide it is.

WHY A ROW IS A BASE SWEEP
-------------------------
Within one row the shoulder and the elbow do not move at all. Only the base
turns. The base rotates about a vertical axis, so it moves the head sideways
and not up or down - which means the height along a row is not merely close to
constant, it is identical by construction, with no model error to accumulate.

Stepping the shoulder between rows moves the cut in and out, and the elbow
follows from the height model at that new reach. So every pose in the pattern
sits on the calibrated plane, and the only place the model has to be right is
at the handful of row heights - not at every point along the way.

Rows alternate direction, so the head sweeps out, steps over, and sweeps back
without lifting or returning empty.
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
import sys
import time

import numpy as np

import trimmer_config as tc
import pathplay
from pathplay import arm_model, elbow_for_height, height_mm, goto, play

ROWS = 6                # passes in and out across the box
PER_ROW = 9             # poses along each sweep - the player smooths between
EDGE_MARGIN_RAW = 40    # keep clear of the very edge of what was taught


def area():
    """How far out and how wide to cut, taken from the taught pass.

    The taught path stops being the motion and becomes only the AREA. Its
    numbers are used for nothing except "the box is this far out and this
    wide", which is the one thing a plane cannot tell you.
    """
    from box_paths import BOX_PATHS
    wps = BOX_PATHS.get('top') or []
    if len(wps) < 4:
        return None
    base = [w[tc.BASE_ID] for w in wps]
    sh = [w[tc.SHOULDER_ID] for w in wps]
    lo, hi = min(sh) + EDGE_MARGIN_RAW, max(sh) - EDGE_MARGIN_RAW
    try:
        from height_plane import SHOULDER_RANGE
        # never plan a row where the plane was not measured - that is the one
        # place the height would be a guess
        lo = max(lo, SHOULDER_RANGE[0])
        hi = min(hi, SHOULDER_RANGE[1])
    except Exception:
        pass
    return (min(base) + EDGE_MARGIN_RAW, max(base) - EDGE_MARGIN_RAW, lo, hi)


def plan(rows=ROWS, per_row=PER_ROW, mm=None):
    """Build the sweep. Every pose sits on the calibrated plane."""
    from height_plane import TOUCHES
    model = arm_model(TOUCHES)
    if model is None:
        return None, None
    box = area()
    if box is None:
        return None, None
    b0, b1, s0, s1 = box
    if s1 - s0 < 60 or b1 - b0 < 60:
        return None, None
    mm = float(getattr(tc, 'LEVEL_HEIGHT_TRIM_MM', 0.0)) if mm is None else mm
    pitch = pitch_target()

    # the height model has to be right at the row reaches; settle UP once here
    if not model.get('UP_SET'):
        from box_paths import BOX_PATHS
        w = BOX_PATHS['top']
        h = height_mm(model, np.array([x[tc.SHOULDER_ID] for x in w], float),
                      np.array([x[tc.ELBOW_ID] for x in w], float))
        model['UP'] = -1.0 if float(np.mean(h)) < 0 else 1.0
        model['UP_SET'] = True

    shoulders = np.linspace(s0, s1, rows)
    elbows = elbow_for_height(model, shoulders, mm,
                              np.full(rows, float(s0 + s1) * 0.0 + 2000.0))
    # seed the search from a taught elbow at a similar reach, so it stays in
    # the configuration the arm was taught in
    from box_paths import BOX_PATHS
    tw = BOX_PATHS['top']
    tsh = np.array([w[tc.SHOULDER_ID] for w in tw], float)
    tel = np.array([w[tc.ELBOW_ID] for w in tw], float)
    near = np.array([tel[int(np.argmin(np.abs(tsh - s)))] for s in shoulders])
    elbows = elbow_for_height(model, shoulders, mm, near)

    out = []
    for i, (s, e) in enumerate(zip(shoulders, elbows)):
        s, e = int(round(s)), int(round(e))
        wrist = int(pitch - s - e)
        if not (0 <= wrist <= 4095):
            continue
        sweep = np.linspace(b0, b1, per_row)
        if i % 2:
            sweep = sweep[::-1]          # alternate, so it does not fly back
        for b in sweep:
            out.append({tc.BASE_ID: int(round(b)), tc.SHOULDER_ID: s,
                        tc.ELBOW_ID: e, tc.WRIST_ID: wrist,
                        tc.HEAD_ID: int(getattr(tc, 'HEAD_FLAT_RAW', 2048))})
    return out, model


def pitch_target():
    from box_paths import BOX_PATHS
    w = BOX_PATHS['top']
    s = sorted(x[tc.SHOULDER_ID] + x[tc.ELBOW_ID] + x[tc.WRIST_ID] for x in w)
    fixed = getattr(tc, 'LEVEL_PITCH_TARGET', {}).get('top')
    return (fixed if fixed is not None else s[len(s) // 2]) + \
        getattr(tc, 'LEVEL_PITCH_TRIM', 0)


def show(poses, model):
    sh = np.array([p[tc.SHOULDER_ID] for p in poses], float)
    el = np.array([p[tc.ELBOW_ID] for p in poses], float)
    h = height_mm(model, sh, el)
    print(f'\n{len(poses)} poses, {ROWS} rows of {PER_ROW}\n')
    print(f'{"row":>4} {"shoulder":>9} {"elbow":>7} {"wrist":>7} '
          f'{"base sweep":>16} {"height":>9}')
    for i in range(0, len(poses), PER_ROW):
        r = poses[i:i + PER_ROW]
        print(f'{i // PER_ROW + 1:>4} {r[0][tc.SHOULDER_ID]:9d} '
              f'{r[0][tc.ELBOW_ID]:7d} {r[0][tc.WRIST_ID]:7d} '
              f'{r[0][tc.BASE_ID]:6d} -> {r[-1][tc.BASE_ID]:<6d} '
              f'{h[i]:7.1f}mm')
    print(f'\nheight across every pose: {h.min():.2f} .. {h.max():.2f} mm '
          f'(spread {h.max() - h.min():.2f} mm)')
    print('within a row it cannot vary at all - only the base moves, and the')
    print('base turns about a vertical axis.')


def main():
    poses, model = plan()
    if not poses:
        print('need a taught top pass (for the area) and a calibrated plane '
              '(trimmer/level_plane.py)')
        return
    show(poses, model)

    if '--save' in sys.argv:
        from paths import CORE
        import shutil
        src = CORE / 'box_paths.py'
        d = CORE / 'backups'
        d.mkdir(exist_ok=True)
        shutil.copy(src, d / f'box_paths-{time.strftime("%Y%m%d-%H%M%S")}.py')
        from box_paths import BOX_PATHS
        allp = {k: [dict(w) for w in v] for k, v in BOX_PATHS.items()}
        allp['top'] = poses
        with src.open('w') as f:
            f.write('# top pass GENERATED by trimmer/sweep_top.py '
                    + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
            f.write('# every pose sits on the plane taught in level_plane.py\n\n')
            f.write('BOX_PATHS = {\n')
            for k, v in allp.items():
                f.write(f'    {k!r}: [\n')
                for w in v:
                    f.write('        {' + ', '.join(f'{a}: {b}'
                            for a, b in sorted(w.items())) + '},\n')
                f.write('    ],\n')
            f.write('}\n')
        for c in (CORE / '__pycache__').glob('box_paths.*.pyc'):
            try:
                c.unlink()
            except OSError:
                pass
        print(f'\nwritten in as the top path (previous kept in core/backups).')
        print('run_box will use it, and will learn from it like any other.')
        return

    if '--run' not in sys.argv:
        print('\nnothing moved. --run to sweep it, --save to make it the top path.')
        return

    from trimmer_arm import TrimmerArm
    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(200, 10)
        pathplay.set_rate(200)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return
    try:
        print('\nsweeping - Ctrl-C to stop')
        play(arm, poses, print, name='top')
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        try:
            if tc.REST_POSE:
                goto(arm, tc.REST_POSE)
            arm.torque_off()
        except Exception:
            pass
        arm.close()


if __name__ == '__main__':
    main()
