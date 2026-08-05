"""Teach the arm to trim a box by hand, then have it replay the moves.

    python trimmer/teach_box.py

Autonomy comes later. This is the step before it: put a box in a fixed place,
show the arm what a top pass looks like and what a side pass looks like, and
get it repeating them on command. Everything after that - finding the box,
choosing the path - is a different problem, and there is no point solving it
while the motion itself is still unproven.

Five paths are taught, each a list of poses:

    TOP     across the top face, flat, the way a mower would
    SIDE 1-4   down each vertical face, the way an edger would

The head joint (5) is what makes the second one possible - it rolls the
cutting plane from horizontal to vertical. Set it per waypoint like any other
joint; it is recorded and replayed with the rest.

The arm holds nothing back: you move it, it remembers. Torque can be released
so it can be pushed by hand, or left on and jogged with the keys - a mix works
too, hand for the coarse move and keys for the last few degrees.

Nothing is written until you press s, and it goes to core/box_paths.py, which
is a plain Python file you can read and edit.

Keys
    t         torque on/off  - off to move it by hand, on to jog or replay
    1 2 3 4 5 pick a joint            A/D jog it        b/n step size
              5 is the HEAD joint - it rolls the cutting plane, so it is the
              one that turns a top pass into a side pass
    SPACE     record a waypoint here
    g         CONTINUOUS record on/off - drag the arm and it samples the whole
              movement. This is the one to use for a trim pass: four waypoints
              cannot describe a path that has to stay at a constant height.
    u         undo the last waypoint
    TAB       next path (TOP -> SIDE 1 -> ... -> SIDE 4 -> TOP)
    p         PLAY the current path back - stops properly at every waypoint
    P         play every path, in order
    e         stop, mid-move
    r         go to REST
    s         save        c   clear the current path        q   quit
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

import cv2
import numpy as np

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline
from camera_session import CameraSession
from keys import Keys
import pathplay

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'teach the box - top pass and side passes'
# Five joints, not four. Joint 5 is the one the trimmer head is bolted to -
# it rolls the cutting plane, which is the entire difference between mowing a
# top and edging a side. Teaching a side pass without it is not possible: the
# head would be lying flat against a vertical face.
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID),
          ('HEAD', tc.HEAD_ID)]
TICKS_PER_DEG = 4096.0 / 360.0
PATHS = ['top', 'side_1', 'side_2', 'side_3', 'side_4']
LABEL = {'top': 'TOP  — across the face, flat',
         'side_1': 'SIDE 1 — down the face, vertical',
         'side_2': 'SIDE 2 — down the face, vertical',
         'side_3': 'SIDE 3 — down the face, vertical',
         'side_4': 'SIDE 4 — down the face, vertical'}
# Replay is deliberately slower than the arm's normal moves. A taught path has
# no collision checking behind it - the only thing keeping the head out of the
# box is that you put it there yourself - so it goes at a speed you can stop.
# Deliberately slow. A taught path has no collision checking behind
# it, so it runs at a speed you can watch and interrupt.
PLAY_SPEED = 700
PLAY_ACC = 10
OUT = 'box_paths.py'

# --- continuous teaching -----------------------------------------------------
# WHY THIS EXISTS
# Four waypoints do not describe a trim pass. Between two of them the player
# interpolates in JOINT space - every joint moves in proportion - and that is a
# straight line for the joints, not for the head. The head traces a curve, and
# over a 140 degree shoulder swing that curve sags well below both ends. The
# arm is doing exactly what it was told; it was told too little.
#
# No amount of cleverness in the player fixes that, because the height it
# should hold was never recorded. Dragging the arm along the cut and sampling
# it does record it: adjacent points end up a few ticks apart, so there is
# nowhere for the path to sag between them.
AUTO_RECORD_RAW = 25        # sample once any joint has moved this far
AUTO_RECORD_MIN_S = 0.06
# Thinning tolerance. A sample on a straight run between its neighbours adds
# nothing - removing it leaves the played path identical - so only the points
# that carry the shape are kept.
SIMPLIFY_TOL_RAW = 18


def _dev(wp, a, b):
    """How far wp sits off the straight line from a to b, in joint space."""
    keys = [k for k in wp if k in a and k in b]
    num = den = 0.0
    for k in keys:
        d = float(b[k] - a[k])
        num += (float(wp[k]) - a[k]) * d
        den += d * d
    t = 0.0 if den == 0 else max(0.0, min(1.0, num / den))
    return max(abs(float(wp[k]) - (a[k] + (b[k] - a[k]) * t)) for k in keys)


def simplify(path, tol=SIMPLIFY_TOL_RAW):
    """Drop samples the played path would pass through anyway.

    Ramer-Douglas-Peucker, in joint space. It only ever removes a point that
    lies within `tol` of the straight line its neighbours already describe, so
    the motion is unchanged - there is just less of it written down.
    """
    if len(path) < 3:
        return list(path)
    worst, at = -1.0, 0
    for i in range(1, len(path) - 1):
        d = _dev(path[i], path[0], path[-1])
        if d > worst:
            worst, at = d, i
    if worst <= tol:
        return [path[0], path[-1]]
    return simplify(path[:at + 1], tol)[:-1] + simplify(path[at:], tol)


# The player lives in core/pathplay.py. It interpolates each move instead of
# commanding a pose and hoping, which is what stops the arm cutting a corner
# through the table between two taught points, and it resolves the head angle
# to the nearest equivalent so it never takes the long way round.
from pathplay import play, goto, head_envelope   # noqa: E402


def save(paths, log):
    """Thin the paths and write them out.

    IN PLACE, deliberately. An earlier version thinned a copy, so the file got
    36 points and the session kept all 136 - and replaying from the session
    played something the file did not describe. Whatever is saved is what is
    now held.
    """
    from paths import CORE
    before = sum(len(v) for v in paths.values())
    for k in list(paths):
        paths[k] = simplify(paths[k])
    after = sum(len(v) for v in paths.values())
    if after < before:
        log(f'thinned {before} samples to {after} - the path is unchanged')
    p = CORE / OUT
    backup(p, log)
    with p.open('w') as f:
        f.write('# Written by trimmer/teach_box.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# Taught trim paths for a box in a fixed position.\n')
        f.write('# Each is a list of 4-joint poses, played in order.\n\n')
        f.write('BOX_PATHS = {\n')
        for name in PATHS:
            f.write(f'    {name!r}: [\n')
            for wp in paths[name]:
                f.write('        {' + ', '.join(f'{k}: {v}'
                                                for k, v in sorted(wp.items())) + '},\n')
            f.write('    ],\n')
        f.write('}\n')
    log(f'saved {sum(len(v) for v in paths.values())} waypoints -> core/{OUT}')


def backup(p, log):
    """Keep a TIMESTAMPED copy, and only when the content actually changed.

    The previous scheme kept exactly one .bak and rewrote it on every save.
    Pressing save several times with no edits in between therefore overwrote
    the one good copy with identical ones, and a taught plane was lost with
    no way back. One slot is not a backup; it is a single undo that any
    repeated action destroys.

    So: a folder of timestamped copies, and nothing written when the file has
    not changed - a no-op save can no longer consume the history.
    """
    if not p.is_file():
        return
    old = p.read_text()
    body = old[old.index('BOX_PATHS'):] if 'BOX_PATHS' in old else old
    d = p.parent / 'backups'
    d.mkdir(exist_ok=True)
    # BY TIME, not by name. Sorted by name, 'box_paths-...-3.py' sorts before
    # 'box_paths-....py', so the "newest" picked was actually the oldest and
    # the unchanged-check compared against the wrong file.
    prev = sorted(d.glob(p.stem + '-*.py'), key=lambda x: x.stat().st_mtime)
    if prev:
        last = prev[-1].read_text()
        if 'BOX_PATHS' in last and last[last.index('BOX_PATHS'):] == body:
            return                      # nothing changed - keep the history
    stamp = time.strftime('%Y%m%d-%H%M%S')
    # Two saves inside the same second would otherwise land on the same name
    # and the second would overwrite the first - the exact failure this whole
    # function exists to prevent, reintroduced one level down.
    out = d / f'{p.stem}-{stamp}.py'
    n = 1
    while out.exists():
        out = d / f'{p.stem}-{stamp}-{n}.py'
        n += 1
    out.write_text(old)
    for f in sorted(d.glob(p.stem + '-*.py'),
                    key=lambda x: x.stat().st_mtime)[:-40]:
        f.unlink()                      # keep the last 40
    log(f'previous version kept as backups/{out.name}')


def load_existing():
    try:
        from box_paths import BOX_PATHS
        return {k: [dict(w) for w in BOX_PATHS.get(k, [])] for k in PATHS}
    except Exception:
        return {k: [] for k in PATHS}


def main():
    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return
    try:
        cam = CameraSession(build_pipeline)
    except Exception as e:
        print(f'camera unavailable ({e}) - teaching without a view')
        cam = None

    paths = load_existing()
    have = sum(len(v) for v in paths.values())
    if have:
        print(f'loaded {have} waypoints already taught - TAB to a path and '
              f'press c to clear it, or keep adding')

    which = 0
    sel = 0
    jog = 3.0
    auto = False                 # continuous recording
    last_rec = None
    last_rec_t = 0.0
    torque = True
    arm.enable()
    arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
    kb = Keys()
    msg = ''
    frame = None
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def log(t):
        nonlocal msg
        msg = t
        print(f'  {t}')

    print('\nPut the box in its place. Teach TOP first, then the four sides.')
    print('Press t to release the torque and move it by hand.\n')
    try:
        while True:
            if cam is not None:
                g = cam.get()
                if g is not None:
                    frame = g['rgb'].getCvFrame()
            disp = (frame.copy() if frame is not None
                    else np.zeros((360, 640, 3), np.uint8))

            name = PATHS[which]
            kb.pump()
            try:
                pose = arm.read_pose()
            except TrimmerError:
                pose = {}
            kb.pump()

            jl = ' '.join(f"{'[' if i == sel else ''}{i+1}:{n[:4]}"
                          f"{']' if i == sel else ''}"
                          for i, (n, _) in enumerate(JOINTS))
            head = [
                (LABEL[name] + f'   ({len(paths[name])} waypoints)', (0, 255, 255)),
                (f'torque {"ON  - jog with A/D" if torque else "OFF - move it by hand"}'
                 f'    {jl}   step {jog:.0f} deg', (90, 255, 90) if torque
                 else (60, 200, 255)),
                ('SPACE record   g CONTINUOUS   u undo   TAB next path   '
                 'p play   P play all   s save   q quit', (200, 200, 200)),
            ]
            if auto:
                head.append((f'RECORDING - {len(paths[name])} samples   '
                             f'drag the arm, g to stop', (60, 60, 255)))
            if pose:
                head.append(('now  ' + '  '.join(
                    f'{n[:4]}:{pose.get(sid, "-")}' for n, sid in JOINTS
                    if sid != tc.HEAD_ID)
                    + ('   HEAD ' + pathplay.head_label(pose[tc.HEAD_ID])
                       if tc.HEAD_ID in pose else ''),
                    (255, 220, 90)))
            if msg:
                head.append((msg, (255, 220, 90)))
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 24 * len(head) + 8),
                          (16, 16, 16), -1)
            for i, (t_, c_) in enumerate(head):
                cv2.putText(disp, t_, (10, 20 + i * 23), FONT, 0.46, c_, 1,
                            cv2.LINE_AA)
            # a simple progress strip: how many waypoints each path holds
            x = 10
            for nm in PATHS:
                col = (90, 255, 90) if paths[nm] else (90, 90, 90)
                if nm == name:
                    cv2.rectangle(disp, (x - 3, disp.shape[0] - 30),
                                  (x + 96, disp.shape[0] - 8), (60, 60, 60), -1)
                cv2.putText(disp, f'{nm} {len(paths[nm])}',
                            (x, disp.shape[0] - 14), FONT, 0.42, col, 1)
                x += 104
            cv2.imshow(WIN, disp)

            kb.pump(1)
            if kb.hit(ord('q')):
                break
            if kb.hit(ord('t')):
                torque = not torque
                try:
                    if torque:
                        arm.enable()
                        arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
                        log('torque ON - it will hold position')
                    else:
                        arm.torque_off()
                        log('torque OFF - SUPPORT THE ARM, it will drop')
                except TrimmerError as e:
                    log(f'torque change failed: {e}')
            if kb.hit(9):                       # TAB
                which = (which + 1) % len(PATHS)
                log(f'now teaching {PATHS[which]}')
            # --- continuous capture ---------------------------------------
            # Sampled on MOVEMENT, not on a timer: holding the arm still adds
            # nothing, and a pause while you reposition your hands does not
            # fill the path with a hundred copies of one pose.
            if auto and pose:
                moved = (AUTO_RECORD_RAW if last_rec is None else
                         max((abs(pose[j] - last_rec[j])
                              for j in pose if j in last_rec), default=0))
                if (moved >= AUTO_RECORD_RAW
                        and time.monotonic() - last_rec_t >= AUTO_RECORD_MIN_S):
                    paths[name].append(dict(pose))
                    last_rec = dict(pose)
                    last_rec_t = time.monotonic()
            if kb.hit(ord('g')):
                auto = not auto
                last_rec = None
                if auto:
                    log(f'CONTINUOUS recording {name} - drag the arm along the '
                        f'cut. Press g again to stop.')
                else:
                    log(f'stopped - {len(paths[name])} samples in {name}')
            if kb.hit(32):                      # SPACE
                if pose:
                    paths[name].append(dict(pose))
                    log(f'recorded #{len(paths[name])} for {name}')
                else:
                    log('no pose read - cannot record')
            if kb.hit(ord('u')):
                if paths[name]:
                    paths[name].pop()
                    log(f'undid one - {len(paths[name])} left in {name}')
            if kb.hit(ord('c')):
                paths[name] = []
                log(f'cleared {name}')
            if kb.hit(ord('r')):
                if tc.REST_POSE:
                    arm.enable()
                    torque = True
                    arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
                    goto(arm, tc.REST_POSE, log=log)
                    log('at REST')
            if kb.hit(ord('p')):
                auto = False
                arm.enable()
                torque = True
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                log(f'playing {name}')
                play(arm, paths[name], log, name=name,
                     stop_check=lambda: kb.pump().hit(ord('e')))
                log(f'{name} done')
            if kb.hit(ord('P')):
                auto = False
                arm.enable()
                torque = True
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                # Bound the head across EVERY path, not each one separately.
                # The dangerous move is the one BETWEEN paths: the end of one
                # and the start of the next were never taught as a pair, and
                # a per-path envelope says nothing about that gap.
                env = head_envelope(*[paths[nm] for nm in PATHS if paths[nm]])
                if env:
                    log(f'head stays within {env[0]}-{env[1]} '
                        f'({(env[1]-env[0])/4096*360:.0f} deg of taught travel)')
                for nm in PATHS:
                    if not paths[nm]:
                        continue
                    log(f'playing {nm}')
                    if not play(arm, paths[nm], log, envelope=env, name=nm,
                                stop_check=lambda: kb.pump().hit(ord('e'))):
                        log('stopped - not continuing to the next path')
                        break
                else:
                    log('all paths done')
            if kb.hit(ord('s')):
                save(paths, log)
            if kb.hit(ord('b')):
                jog = min(20.0, jog + 2)
            if kb.hit(ord('n')):
                jog = max(1.0, jog - 2)
            # checked one at a time, because hit() tells you IF a key fired,
            # not which of several did
            for i, code in enumerate((ord('1'), ord('2'), ord('3'), ord('4'),
                                      ord('5'))):
                if kb.hit(code):
                    sel = i
                    log(f'jogging {JOINTS[sel][0]}')
            step = 0.0
            if kb.tap(ord('a')):
                step = -jog
            elif kb.tap(ord('d')):
                step = +jog
            if step:
                if not torque:
                    log('torque is off - press t before jogging')
                else:
                    _n, sid = JOINTS[sel]
                    try:
                        cur = arm.read_position(sid)
                        if cur is not None:
                            arm.move_to(sid, int(cur + step * TICKS_PER_DEG))
                    except TrimmerError as e:
                        log(f'move failed: {e}')
    finally:
        try:
            arm.enable()
            if tc.REST_POSE:
                arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
                arm.move_pose(tc.REST_POSE)
                arm.wait_arrived(tc.REST_POSE)
            arm.torque_off()
        except Exception:
            pass
        arm.close()
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
