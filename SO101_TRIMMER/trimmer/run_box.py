"""Replay the taught box trim, with live adjustment before the demo.

    python trimmer/run_box.py

Teaching happens in teach_box.py. This is the thing you actually run in front
of someone: it plays the taught paths, and it lets you nudge them first.

The nudging is the point. A taught path is tied to the box being exactly where
it was when you taught it, and it never is - it gets knocked, or the table
gets moved, or you are demoing somewhere else entirely. Re-teaching five paths
before a demo is not realistic. So instead every path can be shifted by a
per-joint OFFSET, applied to every waypoint at once: move the arm to where the
first waypoint SHOULD be, press SPACE, and the whole path shifts with it.

    JOG          moves the arm now, so you can see where things are
    OFFSET       shifts the whole taught path, and is what gets saved

Offsets persist in core/box_offsets.py, so a demo setup survives a restart.

Keys
    1 2 3 4 5   pick a joint     (5 is the HEAD - it rolls the cutting plane)
    A / D       jog that joint now              b / n   step size
    O / L       OFFSET that joint in the path   (whole path shifts)
    SPACE       set the offset so the FIRST waypoint is where the arm is now
    z           clear the offsets for this path
    TAB         next path        p   play it        P   play all of them
    h           go to the first waypoint, without playing
    r           REST            e   stop, mid-move
    s           save offsets    q   quit
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
from keys import Keys
import pathplay
from pathplay import (play, goto, head_envelope, to_arc, head_ok,
                      head_deg, head_label)
import selflearn

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'box trim - replay and adjust'
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID),
          ('HEAD', tc.HEAD_ID)]
TICKS_PER_DEG = 4096.0 / 360.0
PATHS = ['top', 'side_1', 'side_2', 'side_3', 'side_4']
PLAY_SPEED, PLAY_ACC = 700, 10
OFFSET_STEP = 10
OUT = 'box_offsets.py'


def load_paths():
    try:
        from box_paths import BOX_PATHS
        return {k: [dict(w) for w in BOX_PATHS.get(k, [])] for k in PATHS}
    except Exception:
        return {k: [] for k in PATHS}


def load_offsets():
    try:
        from box_offsets import BOX_OFFSETS
        return {k: dict(BOX_OFFSETS.get(k, {})) for k in PATHS}
    except Exception:
        return {k: {} for k in PATHS}


def save_offsets(offsets, log):
    from paths import CORE
    p = CORE / OUT
    with p.open('w') as f:
        f.write('# Written by trimmer/run_box.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# Per-joint shifts applied to the taught box paths, so a\n')
        f.write('# demo can be lined up without re-teaching anything.\n\n')
        f.write('BOX_OFFSETS = {\n')
        for k in PATHS:
            f.write(f'    {k!r}: {{'
                    + ', '.join(f'{j}: {v}' for j, v in sorted(offsets[k].items())
                                if v) + '},\n')
        f.write('}\n')
    log(f'saved offsets -> core/{OUT}')


def shifted(waypoints, off):
    """The taught path with the offsets applied.

    The head is shifted modulo 4096 so an offset that carries it across the
    servo zero still lands on a real angle rather than a negative number.
    """
    out = []
    for wp in waypoints:
        w = {}
        for j, v in wp.items():
            j = int(j)
            v = int(v) + int(off.get(j, 0))
            w[j] = v % 4096 if j == tc.HEAD_ID else max(0, min(4095, v))
        out.append(w)
    return out


def main():
    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return

    taught = load_paths()
    offsets = load_offsets()
    n_wp = sum(len(v) for v in taught.values())
    if not n_wp:
        print('nothing taught yet - run  python trimmer/teach_box.py  first')
        arm.close()
        return
    print(f'\n{n_wp} taught waypoints across '
          f'{sum(1 for k in PATHS if taught[k])} path(s)')
    if getattr(tc, 'AUTO_LEARN', False):
        print('AUTO_LEARN is on: after each play the arm walks the path again,'
              '\n  reads where it landed, and corrects itself. Every run is'
              '\n  closer than the last. Press e during it to skip.')

    which, sel, jog = 0, 0, 3.0
    kb = Keys()
    msg = ''
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def log(t):
        nonlocal msg
        msg = t
        print(f'  {t}')

    def stop():
        return kb.pump().hit(ord('e'))

    try:
        while True:
            name = PATHS[which]
            path = shifted(taught[name], offsets[name])
            kb.pump()
            try:
                pose = arm.read_pose()
            except TrimmerError:
                pose = {}
            kb.pump()

            disp = np.zeros((330, 900, 3), np.uint8)
            off = offsets[name]
            jl = '  '.join(f"{'[' if i == sel else ' '}{i+1}:{n[:4]}"
                           f"{']' if i == sel else ' '}"
                           for i, (n, _) in enumerate(JOINTS))
            rows = [
                (f'{name.upper()}   {len(path)} waypoints', (0, 255, 255)),
                (jl + f'      jog step {jog:.0f} deg', (90, 255, 90)),
                ('offsets  ' + ('  '.join(f'{n[:4]}{off.get(sid,0):+d}'
                                          for n, sid in JOINTS if off.get(sid))
                                or 'none - path plays exactly as taught'),
                 (255, 220, 90)),
            ]
            if pose:
                rows.append(('arm now  ' + '  '.join(
                    f'{n[:4]}:{pose.get(sid,"-")}' for n, sid in JOINTS
                    if sid != tc.HEAD_ID)
                    + ('   HEAD ' + head_label(pose[tc.HEAD_ID])
                       if tc.HEAD_ID in pose else ''), (180, 180, 180)))
            if path:
                w0 = path[0]
                rows.append(('waypoint 1 ' + '  '.join(
                    f'{n[:4]}:{w0.get(sid,"-")}' for n, sid in JOINTS
                    if sid != tc.HEAD_ID)
                    + ('   HEAD ' + head_label(w0[tc.HEAD_ID])
                       if tc.HEAD_ID in w0 else ''), (180, 180, 180)))
                hv = [w[tc.HEAD_ID] for w in path if tc.HEAD_ID in w]
                if hv and not all(head_ok(v) for v in hv):
                    rows.append(('HEAD out of the safe arc after offset - '
                                 'play will refuse', (80, 140, 255)))
            rows.append(('A/D jog   O/L offset   SPACE line up   z clear   '
                         'TAB next   p play   P all   h first   s save   q quit',
                         (200, 200, 200)))
            if msg:
                rows.append((msg, (255, 220, 90)))
            for i, (t_, c_) in enumerate(rows):
                cv2.putText(disp, t_, (12, 28 + i * 30), FONT, 0.5, c_, 1,
                            cv2.LINE_AA)
            x = 12
            for nm in PATHS:
                col = (90, 255, 90) if taught[nm] else (80, 80, 80)
                if nm == name:
                    cv2.rectangle(disp, (x - 4, 300), (x + 150, 324),
                                  (55, 55, 55), -1)
                cv2.putText(disp, f'{nm} {len(taught[nm])}', (x, 318),
                            FONT, 0.45, col, 1)
                x += 165
            cv2.imshow(WIN, disp)

            kb.pump(1)
            if kb.hit(ord('q')):
                break
            if kb.hit(9):
                which = (which + 1) % len(PATHS)
            for i, code in enumerate((ord('1'), ord('2'), ord('3'),
                                      ord('4'), ord('5'))):
                if kb.hit(code):
                    sel = i
            if kb.hit(ord('b')):
                jog = min(20.0, jog + 2)
            if kb.hit(ord('n')):
                jog = max(1.0, jog - 2)

            step = 0.0
            if kb.tap(ord('a')):
                step = -jog
            elif kb.tap(ord('d')):
                step = +jog
            if step:
                _nm, sid = JOINTS[sel]
                try:
                    cur = arm.read_position(sid)
                    if cur is not None:
                        arm.move_to(sid, int(cur + step * TICKS_PER_DEG))
                except TrimmerError as e:
                    log(f'move failed: {e}')

            d_off = 0
            if kb.tap(ord('o')):
                d_off = OFFSET_STEP
            elif kb.tap(ord('l')):
                d_off = -OFFSET_STEP
            if d_off:
                _nm, sid = JOINTS[sel]
                off[sid] = off.get(sid, 0) + d_off
                log(f'{name} {_nm} offset {off[sid]:+d}')

            if kb.hit(32) and path and pose:
                # line the whole path up with where the arm is standing now
                w0 = taught[name][0]
                for _nm, sid in JOINTS:
                    if sid in w0 and sid in pose:
                        off[sid] = int(pose[sid]) - int(w0[sid])
                log(f'{name} lined up to the arm - offsets '
                    + ' '.join(f'{n[:4]}{off.get(s,0):+d}' for n, s in JOINTS))
            if kb.hit(ord('z')):
                offsets[name] = {}
                log(f'{name} offsets cleared')
            if kb.hit(ord('h')) and path:
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                log('going to waypoint 1')
                goto(arm, path[0], log=log, stop_check=stop)
            if kb.hit(ord('r')) and tc.REST_POSE:
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                goto(arm, tc.REST_POSE, log=log, stop_check=stop)
                log('at REST')
            if kb.hit(ord('p')) and path:
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                log(f'playing {name}')
                play(arm, path, log, stop_check=stop, name=name)
                if getattr(tc, 'AUTO_LEARN', False):
                    log('learning from that run - press e to skip')
                    selflearn.learn(arm, path, name,
                                    cycles=getattr(tc, 'AUTO_LEARN_PASSES', 2),
                                    log=log, stop_check=stop)
                    arm.set_profile(PLAY_SPEED, PLAY_ACC)
                    pathplay.set_rate(PLAY_SPEED)
            if kb.hit(ord('P')):
                arm.set_profile(PLAY_SPEED, PLAY_ACC)
                pathplay.set_rate(PLAY_SPEED)
                allp = [shifted(taught[k], offsets[k]) for k in PATHS if taught[k]]
                env = head_envelope(*allp)
                if env:
                    log(f'head stays within the taught arc range')
                for k in PATHS:
                    if not taught[k]:
                        continue
                    log(f'playing {k}')
                    if not play(arm, shifted(taught[k], offsets[k]), log,
                                envelope=env, stop_check=stop, name=k):
                        log('stopped - not continuing')
                        break
                else:
                    log('all paths done')
            if kb.hit(ord('s')):
                save_offsets(offsets, log)
    finally:
        try:
            if tc.REST_POSE:
                arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
                goto(arm, tc.REST_POSE)
            arm.torque_off()
        except Exception:
            pass
        arm.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
