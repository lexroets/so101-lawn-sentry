"""Step through a taught path and fix individual poses by hand.

    python trimmer/edit_path.py

Teaching by dragging captures the shape of the cut but not the precision -
a stretch comes out too low, a corner too wide, the head a few degrees off.
Re-teaching the whole pass to fix one section is wasteful, and re-teaching it
by hand reintroduces the wobble that was just smoothed out.

So this walks the path one pose at a time. Select a waypoint, the arm GOES
there, and the jog keys move the arm and rewrite that waypoint together -
what you see is what gets saved. Nothing is guessed and nothing is inferred.

THE EDIT THAT IS ACTUALLY WANTED IS USUALLY NOT ONE POSE
--------------------------------------------------------
"The whole pass is 5 mm too low" is one correction repeated forty times, and
applying it forty times by hand invites forty slightly different answers. So
an adjustment made at one waypoint can be pushed across the path in three
ways:

    w       to every waypoint          - the pass sits wrong as a whole
    f       forwards from here         - it drifts off partway through
    (none)  this waypoint only         - one pose is out

FADING
------
Applying a correction to a RANGE and stopping dead at its edge puts a step
into the path, which is the jolt the smoothing works to avoid. So a ranged
correction fades out over the last few waypoints instead of ending abruptly.

Keys
    TAB         next path
    , .         previous / next waypoint      < >   jump ten
    h           re-send the current waypoint (if you moved the arm by hand)
    1 2 3 4 5   pick a joint                  b / n   jog step
    a / d       jog it - MOVES THE ARM AND EDITS THIS WAYPOINT
    w           apply this waypoint's edit to the WHOLE path
    f           apply it FORWARDS from here, fading out
    r           revert this waypoint      z   revert the whole path
    p           play from here            P   play the whole path
    e           stop, mid-move            t   torque on/off
    s           SAVE to core/box_paths.py (keeps a .bak)      q   quit
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
import shutil
import time

import cv2
import numpy as np

import trimmer_config as tc
from trimmer_arm import TrimmerArm, TrimmerError
from keys import Keys
import pathplay
from pathplay import goto, play, head_label, head_ok, to_arc, arc_len

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'edit the taught path'
JOINTS = [('base', tc.BASE_ID), ('shoulder', tc.SHOULDER_ID),
          ('elbow', tc.ELBOW_ID), ('wrist', tc.WRIST_ID), ('HEAD', tc.HEAD_ID)]
TICKS_PER_DEG = 4096.0 / 360.0
PATHS = ['top', 'side_1', 'side_2', 'side_3', 'side_4']
PLAY_SPEED, PLAY_ACC = 700, 10
# A ranged correction is faded out over this many waypoints at its far end,
# so the path does not step where the correction stops.
FADE = 6
OUT = 'box_paths.py'


def load():
    try:
        from box_paths import BOX_PATHS
        return {k: [dict(w) for w in BOX_PATHS.get(k, [])] for k in PATHS}
    except Exception:
        return {k: [] for k in PATHS}


def save(paths, log):
    from paths import CORE
    p = CORE / OUT
    backup(p, log)
    with p.open('w') as f:
        f.write('# Edited by trimmer/edit_path.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# Taught trim paths for a box in a fixed position.\n\n')
        f.write('BOX_PATHS = {\n')
        for name in PATHS:
            f.write(f'    {name!r}: [\n')
            for wp in paths[name]:
                f.write('        {' + ', '.join(f'{k}: {v}'
                        for k, v in sorted(wp.items())) + '},\n')
            f.write('    ],\n')
        f.write('}\n')
    log(f'saved {sum(len(v) for v in paths.values())} waypoints '
        f'-> core/{OUT}')


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


def nudge(wp, sid, delta):
    """Move one joint of one waypoint, refusing anything off the head's arc."""
    if sid == tc.HEAD_ID:
        want = (int(wp[sid]) + delta) % 4096
        if not head_ok(want):
            return False
        wp[sid] = want
    else:
        wp[sid] = max(0, min(4095, int(wp[sid]) + delta))
    return True


def spread(path, at, sid, delta, forward_only, log):
    """Push one waypoint's correction across the path, fading at the far end.

    Ending a correction abruptly puts a step into the path - the very thing
    the player's smoothing exists to remove - so the last FADE waypoints get
    a diminishing share of it instead of the full amount.
    """
    lo = at if forward_only else 0
    hi = len(path)
    n = 0
    for i in range(lo, hi):
        if forward_only:
            far = hi - 1 - i                       # fade towards the end
        else:
            far = min(i, hi - 1 - i)               # fade at both ends
        scale = min(1.0, (far + 1) / float(FADE)) if FADE else 1.0
        d = int(round(delta * scale))
        if d and i != at and nudge(path[i], sid, d):
            n += 1
    log(f'applied {delta:+d} to {n} waypoint(s)'
        f'{" from here on" if forward_only else " across the path"}, faded')


def main():
    try:
        arm = TrimmerArm()
        if not arm.read_pose():
            raise RuntimeError('the servos did not reply')
        arm.clamp_enabled = False
        arm.enable()
        arm.set_profile(PLAY_SPEED, PLAY_ACC)
        pathplay.set_rate(PLAY_SPEED)
    except Exception as e:
        print(f'could not talk to the arm: {e}')
        return

    paths = load()
    original = {k: [dict(w) for w in v] for k, v in paths.items()}
    if not sum(len(v) for v in paths.values()):
        print('nothing taught yet - run  python trimmer/teach_box.py  first')
        arm.close()
        return

    which = next((i for i, k in enumerate(PATHS) if paths[k]), 0)
    idx, sel, jog = 0, 0, 2.0
    torque = True
    kb = Keys()
    msg = ''
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def log(t):
        nonlocal msg
        msg = t
        print(f'  {t}')

    def stop():
        return kb.pump().hit(ord('e'))

    def send():
        """Put the arm on the selected waypoint."""
        path = paths[PATHS[which]]
        if path:
            goto(arm, path[idx], log=log, stop_check=stop)

    print('\nSelect a waypoint, the arm goes there, then jog to fix it.')
    print('w = apply that fix to the whole path.  s = save.\n')
    send()
    try:
        while True:
            name = PATHS[which]
            path = paths[name]
            idx = max(0, min(idx, max(0, len(path) - 1)))
            kb.pump()
            try:
                pose = arm.read_pose()
            except TrimmerError:
                pose = {}
            kb.pump()

            wp = path[idx] if path else {}
            orig = (original[name][idx] if idx < len(original[name]) else {})
            disp = np.zeros((360, 940, 3), np.uint8)
            jl = '  '.join(f"{'[' if i == sel else ' '}{i+1}:{n[:4]}"
                           f"{']' if i == sel else ' '}"
                           for i, (n, _) in enumerate(JOINTS))
            rows = [(f'{name.upper()}   waypoint {idx+1} of {len(path)}',
                     (0, 255, 255)),
                    (jl + f'      jog {jog:.0f} deg      '
                     f'torque {"ON" if torque else "OFF"}', (90, 255, 90))]
            if wp:
                rows.append(('taught  ' + '  '.join(
                    f'{n[:4]}:{wp.get(sid,"-")}' for n, sid in JOINTS
                    if sid != tc.HEAD_ID)
                    + ('   HEAD ' + head_label(wp[tc.HEAD_ID])
                       if tc.HEAD_ID in wp else ''), (255, 220, 90)))
                d = {sid: int(wp[sid]) - int(orig[sid])
                     for _n, sid in JOINTS if sid in wp and sid in orig}
                changed = '  '.join(f'{n[:4]}{d[sid]:+d}' for n, sid in JOINTS
                                    if d.get(sid))
                rows.append(('edited  ' + (changed or 'unchanged'),
                             (120, 200, 255) if changed else (120, 120, 120)))
            if pose:
                rows.append(('arm now ' + '  '.join(
                    f'{n[:4]}:{pose.get(sid,"-")}' for n, sid in JOINTS
                    if sid != tc.HEAD_ID), (150, 150, 150)))
            rows.append((', . waypoint   < > ten   a/d jog   w whole   f fwd   '
                         'r revert   p play   s save   q quit', (200, 200, 200)))
            if msg:
                rows.append((msg, (255, 220, 90)))
            for i, (t_, c_) in enumerate(rows):
                cv2.putText(disp, t_, (12, 28 + i * 30), FONT, 0.5, c_, 1,
                            cv2.LINE_AA)
            # a bar showing where in the path you are
            if path:
                w = disp.shape[1] - 24
                cv2.rectangle(disp, (12, 300), (12 + w, 316), (45, 45, 45), -1)
                x = 12 + int(w * idx / max(1, len(path) - 1))
                for i, p_ in enumerate(path):
                    o = original[name][i] if i < len(original[name]) else p_
                    if any(p_.get(k) != o.get(k) for k in p_):
                        px = 12 + int(w * i / max(1, len(path) - 1))
                        cv2.line(disp, (px, 300), (px, 316), (120, 200, 255), 1)
                cv2.rectangle(disp, (x - 2, 296), (x + 2, 320), (0, 255, 255), -1)
            cv2.imshow(WIN, disp)

            kb.pump(1)
            if kb.hit(ord('q')):
                break
            if kb.hit(9):
                which = (which + 1) % len(PATHS)
                idx = 0
                log(f'{PATHS[which]} - {len(paths[PATHS[which]])} waypoints')
                send()
            moved = 0
            if kb.tap(ord(',')):
                moved = -1
            elif kb.tap(ord('.')):
                moved = +1
            elif kb.tap(ord('<')):
                moved = -10
            elif kb.tap(ord('>')):
                moved = +10
            if moved and path:
                idx = max(0, min(len(path) - 1, idx + moved))
                send()
            if kb.hit(ord('h')):
                send()
            if kb.hit(ord('t')):
                torque = not torque
                try:
                    if torque:
                        arm.enable()
                        arm.set_profile(PLAY_SPEED, PLAY_ACC)
                        log('torque ON')
                    else:
                        arm.torque_off()
                        log('torque OFF - SUPPORT THE ARM')
                except TrimmerError as e:
                    log(f'torque change failed: {e}')
            for i, code in enumerate((ord('1'), ord('2'), ord('3'),
                                      ord('4'), ord('5'))):
                if kb.hit(code):
                    sel = i
            if kb.hit(ord('b')):
                jog = min(15.0, jog + 1)
            if kb.hit(ord('n')):
                jog = max(1.0, jog - 1)

            step = 0.0
            if kb.tap(ord('a')):
                step = -jog
            elif kb.tap(ord('d')):
                step = +jog
            if step and path:
                _n, sid = JOINTS[sel]
                delta = int(round(step * TICKS_PER_DEG))
                if nudge(wp, sid, delta):
                    try:
                        arm.move_to(sid, int(wp[sid]))
                    except TrimmerError as e:
                        log(f'move failed: {e}')
                    log(f'{_n} {int(wp[sid]) - int(orig.get(sid, wp[sid])):+d} '
                        f'on waypoint {idx+1}')
                else:
                    log('refused - that would put the head off its safe arc')

            if kb.hit(ord('w')) and path:      # apply to the whole path
                _n, sid = JOINTS[sel]
                d = int(wp[sid]) - int(orig.get(sid, wp[sid]))
                if d:
                    spread(path, idx, sid, d, False, log)
                else:
                    log('nothing to apply - jog this waypoint first')
            if kb.hit(ord('f')) and path:      # apply forwards from here
                _n, sid = JOINTS[sel]
                d = int(wp[sid]) - int(orig.get(sid, wp[sid]))
                if d:
                    spread(path, idx, sid, d, True, log)
                else:
                    log('nothing to apply - jog this waypoint first')
            if kb.hit(ord('r')) and path:
                path[idx] = dict(original[name][idx])
                log(f'waypoint {idx+1} reverted')
                send()
            if kb.hit(ord('z')) and path:
                paths[name] = [dict(w) for w in original[name]]
                log(f'{name} reverted')
                send()
            if kb.hit(ord('p')) and path:
                log(f'playing {name} from waypoint {idx+1}')
                play(arm, path[idx:], log, stop_check=stop, name=name)
            if kb.hit(ord('P')) and path:
                log(f'playing all of {name}')
                play(arm, path, log, stop_check=stop, name=name)
            if kb.hit(ord('s')):
                save(paths, log)
                original = {k: [dict(w) for w in v] for k, v in paths.items()}
    finally:
        try:
            if tc.REST_POSE:
                arm.set_profile(tc.MOVE_SPEED, tc.MOVE_ACC)
                pathplay.set_rate(tc.MOVE_SPEED)
                goto(arm, tc.REST_POSE)
            arm.torque_off()
        except Exception:
            pass
        arm.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
