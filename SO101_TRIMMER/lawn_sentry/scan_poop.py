"""Poop patrol — sweep the base side to side, scan the lawn, log what's found.

The camera-carrying arm holds a fixed look-down pose while the BASE rotates
through a set of yaw stations. At each station it settles, grabs a few frames,
detects poop (colour + depth, plus the CNN if trained), and logs each find's
estimated ground position so you know where NOT to step. Survives OAK-D
crashes (parks, restarts, re-scans the station).

    python scan_poop.py

Keys (while the preview window is focused):
    SPACE  start / pause the sweep
    m      write the running summary + map now
    e      emergency stop (torque off) and quit
    q      quit (parks, torque off, writes summary)
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

import poop_config as cfg
from trimmer_arm import TrimmerArm, TrimmerError
from trim_detector import build_pipeline
from camera_session import CameraSession
from poop_detector import PoopDetector
from poop_log import PoopLog

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WIN = 'SO-101 Poop Patrol'


def resolve_scan_pose():
    """Joints 2/3/4 the arm holds while sweeping. Prefer the taught SCAN_POSE,
    else fall back to the trimmer REST_POSE, else None (must calibrate)."""
    ids = (cfg.SHOULDER_ID, cfg.ELBOW_ID, cfg.WRIST_ID)
    if cfg.SCAN_POSE:
        return {s: cfg.SCAN_POSE[s] for s in ids if s in cfg.SCAN_POSE}
    if cfg.REST_POSE:
        return {s: cfg.REST_POSE[s] for s in ids if s in cfg.REST_POSE}
    return None


def cluster(frame_dets, min_count):
    """Merge detections across a station's frames; keep those seen >=min_count
    (kills single-frame flicker). Returns list with the best-scoring instance."""
    acc = []
    for fi, dets in enumerate(frame_dets):
        for d in dets:
            hit = None
            for a in acc:
                if (a['u'] - d['u']) ** 2 + (a['v'] - d['v']) ** 2 < 40 ** 2:
                    hit = a
                    break
            if hit is None:
                acc.append({**d, 'count': 1, 'frame': fi})
            else:
                hit['count'] += 1
                if d['score'] > hit['score']:
                    hit.update({k: d[k] for k in
                                ('u', 'v', 'z', 'area', 'score', 'bbox', 'fill')})
                    hit['frame'] = fi
                    if 'cnn' in d:
                        hit['cnn'] = d['cnn']
    return [a for a in acc if a['count'] >= min_count]


def main():
    hold = resolve_scan_pose()
    if hold is None:
        print("No scan pose yet. Run  python calibrate_scan.py  first "
              "(pose the camera to look at the lawn).")
        return

    arm = TrimmerArm()
    cam = CameraSession(build_pipeline)
    det = PoopDetector()
    log = PoopLog()
    src = 'colour+depth+CNN' if det.clf.ready else 'colour+depth (no model yet)'
    stations = [int(round(x)) for x in
                np.linspace(cfg.SCAN_BASE_MIN_RAW, cfg.SCAN_BASE_MAX_RAW,
                            cfg.SCAN_STATIONS)]
    min_count = max(1, cfg.SCAN_FRAMES_PER_STATION // 2)

    print(f"[poop] detecting with {src}")
    print(f"[poop] {cfg.SCAN_STATIONS} stations "
          f"{cfg.SCAN_BASE_MIN_RAW}..{cfg.SCAN_BASE_MAX_RAW} raw; "
          f"logging to {log.dir}")
    print("Enabling torque in 3s — hand near the power switch...")
    for n in (3, 2, 1):
        print(n); time.sleep(1)
    arm.enable()

    def pump(msg, hold_s=0.0, base_goal=None):
        """Keep the UI + liveness alive for hold_s (and while base moves).
        Returns 'crash' | 'quit' | 'estop' | None. Draws latest frame."""
        t0 = time.monotonic()
        while True:
            g = cam.get()
            if g is not None:
                img = g['rgb'].getCvFrame()
                cv2.putText(img, msg, (8, 24), _FONT, 0.55, (0, 255, 255), 2)
                cv2.putText(img, f"finds: {len(log.finds)}", (8, 48), _FONT,
                            0.55, (0, 200, 255), 2)
                cv2.imshow(_WIN, img)
            elif cam.crashed():
                return 'crash'
            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                return 'quit'
            if k == ord('e'):
                return 'estop'
            if k == ord('m'):
                print('summary ->', log.finalize())
            done_move = base_goal is None or arm.arrived({cfg.BASE_ID: base_goal},
                                                         tol=cfg.ARRIVE_TOL_RAW * 2)
            if done_move and time.monotonic() - t0 >= hold_s:
                return None

    def grab(n):
        """Pull n fresh frame groups. Returns list or 'crash'/'quit'/'estop'."""
        out = []
        while len(out) < n:
            g = cam.get()
            if g is None:
                if cam.crashed():
                    return 'crash'
                k = cv2.waitKey(1) & 255
                if k == ord('q'):
                    return 'quit'
                if k == ord('e'):
                    return 'estop'
                continue
            out.append(g)
        return out

    def park():
        arm.set_profile(cfg.MOVE_SPEED, cfg.MOVE_ACC)
        arm.move_pose(hold)
        arm.move_to(cfg.BASE_ID, cfg.BASE_FORWARD_RAW)
        arm.wait_arrived({**hold, cfg.BASE_ID: cfg.BASE_FORWARD_RAW},
                         tol=cfg.ARRIVE_TOL_RAW * 3)

    # move to the scan posture and hold it
    park()
    print("READY. SPACE = start sweeping · q = quit.")

    running = False
    loops_done = 0
    quit_reason = None
    try:
        # wait for SPACE to start
        while not running:
            r = pump("SPACE = start sweep", hold_s=0.05)
            if r == 'crash':
                park(); cam.restart(); continue
            if r in ('quit', 'estop'):
                quit_reason = r
                break
            k = cv2.waitKey(1) & 255
            if k == ord(' '):
                running = True

        while running and quit_reason is None:

            frames = grab(cfg.SCAN_FRAMES_PER_STATION)

            if frames == 'crash':
                print("[camera] crash — parking + restarting")
                park()
                cam.restart()
                continue

            if frames in ('quit', 'estop'):
                quit_reason = frames
                break

            try:
                base_raw = arm.read_position(cfg.BASE_ID)
            except TrimmerError:
                base_raw = cfg.BASE_FORWARD_RAW

            imgs = [
                g['rgb'].getCvFrame()
                for g in frames
            ]

            depths = [
                g['depth'].getFrame().astype(np.float32) / 1000.0
                for g in frames
            ]

            frame_dets = [
                det.detect(im, dp)
                for im, dp in zip(imgs, depths)
            ]

            found = cluster(frame_dets, min_count)

            disp = imgs[-1].copy()

            for d in found:

                x, y, w, h = d['bbox']

                cv2.rectangle(
                    disp,
                    (x, y),
                    (x + w, y + h),
                    (0, 90, 230),
                    2
                )

                cv2.putText(
                    disp,
                    f"{d['score']:.2f}",
                    (x, max(14, y - 6)),
                    _FONT,
                    0.5,
                    (0, 90, 230),
                    2
                )

            cv2.putText(
                disp,
                f"stationary scan  found:{len(found)}  total:{len(log.finds)}",
                (8, 24),
                _FONT,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.imshow(_WIN, disp)

            k = cv2.waitKey(1) & 255

            if k == ord('q'):
                quit_reason = 'quit'
                break

            if k == ord('e'):
                quit_reason = 'estop'
                break

            if k == ord('m'):
                print('summary ->', log.finalize())

            for d in found:

                is_new, f = log.add(
                    d,
                    base_raw,
                    image=imgs[d['frame']]
                )

                if is_new:

                    print(
                        f"  POOP  ~{f['y_m']:.2f}m ahead, "
                        f"{abs(f['x_m']):.2f}m "
                        f"{'right' if f['x_m'] >= 0 else 'left'}  "
                        f"score={d['score']:.2f}"
                    )

        if quit_reason == 'estop':
            arm.emergency_stop()
            print("E-STOP.")            
            
    finally:
        summary = log.finalize()
        log.close()
        print(f"\nScan done: {len(log.finds)} distinct detections.")
        print(f"Log + snapshots + map: {log.dir}")
        try:
            with open(summary) as f:
                print("\n" + f.read())
        except OSError:
            pass
        if quit_reason != 'estop':
            try:
                park(); time.sleep(0.3); arm.torque_off()
            except TrimmerError:
                pass
        arm.close()
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
