# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import cv2
import numpy as np
import poop_config as cfg

from trimmer_arm import TrimmerArm
from camera_session import CameraSession
from trim_detector import build_pipeline

from poop_detector import PoopDetector
from poop_log import PoopLog


def main():

    arm = TrimmerArm()

    pose = cfg.SCAN_POSE

    if pose is None:
        print("Run live_calibrate_poop.py first")
        return

    arm.enable()

    arm.move_pose(pose)

    cam = CameraSession(build_pipeline)

    det = PoopDetector()

    log = PoopLog()

    BASE_YAW = cfg.BASE_FORWARD_RAW

    try:

        while True:

            g = cam.get()

            if g is None:

                if cam.crashed():
                    print("camera restart")
                    cam.restart()

                continue

            img = g["rgb"].getCvFrame()

            depth = (
                g["depth"]
                .getFrame()
                .astype(np.float32)
                / 1000.0
            )

            detections = det.detect(
                img,
                depth
            )

            disp = img.copy()

            for d in detections:

                x,y,w,h = d["bbox"]

                cv2.rectangle(
                    disp,
                    (x,y),
                    (x+w, y+h),
                    (0,90,230),
                    2
                )

                new_find, f = log.add(
                    d,
                    BASE_YAW,
                    image=img
                )

                if new_find:

                    print(
                        f"POOP "
                        f"{f['y_m']:.2f}m ahead "
                        f"score={d['score']:.2f}"
                    )

            cv2.imshow(
                "Stationary Poop Patrol",
                disp
            )

            if cv2.waitKey(1) & 255 == ord("q"):
                break

    finally:

        summary = log.finalize()

        print("summary:", summary)

        log.close()

        arm.torque_off()

        arm.close()

        cam.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()