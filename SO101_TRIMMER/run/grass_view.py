"""Live preview of the grass detector — camera only, no arm.

Shows the OAK feed with detected UNCUT-grass regions boxed and scored, so you
can check detection and tune thresholds before wiring it into the trimmer.
Works with GLCM + depth alone; sharper once grass_cls.onnx is trained.

Keys:  [ / ]  lower / raise the uncut threshold live · q quit
"""

# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
import time
import cv2
import numpy as np

import trimmer_config as cfg
from trim_detector import build_pipeline
from grass_detector import GrassDetector
from camera_session import CameraSession

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cam = CameraSession(build_pipeline)
    det = GrassDetector()
    src = 'CNN+GLCM+depth' if det.clf.ready else 'GLCM+depth (no model yet)'
    print(f"[grass] detecting with: {src}")
    window = 'SO-101 grass detector preview'
    try:
        while True:
            group = cam.get()
            if group is None:
                if cam.crashed():
                    print("[camera] stopped — restarting")
                    if not cam.restart():
                        break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            image = group['rgb'].getCvFrame()
            depth = group['depth'].getFrame().astype(np.float32) / 1000.0
            stats = {}
            patches = det.detect(image, depth, stats)
            for i, p in enumerate(patches):
                x, y, w, h = p['bbox']
                col = (80, 220, 100) if i == 0 else (0, 190, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), col, 3 if i == 0 else 2)
                cv2.putText(image, f"uncut {p['score']:.2f} z={p['z']:.2f}",
                            (x, max(14, y - 6)), _FONT, 0.5, col, 2)
            cv2.rectangle(image, (0, 0), (image.shape[1], 26), (18, 18, 18), -1)
            cv2.putText(image, f"{src}   thr={cfg.GRASS_UNCUT_CONF:.2f}   "
                        f"uncut_tiles={stats.get('uncut_tiles', 0)}   "
                        f"regions={stats.get('regions', 0)}",
                        (8, 19), _FONT, 0.5, (0, 255, 255), 1)
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key == ord('['):
                cfg.GRASS_UNCUT_CONF = max(0.2, cfg.GRASS_UNCUT_CONF - 0.05)
            if key == ord(']'):
                cfg.GRASS_UNCUT_CONF = min(0.95, cfg.GRASS_UNCUT_CONF + 0.05)
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
