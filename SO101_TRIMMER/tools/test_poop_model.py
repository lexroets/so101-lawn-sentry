"""Sanity-check the trained poop_yolo.onnx before trusting it on the robot.

Runs the model over a folder of images and writes annotated copies so you can
see exactly what it found. Works fully offline, no torch.

    python test_poop_model.py                 # uses real_poop/ (or sam_images/)
    python test_poop_model.py some/folder     # any folder of images

Writes model_test/ with boxes drawn, and prints a per-image summary plus
timing. If it boxes poop and ignores grass/leaves/kit, you're good to set
POOP_USE_YOLO = True.
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
import os
import sys
import glob
import time

import cv2

from poop_yolo import PoopYolo
import poop_config as cfg

from paths import DATA, OUTPUT
HERE = str(DATA)
OUT = str(OUTPUT / 'model_test')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def pick_folder(argv):
    if len(argv) > 1:
        return argv[1]
    for cand in ('real_poop', 'sam_images', 'poop_photos'):
        p = os.path.join(HERE, cand)
        if os.path.isdir(p) and any(f.lower().endswith(EXTS) for f in os.listdir(p)):
            return p
    return None


def main():
    folder = pick_folder(sys.argv)
    if not folder or not os.path.isdir(folder):
        print("No image folder found. Capture some with  python capture_frames.py  "
              "or pass a folder:  python test_poop_model.py path/to/images")
        return
    det = PoopYolo()
    if not det.ready:
        print(f"No {cfg.POOP_YOLO_MODEL} in this folder yet — train it first "
              "(train_poop_real.py in Colab), then drop the .onnx here.")
        return

    files = sorted(f for f in glob.glob(os.path.join(folder, '*'))
                   if f.lower().endswith(EXTS))[:60]
    if not files:
        print('no images in', folder)
        return
    os.makedirs(OUT, exist_ok=True)
    print(f"testing {len(files)} image(s) from {folder}  "
          f"(conf >= {cfg.POOP_YOLO_CONF})\n")

    hits = blanks = 0
    times = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        t0 = time.perf_counter()
        dets = det.infer(img)
        times.append((time.perf_counter() - t0) * 1000)
        if dets:
            hits += 1
        else:
            blanks += 1
        for d in dets:
            x, y, w, h = d['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (33, 224, 212), 3)
            cv2.putText(img, f"poop {d['conf']:.2f}", (x, max(16, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (33, 224, 212), 2)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f)), img)
        best = max((d['conf'] for d in dets), default=0.0)
        print(f"  {os.path.basename(f):40s} {len(dets)} det   best {best:.2f}")

    avg = sum(times) / max(1, len(times))
    print(f"\n{hits} image(s) with detections, {blanks} without")
    print(f"average inference {avg:.0f} ms  ({1000/max(avg,1e-6):.1f} fps)")
    print(f"annotated images written to: {OUT}")
    print("\nIf the boxes look right: set POOP_USE_YOLO = True in poop_config.py")
    print("Too many false boxes? raise POOP_YOLO_CONF. Missing poop? lower it.")


if __name__ == '__main__':
    main()
