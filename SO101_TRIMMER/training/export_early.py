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
# =====================================================================
#  STOP TRAINING EARLY AND KEEP THE MODEL
#
#  YOLO writes best.pt after every epoch, so an interrupted run is NOT lost.
#
#  How to use, in Colab:
#    1. Press the ■ stop button on the training cell (or just let it stop).
#    2. Paste this whole file into a NEW cell and run it.
#    3. It grabs the best checkpoint so far, scores it, exports ONNX,
#       and downloads poop_yolo.onnx.
#
#  Rule of thumb for the mAP50 it prints:
#      < 0.30   too weak — keep training
#      0.30-0.50  usable for a demo, expect misses
#      0.50-0.70  decent
#      > 0.70   good
# =====================================================================
import glob, os, shutil

IMG = 640          # must match what you trained with

ckpts = sorted(glob.glob('runs/detect/**/weights/best.pt', recursive=True),
               key=os.path.getmtime)
assert ckpts, 'No checkpoint found — did training start? Look for runs/detect/'
best = ckpts[-1]
print('using checkpoint:', best,
      f'({os.path.getsize(best)/1e6:.1f} MB, '
      f'modified {os.path.getmtime(best)})')

from ultralytics import YOLO
model = YOLO(best)

# score it so you know what you are shipping
try:
    m = model.val()
    print(f'\nmAP50 = {float(m.box.map50):.3f}   mAP50-95 = {float(m.box.map):.3f}')
    p = float(m.box.mp); r = float(m.box.mr)
    print(f'precision = {p:.3f}  recall = {r:.3f}')
    if float(m.box.map50) < 0.30:
        print('\n-> Weak. Usable to test the plumbing, but train longer for real use.')
    elif float(m.box.map50) < 0.50:
        print('\n-> Demo-grade. Will find obvious poop, will miss some.')
    else:
        print('\n-> Good enough to deploy.')
except Exception as e:
    print('(validation skipped:', str(e)[:80], ')')

path = model.export(format='onnx', imgsz=IMG, opset=12, simplify=True, nms=False)
shutil.move(path, 'poop_yolo.onnx')
print('\nexported poop_yolo.onnx')
print(f'set POOP_YOLO_IMG = {IMG} in poop_config.py (it already is by default)')

try:
    from google.colab import files
    files.download('poop_yolo.onnx')
except Exception:
    print('download poop_yolo.onnx from the Files panel on the left.')
