# --- make the project's flat imports work from any subfolder (see paths.py) ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve().parent
_R = _R if (_R / 'paths.py').is_file() else _R.parent
for _p in ('', 'core', 'calibrate', 'run', 'dashboard', 'training', 'tools'):
    _d = str(_R / _p) if _p else str(_R)
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# -----------------------------------------------------------------------------
# =====================================================================
#  TRAIN the poop student (YOLOv8-nano) on the SAM auto-labels -> ONNX
#  Paste this whole file into ONE Google Colab cell (Runtime = GPU) and run.
#
#  1. Run sam_autolabel.py first -> download poop_yolo_dataset.zip
#  2. Colab -> GPU runtime -> paste this cell -> run -> upload the zip
#  3. Download poop_yolo.onnx, drop it into SO101_TRIMMER/, and set
#     POOP_USE_YOLO = True in poop_config.py. Verify with: python poop_yolo.py
# =====================================================================
import os, glob, zipfile, yaml
from google.colab import files

# ---- get the labelled dataset ---------------------------------------------
if not os.path.isdir('poop_yolo_dataset'):
    print('Upload poop_yolo_dataset.zip ...'); files.upload()
    with zipfile.ZipFile(glob.glob('*poop_yolo_dataset*.zip')[0]) as z:
        z.extractall('.')

# find the data.yaml autodistill wrote (path can vary slightly)
dy = glob.glob('poop_yolo_dataset/**/data.yaml', recursive=True) \
    or glob.glob('**/data.yaml', recursive=True)
assert dy, 'no data.yaml found in the dataset zip'
DATA = os.path.abspath(dy[0])
cfg = yaml.safe_load(open(DATA))
print('dataset:', DATA, '| classes:', cfg.get('names'))

# ---- train -----------------------------------------------------------------
get_ipython().system('pip -q install ultralytics')
from ultralytics import YOLO

IMG = 640
model = YOLO('yolov8n.pt')          # nano = tiny + fast on the edge
model.train(data=DATA, imgsz=IMG, epochs=80, batch=16, patience=20,
            degrees=10, fliplr=0.5, flipud=0.5, hsv_v=0.4, mosaic=1.0,
            name='poop_yolo', seed=0)

# ---- validate + export -----------------------------------------------------
metrics = model.val()
print('mAP50:', round(float(metrics.box.map50), 3),
      '| mAP50-95:', round(float(metrics.box.map), 3))

# export to a single-file ONNX the robot can run with onnxruntime/cv2.dnn
path = model.export(format='onnx', imgsz=IMG, opset=12, simplify=True, nms=False)
os.rename(path, 'poop_yolo.onnx')
print('exported poop_yolo.onnx  (imgsz=%d)  -> set POOP_YOLO_IMG=%d' % (IMG, IMG))
files.download('poop_yolo.onnx')
