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
#  AUTO-LABEL poop images with SAM 3.1 (teacher) -> YOLO dataset
#  Paste this whole file into ONE Google Colab cell (Runtime = GPU) and run.
#
#  What it does: takes your folder of images (sam_images/) and, using a
#  text prompt, has a Segment-Anything model draw a box around every piece of
#  dog poop — producing YOLO-format labels with NO manual clicking. You then
#  train the small student on these labels with train_poop_yolo.py.
#
#  1. On your Mac:   python gather_images.py      (fills sam_images/)
#                    zip -r sam_images.zip sam_images
#  2. Colab -> GPU runtime -> paste this cell -> run -> upload sam_images.zip
#  3. Download poop_yolo_dataset.zip at the end.
# =====================================================================
import os, glob, zipfile
from google.colab import files

# ---- get the images --------------------------------------------------------
if not os.path.isdir('sam_images'):
    print('Upload sam_images.zip ...'); files.upload()
    with zipfile.ZipFile(glob.glob('*sam_images*.zip')[0]) as z:
        z.extractall('.')
IMG_DIR = 'sam_images' if os.path.isdir('sam_images') else '.'
print('images:', len(glob.glob(f'{IMG_DIR}/**/*.*', recursive=True)))

# ---- install autodistill + a Segment-Anything base model -------------------
# Autodistill is the standard "foundation model -> labels -> small model"
# framework. We try SAM 3.1 first, then fall back to Grounded-SAM variants so
# this runs even if the SAM-3 plugin isn't published for your Colab yet.
get_ipython().system('pip -q install autodistill autodistill-yolov8 supervision')

BaseModel = None
for pkg, mod, cls in [
    ('autodistill-sam-3', 'autodistill_sam_3', 'SAM3'),
    ('autodistill-grounded-sam-2', 'autodistill_grounded_sam_2', 'GroundedSAM2'),
    ('autodistill-grounded-sam', 'autodistill_grounded_sam', 'GroundedSAM'),
]:
    try:
        get_ipython().system('pip -q install ' + pkg)
        BaseModel = getattr(__import__(mod, fromlist=[cls]), cls)
        print('using base model:', cls); break
    except Exception as e:
        print('skip', pkg, '->', str(e)[:80])
assert BaseModel is not None, 'No Segment-Anything base model could be installed.'

from autodistill.detection import CaptionOntology

# The text prompt IS the labeller. Tune the left-hand phrase if it misses/over-
# fires; the right-hand word ("poop") is the single class name we train on.
ontology = CaptionOntology({
    "a piece of dog poop on grass": "poop",
    "animal feces on the ground": "poop",
})

base = BaseModel(ontology=ontology)
base.label(input_folder=IMG_DIR, output_folder='poop_yolo_dataset',
           extension='')          # writes YOLO images/ + labels/ + data.yaml

# ---- quick sanity: how many images got a poop box? -------------------------
labels = glob.glob('poop_yolo_dataset/**/labels/**/*.txt', recursive=True)
nonempty = [p for p in labels if os.path.getsize(p) > 0]
print(f'labelled {len(nonempty)}/{len(labels)} images with >=1 poop box')
print('IMPORTANT: skim poop_yolo_dataset for obviously wrong boxes before '
      'training — foundation labels are good but not perfect.')

# ---- package for training --------------------------------------------------
get_ipython().system('zip -q -r poop_yolo_dataset.zip poop_yolo_dataset')
files.download('poop_yolo_dataset.zip')
