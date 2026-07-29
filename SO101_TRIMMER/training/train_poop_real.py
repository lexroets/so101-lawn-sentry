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
#  POOP DETECTOR — REAL DATA TRAINING  (one Colab cell, end to end)
#
#  Paste this whole file into ONE Google Colab cell.
#  Runtime -> Change runtime type -> GPU (T4 is fine).  Then run.
#
#  What it does
#  ------------
#  1. Streams real dog-poop photos + HUMAN polygon labels from the public
#     ScatSpotter dataset (CC-BY-4.0). No SAM needed for these — they are
#     already expertly labelled, which beats pseudo-labels.
#     Crucially it also contains the exact distractors that cause your false
#     positives (leaf, stick, mulch, rock, dirt, shadow, acorn...). Those are
#     kept as UNLABELLED background, which actively teaches the model
#     "brown thing != poop".
#  2. OPTIONAL: if you upload your own OAK-D frames (sam_images.zip), SAM 3.1
#     auto-labels them so the model also learns YOUR lawn, camera and light.
#  3. Trains YOLOv8-nano, reports mAP, exports poop_yolo.onnx.
#
#  Afterwards on the Mac: drop poop_yolo.onnx into SO101_TRIMMER/,
#  set POOP_USE_YOLO = True in poop_config.py, run  python poop_yolo.py.
# =====================================================================

# ---- knobs ----------------------------------------------------------------
N_REAL      = 1200   # how many ScatSpotter images to pull (1200 ~= 1.2 GB, ~8 min)
EPOCHS      = 120    # v1 early-stopped at 37; more headroom once you add data
PATIENCE    = 30     # was 15 — stopped too eagerly on a small dataset
IMG         = 960    # 640 -> 960 is the single biggest win for DISTANT poop:
                     # a 18 px target becomes ~27 px to the model. Costs ~2x
                     # training time. Drop back to 640 if Colab runs out of GPU
                     # memory (and set POOP_YOLO_IMG to match in poop_config.py!)
USE_MY_OWN  = True   # upload your own labelled frames (mydata.zip)?
# ---------------------------------------------------------------------------

import os, io, json, glob, tarfile, zipfile, random, shutil, urllib.request

# ---- GPU CHECK — do this BEFORE spending 8 minutes downloading -------------
# On CPU this job takes ~12 hours; on a free T4 it takes ~40 minutes.
_gpu = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null').read().strip()
if _gpu:
    print(f'GPU: {_gpu}  — good, expect ~30 s/epoch\n')
else:
    print('\n' + '!' * 70)
    print('NO GPU DETECTED — training would take ~12 HOURS instead of ~40 min.')
    print('Fix: Runtime -> Change runtime type -> Hardware accelerator: T4 GPU')
    print('     -> Save, then re-run this cell.')
    print('!' * 70 + '\n')
    raise SystemExit('halted: switch to a GPU runtime first (see above)')

os.makedirs('ds/images/train', exist_ok=True)
os.makedirs('ds/labels/train', exist_ok=True)
os.makedirs('ds/images/val',   exist_ok=True)
os.makedirs('ds/labels/val',   exist_ok=True)

REPO  = 'erotemic/shitspotter'
BASE  = f'https://huggingface.co/datasets/{REPO}/resolve/main/webdataset_shards'
SHARDS = [f'{BASE}/train/train-{i:06d}.tar' for i in range(9)]

# ---------------------------------------------------------------------------
# 1. stream real labelled images out of the webdataset shards
# ---------------------------------------------------------------------------
from PIL import Image

def poly_to_boxes(ann_obj, W, H):
    """Pull POOP boxes (normalised xywh) out of one sample's annotation blob.
    Handles the shapes kwcoco/webdataset commonly emit; anything that is not
    poop is deliberately ignored so it trains as background."""
    boxes = []
    anns = None
    if isinstance(ann_obj, dict):
        for k in ('annotations', 'anns', 'objects', 'shapes'):
            if isinstance(ann_obj.get(k), list):
                anns = ann_obj[k]; break
        if anns is None and 'bbox' in ann_obj:
            anns = [ann_obj]
    elif isinstance(ann_obj, list):
        anns = ann_obj
    if not anns:
        return boxes

    for a in anns:
        if not isinstance(a, dict):
            continue
        # --- is this a poop? (name, or category id 1 per categories.json) ---
        name = str(a.get('category_name') or a.get('category') or
                   a.get('name') or a.get('label') or '').lower()
        cid = a.get('category_id', a.get('class_id'))
        is_poop = ('poop' in name) if name else (cid in (1, '1'))
        if not is_poop:
            continue
        # --- geometry: bbox, else polygon/segmentation bounds ---
        x = y = w = h = None
        bb = a.get('bbox')
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            x, y, w, h = [float(v) for v in bb]
        else:
            seg = a.get('segmentation') or a.get('polygon') or a.get('points')
            pts = []
            if isinstance(seg, dict):
                for key in ('exterior', 'points', 'coords'):
                    if isinstance(seg.get(key), list):
                        seg = seg[key]; break
            if isinstance(seg, list) and seg:
                first = seg[0]
                if isinstance(first, (int, float)):
                    # already a flat [x1,y1,x2,y2,...] list
                    pts = [(float(seg[i]), float(seg[i+1]))
                           for i in range(0, len(seg) - 1, 2)]
                elif (isinstance(first, (list, tuple)) and len(first) == 2
                      and all(isinstance(v, (int, float)) for v in first)):
                    # a list of [x, y] points (kwcoco 'exterior' style)
                    pts = [(float(p[0]), float(p[1])) for p in seg
                           if isinstance(p, (list, tuple)) and len(p) >= 2]
                elif isinstance(first, (list, tuple)):
                    # COCO nesting: [[x1,y1,x2,y2,...]]
                    flat = first
                    if flat and isinstance(flat[0], (int, float)):
                        pts = [(float(flat[i]), float(flat[i+1]))
                               for i in range(0, len(flat) - 1, 2)]
                    elif flat:
                        pts = [(float(p[0]), float(p[1])) for p in flat
                               if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(pts) >= 3:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x, y = min(xs), min(ys); w, h = max(xs) - x, max(ys) - y
        if x is None or w is None or w <= 1 or h <= 1:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        if 0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1:
            boxes.append((cx, cy, nw, nh))
    return boxes


def harvest(n_target):
    """Stream shards, writing image+label pairs until we have n_target images."""
    kept = withbox = 0
    printed = False
    for url in SHARDS:
        if kept >= n_target:
            break
        print(f'  streaming {url.rsplit("/",1)[-1]} ...')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'colab'})
            resp = urllib.request.urlopen(req)
            tf = tarfile.open(fileobj=resp, mode='r|')
        except Exception as e:
            print('   shard unavailable:', str(e)[:90]); continue
        sample, key = {}, None
        try:
            for m in tf:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(m.name)
                ext = ext.lower().lstrip('.')
                if key is not None and stem != key:
                    kept, withbox = flush(sample, kept, withbox, n_target)
                    sample = {}
                    if kept >= n_target:
                        break
                key = stem
                data = tf.extractfile(m).read()
                if not printed:
                    print('   sample member example:', m.name); printed = True
                    if ext in ('json',):
                        # show the real annotation schema once, so a parser
                        # mismatch is obvious instead of silent
                        try:
                            probe = json.loads(data.decode('utf-8', 'ignore'))
                            print('   annotation schema:',
                                  json.dumps(probe, default=str)[:600])
                        except Exception:
                            pass
                sample[ext] = data
            # flush the final sample of the shard (no following key to trigger it)
            kept, withbox = flush(sample, kept, withbox, n_target)
        except Exception as e:
            print('   stream ended:', str(e)[:90])
        finally:
            try: tf.close(); resp.close()
            except Exception: pass
    return kept, withbox


def flush(sample, kept, withbox, n_target):
    """Write one webdataset sample as a YOLO image+label pair."""
    if kept >= n_target or not sample:
        return kept, withbox
    img_b = None
    for e in ('jpg', 'jpeg', 'png', 'webp'):
        if e in sample:
            img_b = sample[e]; break
    if img_b is None:
        return kept, withbox
    ann = None
    for e in ('json', 'ann.json', 'annotations.json', 'cls', 'txt'):
        if e in sample:
            try: ann = json.loads(sample[e].decode('utf-8', 'ignore'))
            except Exception: ann = None
            break
    try:
        im = Image.open(io.BytesIO(img_b)).convert('RGB')
    except Exception:
        return kept, withbox
    W0, H0 = im.size
    im.thumbnail((1280, 1280))                    # shrink: plenty for 640 training
    sx, sy = im.size[0] / W0, im.size[1] / H0
    boxes = poly_to_boxes(ann, W0, H0) if ann is not None else []
    split = 'val' if kept % 8 == 0 else 'train'   # ~12% validation
    stem = f'ss_{kept:06d}'
    im.save(f'ds/images/{split}/{stem}.jpg', quality=88)
    with open(f'ds/labels/{split}/{stem}.txt', 'w') as f:
        for (cx, cy, w, h) in boxes:
            f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
    return kept + 1, withbox + (1 if boxes else 0)


print(f'STEP 1 — pulling {N_REAL} real labelled images from ScatSpotter...')
kept, withbox = harvest(N_REAL)
print(f'  got {kept} images; {withbox} contain >=1 poop box '
      f'({kept - withbox} are pure background/distractor images)')
assert kept > 50, 'Too few images streamed — check the network / shard URLs.'

# ---- HARD SAFETY CHECK -----------------------------------------------------
# If almost nothing got a box, the annotation parser did not understand this
# dataset's schema. Training now would produce a model that detects NOTHING.
# Stop loudly and print the raw schema so the parser can be corrected.
frac = withbox / max(1, kept)
print(f'  -> {frac*100:.0f}% of images have a poop box')
if frac < 0.15:
    print('\n' + '!' * 68)
    print('STOP: almost no poop boxes were parsed out of the real annotations.')
    print('The images are real, but the LABELS were not understood, so training')
    print('now would produce a useless model.')
    print('Look at the "annotation schema:" line printed above and send it to')
    print('the assistant — the parser (poly_to_boxes) needs one small fix.')
    print('!' * 68 + '\n')
    raise SystemExit('halted: label parsing failed — see message above')
if frac < 0.4:
    print('  NOTE: a low share of images have boxes. ScatSpotter includes '
          '"after"/negative shots by design, so this can be normal — but check '
          'the label preview below carefully.')

# ---------------------------------------------------------------------------
# 2. OPTIONAL — SAM 3.1 auto-labels YOUR OWN OAK-D frames (domain adaptation)
# ---------------------------------------------------------------------------
if USE_MY_OWN:
    print('\nSTEP 2 — YOUR OWN labelled frames (this is what fixes '
          '"only sees big foreground poop").')
    print('  Upload mydata.zip  (made by capture_dataset.py + label_frames.py),')
    print('  or press Cancel to train on the public data only.')
    try:
        from google.colab import files
        up = files.upload()
    except Exception:
        up = {}
    zips = [f for f in up if f.endswith('.zip')]
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall('mine')
        # find image/label pairs anywhere in the zip
        imgs = [p for p in glob.glob('mine/**/images/**/*.*', recursive=True)
                if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:      # tolerate a flat folder of images
            imgs = [p for p in glob.glob('mine/**/*.*', recursive=True)
                    if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        pairs, unlabelled = [], 0
        for ip in imgs:
            lp = ip.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if os.path.isfile(lp):
                pairs.append((ip, lp))
            else:
                unlabelled += 1
        print(f'  {len(pairs)} labelled image(s) found'
              + (f'  ({unlabelled} unlabelled — run label_frames.py on those)'
                 if unlabelled else ''))

        if pairs:
            # OVERSAMPLE: your frames are few but far more relevant than the
            # public photos, so repeat them until they are a meaningful share
            # of the training set. Validation copies are NOT repeated.
            n_pub = len(glob.glob('ds/images/train/*.jpg'))
            reps = max(1, min(8, round(n_pub * 0.5 / max(1, len(pairs) * 0.875))))
            print(f'  oversampling your data x{reps} so it carries real weight '
                  f'against {n_pub} public images')
            n = boxes = 0
            for k, (ip, lp) in enumerate(pairs):
                nb = len([l for l in open(lp).read().splitlines() if l.strip()])
                boxes += nb
                if k % 8 == 0:                       # ~12% held out, once only
                    shutil.copy(ip, f'ds/images/val/mine_{k:05d}.jpg')
                    shutil.copy(lp, f'ds/labels/val/mine_{k:05d}.txt')
                else:
                    for r in range(reps):
                        shutil.copy(ip, f'ds/images/train/mine_{k:05d}_{r}.jpg')
                        shutil.copy(lp, f'ds/labels/train/mine_{k:05d}_{r}.txt')
                n += 1
            print(f'  added {n} of your frames ({boxes} boxes, '
                  f'{sum(1 for _, lp in pairs if os.path.getsize(lp) == 0)} negatives)')
    else:
        print('  skipped — training on public data only. Expect it to keep '
              'missing small/distant targets in YOUR camera view.')

# ---------------------------------------------------------------------------
# 2b. EYEBALL THE LABELS before spending 40 minutes training on them
# ---------------------------------------------------------------------------
try:
    from PIL import ImageDraw
    import matplotlib.pyplot as plt
    lbls = [p for p in glob.glob('ds/labels/train/*.txt') if os.path.getsize(p) > 0]
    random.seed(0); random.shuffle(lbls)
    show = lbls[:8]
    if show:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, lp in zip(axes.ravel(), show):
            ip = lp.replace('/labels/', '/images/').replace('.txt', '.jpg')
            im = Image.open(ip).convert('RGB'); d = ImageDraw.Draw(im)
            w, h = im.size
            for line in open(lp):
                _, cx, cy, bw, bh = [float(v) for v in line.split()]
                d.rectangle([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
                            outline=(212, 224, 33), width=4)
            ax.imshow(im); ax.axis('off'); ax.set_title(os.path.basename(ip), fontsize=8)
        for ax in axes.ravel()[len(show):]:
            ax.axis('off')
        plt.suptitle('LABEL CHECK — boxes must sit on actual poop', fontsize=13)
        plt.tight_layout(); plt.show()
        print('If these boxes look wrong, stop and tell the assistant before training.')
except Exception as e:
    print('(label preview skipped:', str(e)[:70], ')')

# ---------------------------------------------------------------------------
# 3. train
# ---------------------------------------------------------------------------
with open('ds/data.yaml', 'w') as f:
    f.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: poop\n'
            % os.path.abspath('ds'))
ntr = len(glob.glob('ds/images/train/*.jpg'))
nva = len(glob.glob('ds/images/val/*.jpg'))
print(f'\nSTEP 3 — training on {ntr} train / {nva} val images')

os.system('pip -q install ultralytics')
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data=os.path.abspath('ds/data.yaml'), imgsz=IMG, epochs=EPOCHS,
            batch=16, patience=PATIENCE, seed=0, name='poop',
            degrees=12, fliplr=0.5, flipud=0.3, hsv_h=0.02, hsv_s=0.5,
            hsv_v=0.4, mosaic=1.0, close_mosaic=10,
            # --- small / distant object tuning ---
            # scale=0.9 lets training see targets shrunk to ~0.1x, which
            # manufactures the tiny distant poop the public photos lack.
            scale=0.9,
            # mild translation so targets appear away from frame centre
            translate=0.2)

m = model.val()
print(f'\nRESULT  mAP50={float(m.box.map50):.3f}  mAP50-95={float(m.box.map):.3f}')
print('  (>0.70 mAP50 is a good, usable detector)')

# ---------------------------------------------------------------------------
# 4. export for the robot
# ---------------------------------------------------------------------------
p = model.export(format='onnx', imgsz=IMG, opset=12, simplify=True, nms=False)
shutil.move(p, 'poop_yolo.onnx')
print('\nexported poop_yolo.onnx  — set POOP_YOLO_IMG = %d in poop_config.py' % IMG)
try:
    from google.colab import files as _f
    _f.download('poop_yolo.onnx')
except Exception:
    print('download poop_yolo.onnx from the Files panel on the left.')
