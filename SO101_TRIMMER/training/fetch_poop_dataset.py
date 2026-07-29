"""Pull REAL dog-poop images and turn them into training tiles.

Run this ON YOUR MACHINE (it needs open internet; the assistant's sandbox is
network-restricted, so it can't download the images for you — this script can).

It uses the public, openly-licensed **ScatSpotter / "shitspotter"** dataset
(9000+ phone photos of dog poop with polygon annotations, CC-BY-4.0, by
Jon Crall). It downloads the images + COCO-style annotations from Hugging Face,
crops a square tile around each annotated poop into  poop_dataset/poop/ , and
samples background tiles from the same photos into  poop_dataset/other/  and
poop_dataset/grass/  (green backgrounds go to grass, the rest to other).

    pip install huggingface_hub pycocotools opencv-python numpy
    python fetch_poop_dataset.py --limit 400        # ~400 images -> lots of tiles
    python fetch_poop_dataset.py --limit 0          # everything (large!)

Then:  zip -r poop_dataset.zip poop_dataset  and train with train_poop_colab.py.

Dataset:   https://github.com/Erotemic/shitspotter   (CC-BY-4.0)
           https://huggingface.co/datasets/erotemic/shitspotter
Alternative: Roboflow Universe has ready-made poop classification/detection
sets (search "dog poop"); export as "Folder"/"Classification" straight into
poop_dataset/. See the notes at the bottom.
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
import os
import sys
import glob
import json
import argparse
import random

import numpy as np
import cv2

from paths import DATA
HERE = str(DATA)
OUT = str(DATA / 'poop_dataset')
TILE = 96                      # matches POOP_TILE_PX
HF_REPO = 'erotemic/shitspotter'


def _download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub  (then re-run)")
    print(f"downloading {HF_REPO} from Hugging Face (this can take a while)...")
    try:
        # grab only image + annotation files (skips the huge extras), single
        # worker to avoid the threaded file-lock issues seen under Thonny.
        return snapshot_download(
            repo_id=HF_REPO, repo_type='dataset', max_workers=1,
            allow_patterns=['*.jpg', '*.jpeg', '*.png', '*.json', '*.kwcoco.*'])
    except TypeError as exc:
        if 'mode' in str(exc):
            sys.exit(
                "\n>>> This failed because of a library clash inside Thonny "
                "(its bundled 'filelock' is too old for huggingface_hub).\n"
                ">>> FIX: run this script from the macOS Terminal instead, NOT "
                "Thonny:\n"
                "      cd ~/so101_autonomy_architecture_disabled/SO101_TRIMMER\n"
                "      python3 fetch_poop_dataset.py\n"
                ">>> If it still fails there:  pip3 install -U huggingface_hub "
                "filelock\n"
                ">>> Or skip this entirely — see the note the assistant gave you "
                "(Roboflow / your own camera images work just as well).\n")
        raise


def _find_coco(root):
    """Locate a COCO/kwcoco json (has 'images' + 'annotations')."""
    cands = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
    cands += glob.glob(os.path.join(root, '**', '*.kwcoco.json'), recursive=True)
    best = None
    for p in sorted(set(cands)):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if isinstance(d, dict) and 'images' in d and 'annotations' in d:
            n = len(d['annotations'])
            if best is None or n > best[1]:
                best = (p, n, d)
    return best


def _bbox_from_ann(a):
    if a.get('bbox'):
        x, y, w, h = a['bbox']
        return int(x), int(y), int(w), int(h)
    seg = a.get('segmentation')
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        pts = np.array(seg[0], np.float32).reshape(-1, 2)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
    return None


def _square(img, cx, cy, side):
    h, w = img.shape[:2]
    half = side // 2
    x0 = max(0, min(w - side, cx - half)); y0 = max(0, min(h - side, cy - half))
    crop = img[y0:y0 + side, x0:x0 + side]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)


def _is_green(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return float(((h >= 32) & (h <= 95) & (s >= 40)).mean()) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=300,
                    help='max images to process (0 = all)')
    ap.add_argument('--neg-per-img', type=int, default=3,
                    help='background tiles sampled per image')
    args = ap.parse_args()
    for c in ('poop', 'grass', 'other'):
        os.makedirs(os.path.join(OUT, c), exist_ok=True)

    root = _download()
    found = _find_coco(root)
    if not found:
        sys.exit("No COCO/kwcoco annotation json found in the download. Open the "
                 "dataset folder and check its layout, or use the Roboflow route "
                 "described at the top of this file.")
    coco_path, nann, d = found
    print(f"annotations: {coco_path}  ({nann} polygons)")
    base = os.path.dirname(coco_path)
    id2img = {im['id']: im for im in d['images']}
    by_img = {}
    for a in d['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)

    img_ids = list(by_img.keys())
    random.seed(0); random.shuffle(img_ids)
    if args.limit:
        img_ids = img_ids[:args.limit]

    np_, ng_, no_ = 0, 0, 0
    for k, iid in enumerate(img_ids):
        im = id2img.get(iid)
        if not im:
            continue
        fn = im.get('file_name') or im.get('name')
        path = fn if os.path.isabs(fn) else os.path.join(base, fn)
        if not os.path.isfile(path):
            hits = glob.glob(os.path.join(base, '**', os.path.basename(fn)),
                             recursive=True)
            if not hits:
                continue
            path = hits[0]
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for a in by_img[iid]:
            bb = _bbox_from_ann(a)
            if not bb:
                continue
            x, y, w, h = bb
            side = int(max(w, h) * 1.3)
            t = _square(img, x + w // 2, y + h // 2, max(24, side))
            if t is not None:
                cv2.imwrite(os.path.join(OUT, 'poop', f'sp_{iid}_{np_}.png'), t)
                np_ += 1
                boxes.append((x, y, w, h))
        # negatives: random squares that don't overlap a poop box
        for _ in range(args.neg_per_img):
            side = random.randint(TILE, max(TILE + 1, min(H, W) // 3))
            cx = random.randint(side, max(side + 1, W - side))
            cy = random.randint(side, max(side + 1, H - side))
            if any(abs(cx - (x + w // 2)) < (w + side) // 2 and
                   abs(cy - (y + h // 2)) < (h + side) // 2 for x, y, w, h in boxes):
                continue
            t = _square(img, cx, cy, side)
            if t is None:
                continue
            if _is_green(t):
                cv2.imwrite(os.path.join(OUT, 'grass', f'sp_{iid}_{ng_}.png'), t)
                ng_ += 1
            else:
                cv2.imwrite(os.path.join(OUT, 'other', f'sp_{iid}_{no_}.png'), t)
                no_ += 1
        if k % 25 == 0:
            print(f"  {k}/{len(img_ids)} imgs  poop={np_} grass={ng_} other={no_}")

    print(f"\nDONE. poop={np_}  grass={ng_}  other={no_} in {OUT}")
    print("Mix in the synthetic tiles if you like, then:")
    print("  zip -r poop_dataset.zip poop_dataset   # and train_poop_colab.py")


# ---------------------------------------------------------------------------
# ROBOFLOW ALTERNATIVE (ready-made, often cleaner for a first model):
#   pip install roboflow
#   from roboflow import Roboflow
#   rf = Roboflow(api_key="YOUR_KEY")
#   # pick a project from https://universe.roboflow.com (search "dog poop")
#   ds = rf.workspace("...").project("...").version(N).download("folder")
#   # then move ds/train/<class>/* into poop_dataset/<poop|grass|other>/
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
