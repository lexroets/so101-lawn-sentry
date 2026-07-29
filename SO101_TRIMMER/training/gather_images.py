"""Collect every poop-ish image you already have into one folder for labelling.

The SAM auto-labeller needs a single folder of images. This scoops up:
  - the annotated snapshots the web app saved   (scan_sessions/*/snapshots/*)
  - your labelled poop tiles                     (poop_dataset/poop/*)
  - any real photos you dropped in               (real_poop/*)
  - the fetched ScatSpotter images if present    (poop_dataset/_raw/* etc.)
...de-duplicates, and copies them to  sam_images/ .

    python gather_images.py

Then zip sam_images/ (or point Colab at it) and run sam_autolabel.py.
Aim for a few hundred+ varied real poop-on-grass shots for a good model.
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
import hashlib
import shutil
from pathlib import Path

from paths import DATA
HERE = DATA
OUT = DATA / 'sam_images'
# REAL photos only — SAM needs real images, so the SYNTHETIC poop_dataset/poop
# tiles are deliberately NOT included here.
SOURCES = [
    HERE / 'real_poop',              # capture_frames.py / your own photos
    HERE / 'poop_photos',            # ..or drop photos here
    HERE / 'roboflow',               # unzip a Roboflow dataset here
]
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    OUT.mkdir(exist_ok=True)
    seen, copied = set(), 0
    for src in SOURCES:
        if not src.exists():
            continue
        for p in src.rglob('*'):
            if p.suffix.lower() not in EXTS or not p.is_file():
                continue
            # skip our own output folder
            if OUT in p.parents:
                continue
            try:
                digest = hashlib.md5(p.read_bytes()).hexdigest()[:16]
            except OSError:
                continue
            if digest in seen:
                continue
            seen.add(digest)
            dst = OUT / f'{digest}{p.suffix.lower()}'
            if not dst.exists():
                shutil.copy2(p, dst)
                copied += 1
    total = len([p for p in OUT.iterdir() if p.suffix.lower() in EXTS])
    print(f'copied {copied} new image(s); {total} total in {OUT}')
    if total < 50:
        print('\nThat is thin for training. Add real dog-poop-on-grass photos:')
        print('  - run fetch_poop_dataset.py for the 9k ScatSpotter set, and/or')
        print('  - drop your own photos into a real_poop/ folder, then re-run.')


if __name__ == '__main__':
    main()
