"""Fast box labeller for your own frames — drag a box round each poop.

SAM is great on big clear objects but unreliable on the tiny distant targets
that are exactly what your model is missing. Hand-drawing ~150 boxes takes
about 15 minutes and gives PERFECT labels for your camera, mount and lawn.

    python label_frames.py                 # labels mydata/images/
    python label_frames.py some/folder

Writes YOLO labels to mydata/labels/<same name>.txt. An image with no poop is
saved as an EMPTY label file — those "hard negative" frames are just as
valuable as the positives, so don't skip the empty-lawn shots.

Mouse:  drag = draw a box round one poop
Keys:   d / -> next image      a / <- previous image
        u undo last box        c clear this image
        e mark EMPTY (no poop, still saved as a negative)
        z zoom toggle (2x, helps with distant targets)
        q save + quit
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
from pathlib import Path

import cv2

from paths import DATA
HERE = DATA
_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Label poop — drag a box · d next · a prev · u undo · e empty · q quit'


def yolo_path(img_path, img_root, lbl_root):
    rel = Path(img_path).relative_to(img_root)
    return Path(lbl_root) / rel.with_suffix('.txt')


def load_boxes(lbl, W, H):
    """YOLO file -> pixel boxes [(x0,y0,x1,y1)]."""
    out = []
    if not os.path.isfile(lbl):
        return out
    for line in open(lbl):
        p = line.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in p)
        out.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                    int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
    return out


def save_boxes(lbl, boxes, W, H):
    os.makedirs(os.path.dirname(lbl), exist_ok=True)
    with open(lbl, 'w') as f:
        for (x0, y0, x1, y1) in boxes:
            x0, x1 = sorted((max(0, x0), min(W, x1)))
            y0, y1 = sorted((max(0, y0), min(H, y1)))
            bw, bh = x1 - x0, y1 - y0
            if bw < 3 or bh < 3:
                continue
            f.write(f'0 {(x0+bw/2)/W:.6f} {(y0+bh/2)/H:.6f} '
                    f'{bw/W:.6f} {bh/H:.6f}\n')


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'mydata' / 'images')
    img_root = Path(folder)
    lbl_root = img_root.parent / 'labels'
    files = sorted(f for f in glob.glob(str(img_root / '*'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if not files:
        print(f'No images in {img_root}. Capture some first: '
              'python capture_dataset.py')
        return

    st = {'i': 0, 'boxes': [], 'drag': None, 'cur': None, 'zoom': False}

    def load(i):
        st['i'] = max(0, min(len(files) - 1, i))
        img = cv2.imread(files[st['i']])
        H, W = img.shape[:2]
        st['boxes'] = load_boxes(str(yolo_path(files[st['i']], img_root, lbl_root)),
                                 W, H)
        return img

    def store(img):
        H, W = img.shape[:2]
        save_boxes(str(yolo_path(files[st['i']], img_root, lbl_root)),
                   st['boxes'], W, H)

    img = load(0)
    cv2.namedWindow(WIN)

    def on_mouse(ev, x, y, flags, _):
        if st['zoom']:                      # map display coords back to image
            x, y = x // 2, y // 2
        if ev == cv2.EVENT_LBUTTONDOWN:
            st['drag'] = (x, y); st['cur'] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and st['drag']:
            st['cur'] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and st['drag']:
            x0, y0 = st['drag']
            if abs(x - x0) > 2 and abs(y - y0) > 2:
                st['boxes'].append((min(x0, x), min(y0, y),
                                    max(x0, x), max(y0, y)))
            st['drag'] = st['cur'] = None
    cv2.setMouseCallback(WIN, on_mouse)

    labelled = sum(1 for f in files
                   if os.path.isfile(yolo_path(f, img_root, lbl_root)))
    print(f'{len(files)} images, {labelled} already labelled. '
          'Drag a box round every poop; press e for empty frames.')

    try:
        while True:
            disp = img.copy()
            for (x0, y0, x1, y1) in st['boxes']:
                cv2.rectangle(disp, (x0, y0), (x1, y1), (33, 224, 212), 2)
            if st['drag'] and st['cur']:
                cv2.rectangle(disp, st['drag'], st['cur'], (0, 200, 255), 1)
            lbl_file = yolo_path(files[st['i']], img_root, lbl_root)
            done = os.path.isfile(lbl_file)
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (w, 26), (18, 18, 18), -1)
            cv2.putText(disp, f"{st['i']+1}/{len(files)}  boxes:{len(st['boxes'])}"
                        f"  {'SAVED' if done else 'unlabelled'}"
                        f"   {os.path.basename(files[st['i']])}",
                        (8, 18), _FONT, 0.48,
                        (120, 230, 120) if done else (0, 200, 255), 1)
            shown = cv2.resize(disp, None, fx=2, fy=2,
                               interpolation=cv2.INTER_NEAREST) if st['zoom'] else disp
            cv2.imshow(WIN, shown)

            k = cv2.waitKey(20) & 255
            if k in (ord('q'), 27):
                store(img); break
            elif k in (ord('d'), 83, 3):
                store(img); img = load(st['i'] + 1)
            elif k in (ord('a'), 81, 2):
                store(img); img = load(st['i'] - 1)
            elif k == ord('u') and st['boxes']:
                st['boxes'].pop()
            elif k == ord('c'):
                st['boxes'] = []
            elif k == ord('e'):
                st['boxes'] = []; store(img); img = load(st['i'] + 1)
            elif k == ord('z'):
                st['zoom'] = not st['zoom']
    finally:
        cv2.destroyAllWindows()
        lbls = glob.glob(str(lbl_root / '*.txt'))
        boxes = sum(len(open(p).read().strip().splitlines()) for p in lbls
                    if os.path.getsize(p))
        empties = sum(1 for p in lbls if os.path.getsize(p) == 0)
        print(f'\n{len(lbls)}/{len(files)} images labelled — '
              f'{boxes} boxes, {empties} negative frames')
        print(f'labels in {lbl_root}')
        if len(lbls) < len(files):
            print('Re-run to finish the rest.')
        else:
            print('Done. Package it:   zip -r mydata.zip mydata')
            print('Then re-run train_poop_real.py and upload mydata.zip.')


if __name__ == '__main__':
    main()
