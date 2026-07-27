"""MobileNetV2 poop-tile classifier — ONNX via OpenCV DNN / onnxruntime (no torch).

Identical inference recipe to the grass/battery classifiers: single-file ONNX,
bilinear resize to 224, ImageNet normalisation, batch a list of BGR tiles.

Degrades gracefully: if poop_cls.onnx is absent, .ready is False and the
detector runs on colour + depth alone. Train with train_poop_colab.py.
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
import numpy as np
import cv2

from poop_config import POOP_MODEL, POOP_LABELS_FILE, POOP_DATASET_DIR

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PoopClassifier:
    def __init__(self):
        self.ready = False
        self.classes = ['grass', 'other', 'poop']   # alphabetical default
        self.backend = None
        from paths import find as _find
        path = str(_find(POOP_MODEL))
        if not os.path.isfile(path):
            print(f"[poop] no model at {POOP_MODEL} — running on colour + depth "
                  "only (train with train_poop_colab.py to sharpen).")
            return
        lp = str(_find(POOP_LABELS_FILE))
        if os.path.isfile(lp):
            with open(lp) as f:
                self.classes = [ln.strip() for ln in f if ln.strip()]
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                path, providers=['CPUExecutionProvider'])
            self._in = self._sess.get_inputs()[0].name
            self.backend = 'onnxruntime'
        except Exception:
            self._sess = None
            self.net = cv2.dnn.readNetFromONNX(path)
            self.backend = 'cv2.dnn'
        self.poop_idx = (self.classes.index('poop')
                         if 'poop' in self.classes else None)
        self.ready = True
        print(f"[poop] model loaded via {self.backend}, classes={self.classes}")

    def _prep(self, tile):
        img = cv2.resize(tile, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)

    def probs(self, tiles):
        """Full per-tile softmax (N x C) for a list of BGR tiles, or None."""
        if not self.ready or not tiles:
            return None
        blob = np.stack([self._prep(t) for t in tiles]).astype(np.float32)
        if self._sess is not None:
            out = self._sess.run(None, {self._in: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        out = np.asarray(out, np.float32)
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def poop_probs(self, tiles):
        """Per-tile P(poop) for a list of BGR tiles, or None."""
        if self.poop_idx is None:
            return None
        p = self.probs(tiles)
        return None if p is None else p[:, self.poop_idx]


def _selftest():
    """Confusion matrix of the exported ONNX over poop_dataset/ — separates a
    broken export from a weak model. Run: python poop_infer.py"""
    root = POOP_DATASET_DIR
    if not os.path.isdir(root):
        print('No poop_dataset/ to test against.')
        return
    clf = PoopClassifier()
    if not clf.ready:
        print('No model loaded — nothing to self-test.')
        return
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), int)
    for n in names:
        d = os.path.join(root, n)
        files = [f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))][:400]
        for i in range(0, len(files), 32):
            crops = [cv2.imread(os.path.join(d, f)) for f in files[i:i + 32]]
            crops = [c for c in crops if c is not None]
            if not crops:
                continue
            for p in clf.probs(crops):
                pred = clf.classes[int(np.argmax(p))]
                if pred in idx:
                    cm[idx[n], idx[pred]] += 1
    print('\nrows = truth, cols = predicted')
    w = max(len(n) for n in names) + 1
    print(' ' * w + ''.join(f'{n[:10]:>12}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:<{w}}' + ''.join(f'{v:>12}' for v in cm[i]))
    acc = cm.trace() / max(1, cm.sum())
    print(f'\noverall {acc:.3f}  (should match Colab val; if far lower, the '
          'EXPORT is wrong, not the model)')
    for i, n in enumerate(names):
        print(f'  {n:<8} recall {cm[i, i] / max(1, cm[i].sum()):.3f}')


if __name__ == '__main__':
    _selftest()
