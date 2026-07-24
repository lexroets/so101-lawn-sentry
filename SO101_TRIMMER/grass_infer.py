"""MobileNetV2 grass-tile classifier — ONNX via OpenCV DNN (no torch).

Same inference approach that works for the battery classifier: single-file
ONNX, cv2.dnn (or onnxruntime if present), bilinear resize to 224, ImageNet
normalisation. Classifies a BATCH of tiles at once.

Degrades gracefully: if grass_cls.onnx is absent, .ready is False and the
detector falls back to GLCM texture + depth alone.
"""
import os
import numpy as np
import cv2

from trimmer_config import GRASS_MODEL, GRASS_LABELS_FILE

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class GrassClassifier:
    def __init__(self):
        self.ready = False
        self.classes = ['cut', 'other', 'uncut']   # alphabetical default
        self.backend = None
        path = os.path.join(_HERE, GRASS_MODEL)
        if not os.path.isfile(path):
            print(f"[grass] no model at {GRASS_MODEL} — running on GLCM + depth "
                  "only (train with train_grass_colab.py to improve).")
            return
        lp = os.path.join(_HERE, GRASS_LABELS_FILE)
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
        self.uncut_idx = (self.classes.index('uncut')
                          if 'uncut' in self.classes else None)
        self.ready = True
        print(f"[grass] model loaded via {self.backend}, classes={self.classes}")

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

    def uncut_probs(self, tiles):
        """Return the per-tile P(uncut) for a list of BGR tiles, or None."""
        if self.uncut_idx is None:
            return None
        p = self.probs(tiles)
        return None if p is None else p[:, self.uncut_idx]


def _selftest():
    """Verify the exported ONNX reproduces training accuracy on the tiles —
    separates 'export is broken' from 'model is weak'. Run: python grass_infer.py
    """
    import os
    root = os.path.join(_HERE, 'grass_dataset')
    if not os.path.isdir(root):
        print('No grass_dataset/ to test against.')
        return
    clf = GrassClassifier()
    if not clf.ready:
        print('No model loaded — nothing to self-test.')
        return
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), int)
    for n in names:
        d = os.path.join(root, n)
        files = [f for f in os.listdir(d) if f.endswith('.png')][:400]
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
