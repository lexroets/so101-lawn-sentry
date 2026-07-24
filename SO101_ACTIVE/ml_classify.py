"""MobileNetV2 crop classifier — colour AND orientation from one model.

Runs the ONNX exported by ml_training/train_mobilenetv2.py.

NO NEW DEPENDENCIES: uses OpenCV's DNN module, which you already have.
onnxruntime is used instead when available (roughly 2x faster), but is not
required — relevant because torch/onnxruntime have no Python 3.14 wheels yet.

Cheap by construction:
  * only the N nearest candidates are classified each frame
  * results are cached by position for a short window, so a stationary
    battery is not re-run every frame
A pick only needs the verdict during CENTRE, so this is nowhere near the
per-frame cost of the old per-candidate depth statistics.
"""
import os
import time
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH = [
    os.path.join(_HERE, 'battery_cls.onnx'),
    os.path.join(os.path.dirname(_HERE), 'ml_training', 'battery_cls.onnx'),
    os.path.join(_HERE, 'ml_training', 'battery_cls.onnx'),
]
_LABELS_NAME = 'labels.txt'

IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Defaults; main_ml.py can override.
MAX_PER_FRAME = 4        # classify only the nearest few candidates
CACHE_S = 0.25           # reuse a verdict for this long
CACHE_RADIUS_PX = 18     # ...if the box centre moved less than this
MIN_CONF = 0.40          # below this a battery verdict is 'unknown'. The model
                         # is well-trained (99.5%); a lower bar stops it
                         # flickering to 'unknown' on borderline live frames.
BG_MIN_CONF = 0.80       # 'background' needs THIS much confidence to stick,
                         # because that class was under-trained (30 samples)
                         # and otherwise swallows real batteries.


class ModelMissing(RuntimeError):
    pass


def _find_model():
    for p in _SEARCH:
        if os.path.isfile(p):
            return p
    raise ModelMissing(
        'battery_cls.onnx not found. Train it first:\n'
        '  1. python collect_crops.py            (collect labelled crops)\n'
        '  2. train in Colab with ml_training/train_mobilenetv2.py\n'
        '  3. put battery_cls.onnx + labels.txt beside main_ml.py\n'
        'Searched:\n  ' + '\n  '.join(_SEARCH))


class BatteryClassifier:
    def __init__(self, model_path=None, labels_path=None):
        self.path = model_path or _find_model()
        lp = labels_path or os.path.join(os.path.dirname(self.path),
                                         _LABELS_NAME)
        if os.path.isfile(lp):
            with open(lp) as f:
                self.classes = [ln.strip() for ln in f if ln.strip()]
        else:
            # ImageFolder orders classes alphabetically — this is that order.
            self.classes = ['background', 'pink_lying', 'pink_upright',
                            'ryobi_lying', 'ryobi_upright']
            print(f'[ml] {_LABELS_NAME} not found — assuming alphabetical '
                  f'order {self.classes}')

        self.backend = 'cv2.dnn'
        self._sess = None
        try:                                   # faster path if it exists
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                self.path, providers=['CPUExecutionProvider'])
            self._in = self._sess.get_inputs()[0].name
            self.backend = 'onnxruntime'
        except Exception:
            self.net = cv2.dnn.readNetFromONNX(self.path)
            # Don't set a preferable target: OpenCV 5.0's new graph engine
            # ignores it and logs a WARN. Default is CPU, which is what we
            # want anyway.

        self._cache = []          # [(u, v, t, verdict), ...]
        self.n_calls = 0
        self.last_ms = 0.0
        print(f'[ml] loaded {os.path.basename(self.path)} via {self.backend} '
              f'({len(self.classes)} classes)')

    # ---------------------------------------------------------------- infer
    def _prep(self, crop):
        # BILINEAR to match torchvision's training-time Resize. Battery crops
        # are mostly SMALLER than 224px, so this is upscaling — INTER_AREA
        # (a downscaling filter) produced blocky results that shifted the
        # distribution and pushed dark lying crops into 'background'.
        img = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)

    def _forward(self, batch):
        blob = np.stack(batch).astype(np.float32)
        if self._sess is not None:
            out = self._sess.run(None, {self._in: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        out = np.asarray(out, np.float32)
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _verdict(self, probs):
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        name, conf = self.classes[top], float(probs[top])
        raw = (name, round(conf, 2))          # for diagnostics
        # The background class was trained on very few samples, so it tends to
        # over-fire and swallow real batteries. Only accept 'background' when
        # it is HIGHLY confident; otherwise take the best BATTERY class and
        # judge that against the normal threshold.
        if name == 'background':
            if conf >= BG_MIN_CONF:
                return {'cls': name, 'conf': conf, 'colour': 'unknown',
                        'lying': None, 'ok': False, 'raw': raw}
            for j in order[1:]:
                if self.classes[int(j)] != 'background':
                    top = int(j)
                    name, conf = self.classes[top], float(probs[top])
                    break
        if conf < MIN_CONF:
            return {'cls': name, 'conf': conf, 'colour': 'unknown',
                    'lying': None, 'ok': False, 'raw': raw}
        colour = 'ryobi' if name.startswith('ryobi') else 'pink'
        return {'cls': name, 'conf': conf, 'colour': colour,
                'lying': name.endswith('_lying'), 'ok': True, 'raw': raw}

    # ---------------------------------------------------------------- cache
    def _cached(self, u, v, now):
        for cu, cv_, t, verdict in self._cache:
            if now - t <= CACHE_S and abs(cu - u) <= CACHE_RADIUS_PX \
                    and abs(cv_ - v) <= CACHE_RADIUS_PX:
                return verdict
        return None

    def _store(self, u, v, now, verdict):
        self._cache = [c for c in self._cache if now - c[2] <= CACHE_S]
        self._cache.append((u, v, now, verdict))

    # ----------------------------------------------------------------- API
    def classify_candidates(self, image, candidates, pad_frac=0.18):
        """Attach 'ml' to each candidate, nearest ones first.

        Padding matches collect_crops.py — the model was trained on crops
        with that much context, so inference must reproduce it."""
        now = time.monotonic()
        todo, batch = [], []
        h_img, w_img = image.shape[:2]
        for c in sorted(candidates, key=lambda c: c['z'])[:MAX_PER_FRAME]:
            hit = self._cached(c['u'], c['v'], now)
            if hit is not None:
                c['ml'] = hit
                continue
            x, y, w, h = c['bbox']
            px, py = int(w * pad_frac), int(h * pad_frac)
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(w_img, x + w + px), min(h_img, y + h + py)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            todo.append(c)
            batch.append(self._prep(image[y0:y1, x0:x1]))
        if batch:
            t0 = time.monotonic()
            probs = self._forward(batch)
            self.last_ms = (time.monotonic() - t0) * 1000.0
            self.n_calls += len(batch)
            for c, p in zip(todo, probs):
                v = self._verdict(p)
                c['ml'] = v
                self._store(c['u'], c['v'], now, v)
        return candidates


# --------------------------------------------------------------- self-test
def _selftest():
    """Check the exported ONNX reproduces training accuracy on the dataset.

    Run this BEFORE wiring the model into the robot — it separates 'the
    export is broken' from 'the model is weak', which are very different
    problems."""
    root = os.path.join(os.path.dirname(_HERE), 'ml_training', 'dataset')
    if not os.path.isdir(root):
        print('No dataset at', root)
        return
    clf = BatteryClassifier()
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), int)

    def _op_class(verdict):
        """What the ROBOT would conclude (uses the background override)."""
        if not verdict['ok']:
            return 'background'
        return f"{verdict['colour']}_{'lying' if verdict['lying'] else 'upright'}"

    op_correct = op_total = 0
    for n in names:
        d = os.path.join(root, n)
        files = [f for f in os.listdir(d) if f.endswith('.png')][:120]
        for i in range(0, len(files), 16):
            crops = []
            for f in files[i:i + 16]:
                img = cv2.imread(os.path.join(d, f))
                if img is not None:
                    crops.append(clf._prep(img))
            if not crops:
                continue
            for p in clf._forward(crops):
                pred = clf.classes[int(np.argmax(p))]
                if pred in idx:
                    cm[idx[n], idx[pred]] += 1
                # Operational verdict (what the robot actually acts on, with
                # the background-override applied).
                op = _op_class(clf._verdict(p))
                op_total += 1
                if op == n:
                    op_correct += 1
    print('\nrows = truth, cols = predicted (raw argmax)')
    w = max(len(n) for n in names) + 1
    print(' ' * w + ''.join(f'{n[:11]:>13}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:<{w}}' + ''.join(f'{v:>13}' for v in cm[i]))
    acc = cm.trace() / max(1, cm.sum())
    print(f'\nraw-argmax overall {acc:.3f}')
    print(f'OPERATIONAL overall {op_correct / max(1, op_total):.3f}  '
          '<-- what the robot actually achieves (background-override applied)')
    for a, b in (('ryobi_upright', 'ryobi_lying'),
                 ('pink_upright', 'pink_lying')):
        if a in idx and b in idx:
            wrong = cm[idx[a], idx[b]] + cm[idx[b], idx[a]]
            tot = cm[idx[a]].sum() + cm[idx[b]].sum()
            print(f'{a} <-> {b}: {wrong / max(1, tot):.1%}')


if __name__ == '__main__':
    _selftest()
