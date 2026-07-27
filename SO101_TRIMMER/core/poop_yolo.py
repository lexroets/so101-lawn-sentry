"""Run a distilled YOLOv8 poop detector (ONNX) — no torch, OAK-D friendly.

This is the STUDENT model: trained on SAM-3.1 auto-labels (see sam_autolabel.py
+ train_poop_yolo.py), exported to poop_yolo.onnx. It replaces the colour
heuristic with a learned detector, but returns the SAME box format, so
PoopDetector / the lawn map / the dashboard all keep working unchanged.

Decodes standard Ultralytics YOLOv8 export (output [1, 4+nc, N] or [1, N, 4+nc],
single class), with letterbox pre-processing and class-agnostic NMS. Runs via
onnxruntime if available, else OpenCV DNN.

Degrades gracefully: if poop_yolo.onnx is absent, .ready is False.
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

import poop_config as _cfg
from poop_config import (POOP_YOLO_MODEL, POOP_YOLO_IMG, POOP_YOLO_CONF,
                         POOP_YOLO_IOU)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _cfg_int(name, default):
    try:
        return int(getattr(_cfg, name, default))
    except Exception:
        return default


def _cfg_float(name, default):
    try:
        return float(getattr(_cfg, name, default))
    except Exception:
        return default


class PoopYolo:
    def __init__(self):
        self.ready = False
        self.backend = None
        from paths import find as _find
        path = str(_find(POOP_YOLO_MODEL))
        if not os.path.isfile(path):
            return
        if os.path.getsize(path) < 10_000:
            print(f"[poop-yolo] {POOP_YOLO_MODEL} looks empty/truncated "
                  f"({os.path.getsize(path)} bytes) — ignoring it and using the "
                  "colour detector. Re-download the trained model.")
            return
        # try onnxruntime, then OpenCV DNN; if BOTH fail the file is corrupt, so
        # fall back to the heuristic instead of taking the whole robot down.
        err = None
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                path, providers=['CPUExecutionProvider'])
            self._in = self._sess.get_inputs()[0].name
            self.backend = 'onnxruntime'
        except Exception as e1:
            err = e1
            self._sess = None
            try:
                self.net = cv2.dnn.readNetFromONNX(path)
                self.backend = 'cv2.dnn'
                err = None
            except Exception as e2:
                err = e2
        if err is not None:
            print(f"[poop-yolo] could not load {POOP_YOLO_MODEL} "
                  f"({str(err)[:110]}). Falling back to the colour detector.")
            return

        # Read the input size FROM THE MODEL. If you train at 960 but leave
        # POOP_YOLO_IMG at 640, every frame gets squashed to the wrong scale
        # and accuracy quietly collapses. Trust the file over the config.
        self.imgsz = POOP_YOLO_IMG
        try:
            if self._sess is not None:
                shp = self._sess.get_inputs()[0].shape      # [1,3,H,W]
                h, w = shp[2], shp[3]
                if isinstance(h, int) and isinstance(w, int) and h == w and h >= 64:
                    if h != POOP_YOLO_IMG:
                        print(f"[poop-yolo] model expects {h}px but config says "
                              f"{POOP_YOLO_IMG}px — using {h}px (the model wins). "
                              f"Set POOP_YOLO_IMG = {h} to silence this.")
                    self.imgsz = h
        except Exception:
            pass
        self.ready = True
        print(f"[poop-yolo] model loaded via {self.backend} @ {self.imgsz}px")

    # -- pre / post ----------------------------------------------------------
    def _letterbox(self, img):
        """Resize keeping aspect ratio, pad to the model's square input."""
        S = getattr(self, 'imgsz', POOP_YOLO_IMG)
        h, w = img.shape[:2]
        r = min(S / h, S / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((S, S, 3), 114, np.uint8)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, r, dx, dy

    def infer(self, image, tiles=None):
        """Detections [{'bbox':(x,y,w,h),'conf','u','v'}] in ORIGINAL pixels.

        With tiling enabled the frame is ALSO scanned as a grid of overlapping
        crops, each blown up to the full model input. A distant poop that is
        only ~18 px in the frame becomes ~36 px to the model, which is the
        difference between invisible and detected. Results from every pass are
        merged with NMS."""
        if not self.ready:
            return []
        n = _cfg_int('POOP_YOLO_TILES', 0) if tiles is None else tiles
        if n and n > 1:
            return self._infer_tiled(image, n)
        return self._infer_one(image)

    def _infer_tiled(self, image, n):
        """Full frame + an n x n grid of overlapping crops, merged."""
        H, W = image.shape[:2]
        ov = _cfg_float('POOP_YOLO_TILE_OVERLAP', 0.2)
        dets = self._infer_one(image)              # full frame: catches big ones
        tw, th = W / n, H / n
        ox, oy = tw * ov, th * ov
        for iy in range(n):
            for ix in range(n):
                x0 = max(0, int(ix * tw - ox)); y0 = max(0, int(iy * th - oy))
                x1 = min(W, int((ix + 1) * tw + ox)); y1 = min(H, int((iy + 1) * th + oy))
                crop = image[y0:y1, x0:x1]
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue
                for d in self._infer_one(crop):    # offset back to full frame
                    bx, by, bw, bh = d['bbox']
                    d['bbox'] = (bx + x0, by + y0, bw, bh)
                    d['u'] += x0; d['v'] += y0
                    dets.append(d)
        return self._merge(dets)

    def _merge(self, dets):
        """Class-agnostic NMS over detections gathered from several passes."""
        if len(dets) < 2:
            return dets
        rects = [list(d['bbox']) for d in dets]
        scores = [float(d['conf']) for d in dets]
        idx = cv2.dnn.NMSBoxes(rects, scores, POOP_YOLO_CONF, POOP_YOLO_IOU)
        keep = [dets[int(i)] for i in np.array(idx).flatten()] if len(idx) else []
        keep.sort(key=lambda d: -d['conf'])
        return keep

    def _infer_one(self, image):
        """Single forward pass over one image (or crop)."""
        S = getattr(self, 'imgsz', POOP_YOLO_IMG)
        canvas, r, dx, dy = self._letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        if self._sess is not None:
            out = self._sess.run(None, {self._in: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        out = np.asarray(out, np.float32)
        if out.ndim == 3:
            out = out[0]
        # normalise to rows = detections, cols = [cx,cy,w,h, scores...]
        if out.shape[0] < out.shape[1]:      # [4+nc, N] -> [N, 4+nc]
            out = out.T
        if out.shape[1] < 5:
            return []
        boxes_xywh = out[:, :4]
        scores = out[:, 4:].max(axis=1) if out.shape[1] > 5 else out[:, 4]
        keep = scores >= POOP_YOLO_CONF
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]
        if not len(boxes_xywh):
            return []
        # xywh (letterbox px) -> xyxy (original px)
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r
        y2 = (cy + bh / 2 - dy) / r
        H, W = image.shape[:2]
        rects = [[int(max(0, a)), int(max(0, b)),
                  int(min(W, c) - max(0, a)), int(min(H, d) - max(0, b))]
                 for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(),
                               POOP_YOLO_CONF, POOP_YOLO_IOU)
        dets = []
        for i in np.array(idx).flatten():
            x, y, w, h = rects[int(i)]
            if w < 3 or h < 3:
                continue
            dets.append({'bbox': (x, y, w, h), 'conf': float(scores[int(i)]),
                         'u': int(x + w / 2), 'v': int(y + h / 2)})
        dets.sort(key=lambda d: -d['conf'])
        return dets


def _selftest():
    clf = PoopYolo()
    print('ready:', clf.ready, '| backend:', clf.backend)
    if clf.ready:
        import numpy as _np
        d = clf.infer(_np.zeros((480, 640, 3), _np.uint8))
        print('inference ran, detections:', len(d))


if __name__ == '__main__':
    _selftest()
