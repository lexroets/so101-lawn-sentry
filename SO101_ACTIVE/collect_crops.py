"""Harvest labelled battery crops for MobileNetV2 training — camera only.

SAFE TO RUN DURING DEMO PREP: this script never opens the serial port, so
the arm is not powered and cannot move. It only reads the camera and writes
image files. It imports config.py and detector.py READ-ONLY and modifies
nothing the demo depends on.

WHAT IT DOES
  Shows the live view with the existing depth detector's boxes. You click a
  battery to select it, then press a number key to file a crop under that
  label. Each press grabs a short BURST across several frames, so you get
  natural variation (noise, slight movement) rather than one duplicate.

LABELS
  1  ryobi_upright     black RB4L30 standing on its end
  2  ryobi_lying       black RB4L30 on its side
  3  pink_upright      pink cell standing
  4  pink_lying        pink cell on its side
  5  background        mat, bucket, gripper, hand, clutter — anything NOT a
                       battery. Collect plenty: this is what teaches the
                       model to stop calling empty mat a "pink battery".

KEYS
  click  select the nearest detected box
  1-5    file a burst of crops under that label
  u      undo — delete the most recent burst
  b      toggle BACKGROUND SWEEP: files whatever is detected as class 5
         automatically, one per second. Point at empty mat / buckets and
         let it run for a minute.
  q      quit

HOW MANY
  Aim for ~150 per battery class and ~300 background. Vary distance
  (25-40cm), heading, lighting, and which face shows. Move a battery
  between bursts rather than taking 150 shots of one pose — pose variety
  matters far more than raw count.

Crops land in  ml_training/dataset/<label>/  next to this project, plus a
crops.csv of metadata for later analysis.
"""
from collections import deque
from datetime import timedelta
import csv
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import detect_candidates

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(_HERE), 'ml_training', 'dataset')
CSV_PATH = os.path.join(os.path.dirname(_HERE), 'ml_training', 'crops.csv')

LABELS = {
    ord('1'): 'ryobi_upright',
    ord('2'): 'ryobi_lying',
    ord('3'): 'pink_upright',
    ord('4'): 'pink_lying',
    ord('5'): 'background',
}
BURST = 5              # crops captured per key press, across successive frames
BURST_GAP_S = 0.12     # spacing between burst frames
PAD_FRAC = 0.18        # context padding around the box — helps the model
MIN_CROP_PX = 24
BG_SWEEP_GAP_S = 1.0

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_pipeline():
    pipeline = dai.Pipeline()
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rgb_out = rgb.requestOutput(size=(CAMERA_WIDTH, CAMERA_HEIGHT),
                                fps=CAMERA_FPS, enableUndistortion=True)
    left_out = left.requestOutput(size=(640, 400),
                                  type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    right_out = right.requestOutput(size=(640, 400),
                                    type=dai.ImgFrame.Type.GRAY8, fps=CAMERA_FPS)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    stereo.setSubpixel(False)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    rgb_out.link(stereo.inputAlignTo)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=50))
    sync.setSyncAttempts(-1)
    rgb_out.link(sync.inputs['rgb'])
    stereo.depth.link(sync.inputs['depth'])
    return pipeline, sync.out.createOutputQueue(maxSize=1, blocking=False)


def ensure_dirs():
    for name in set(LABELS.values()):
        os.makedirs(os.path.join(OUT_ROOT, name), exist_ok=True)


def counts():
    out = {}
    for name in sorted(set(LABELS.values())):
        d = os.path.join(OUT_ROOT, name)
        out[name] = len([f for f in os.listdir(d)
                         if f.endswith('.png')]) if os.path.isdir(d) else 0
    return out


def crop_of(image, bbox):
    """Box plus context padding, clamped to the frame."""
    x, y, w, h = bbox
    px, py = int(w * PAD_FRAC), int(h * PAD_FRAC)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1 = min(image.shape[1], x + w + px)
    y1 = min(image.shape[0], y + h + py)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    return image[y0:y1, x0:x1].copy()


def main():
    ensure_dirs()
    new_rows = []
    pipeline, queue = build_pipeline()
    pipeline.start()
    print(__doc__)
    print('Existing dataset:', counts())

    click = {'pt': None}
    selected = None
    pending = deque()          # queued burst captures: (label, due_time)
    last_burst = []            # file paths of the most recent burst (for undo)
    bg_sweep = False
    bg_next = 0.0
    message, message_until = '', 0.0
    window = 'SO-101 crop collector  (click, then 1-5 · b sweep · u undo · q quit)'

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    def save(label, image, bbox, z):
        patch = crop_of(image, bbox)
        if patch is None:
            return None
        d = os.path.join(OUT_ROOT, label)
        path = os.path.join(d, f'{label}_{int(time.time() * 1000)}.png')
        cv2.imwrite(path, patch)
        new_rows.append({
            'file': os.path.relpath(path, os.path.dirname(CSV_PATH)),
            'label': label, 'z_m': round(float(z), 4) if z else '',
            'w_px': bbox[2], 'h_px': bbox[3],
            'aspect': round(bbox[3] / max(1, bbox[2]), 3),
        })
        return path

    try:
        while pipeline.isRunning():
            now = time.monotonic()
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                cands = [c for c in detect_candidates(image, depth)
                         if c['kind'] == 'target']
                clean = image.copy()          # save crops WITHOUT overlay

                if click['pt'] is not None:
                    cx, cy = click['pt']
                    click['pt'] = None
                    if cands:
                        selected = min(cands, key=lambda c: (c['u'] - cx) ** 2
                                       + (c['v'] - cy) ** 2)
                    else:
                        message, message_until = 'Nothing detected there', now + 1.5
                if selected is not None and cands:
                    selected = min(cands, key=lambda c: (c['u'] - selected['u']) ** 2
                                   + (c['v'] - selected['v']) ** 2)

                # queued burst frames
                while pending and pending[0][1] <= now:
                    label, _ = pending.popleft()
                    if label == 'background':
                        # random patch away from any detection
                        for _ in range(12):
                            bw = np.random.randint(40, 160)
                            bh = np.random.randint(40, 160)
                            bx = np.random.randint(0, max(1, image.shape[1] - bw))
                            by = np.random.randint(0, max(1, image.shape[0] - bh))
                            if all(abs(bx + bw // 2 - c['u']) > 80
                                   or abs(by + bh // 2 - c['v']) > 80
                                   for c in cands):
                                p = save(label, clean, (bx, by, bw, bh), None)
                                if p:
                                    last_burst.append(p)
                                break
                    elif selected is not None:
                        p = save(label, clean, selected['bbox'], selected['z'])
                        if p:
                            last_burst.append(p)

                if bg_sweep and now >= bg_next:
                    bg_next = now + BG_SWEEP_GAP_S
                    pending.append(('background', now))

                # ---------------- overlay ----------------
                for c in cands:
                    x, y, w, h = c['bbox']
                    sel = selected is not None and c['u'] == selected['u']
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (255, 255, 255) if sel else (0, 190, 255),
                                  3 if sel else 1)
                cnt = counts()
                bar = np.zeros((96, image.shape[1], 3), np.uint8)
                bar[:] = 22
                cv2.putText(bar, 'CLICK a battery, then:  1 ryobi-up   2 ryobi-lying'
                            '   3 pink-up   4 pink-lying   5 background',
                            (10, 20), _FONT, 0.42, (236, 240, 241), 1, cv2.LINE_AA)
                xoff = 10
                for name in ('ryobi_upright', 'ryobi_lying', 'pink_upright',
                             'pink_lying', 'background'):
                    col = ((80, 220, 100) if name.startswith('ryobi')
                           else (255, 90, 200) if name.startswith('pink')
                           else (150, 160, 165))
                    cv2.putText(bar, f'{name}: {cnt[name]}', (xoff, 44),
                                _FONT, 0.42, col, 1, cv2.LINE_AA)
                    xoff += 150
                cv2.putText(bar, ('BACKGROUND SWEEP ON — point at empty mat/buckets'
                                  if bg_sweep else
                                  'b = background sweep    u = undo last    q = quit'),
                            (10, 68), _FONT, 0.42,
                            (0, 190, 255) if bg_sweep else (150, 160, 165), 1,
                            cv2.LINE_AA)
                if now < message_until:
                    cv2.putText(bar, message, (10, 88), _FONT, 0.42,
                                (60, 60, 235), 1, cv2.LINE_AA)
                cv2.imshow(window, np.vstack([bar, image]))

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key in LABELS:
                label = LABELS[key]
                if label != 'background' and selected is None:
                    message, message_until = 'Click a battery first', now + 1.5
                else:
                    last_burst = []
                    for i in range(BURST):
                        pending.append((label, now + i * BURST_GAP_S))
                    message = f'{label}  x{BURST}'
                    message_until = now + 1.0
            if key == ord('b'):
                bg_sweep = not bg_sweep
                bg_next = 0.0
            if key == ord('u'):
                n = 0
                for p in last_burst:
                    try:
                        os.remove(p)
                        n += 1
                    except OSError:
                        pass
                new_rows[:] = [r for r in new_rows
                               if not any(r['file'].endswith(os.path.basename(p))
                                          for p in last_burst)]
                last_burst = []
                message, message_until = f'undid {n}', now + 1.5
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        if new_rows:
            exists = os.path.isfile(CSV_PATH)
            with open(CSV_PATH, 'a', newline='') as f:
                wtr = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
                if not exists:
                    wtr.writeheader()
                wtr.writerows(new_rows)
        final = counts()
        print('\nDataset now:', final)
        total_batt = sum(v for k, v in final.items() if k != 'background')
        print(f'  battery crops: {total_batt}   background: {final["background"]}')
        if min((v for k, v in final.items() if k != 'background'), default=0) < 80:
            print('  -> thin classes still under 80. More pose variety beats '
                  'more shots of the same pose.')


if __name__ == '__main__':
    main()
