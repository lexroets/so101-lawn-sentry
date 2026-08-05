"""See what the detector sees, and fix it. The arm never moves.

The trimmer does not use a trained model - it finds things standing PROUD of
the deck, from depth alone. That makes it fast and it makes it honest, but it
means four numbers decide everything, and if any of them is wrong for your
scene the detections look arbitrary:

    min proud   below this it is deck noise, above it is an object.
                Too low and every ripple in the stereo depth is a target;
                too high and a short grass patch is invisible.
    max proud   above this it is not on the deck at all - your hand, the arm
                itself, the far wall.
    min area    below this it is a speck. This is what usually kills real
                detections: at 0.35 m a small patch is not many pixels.
    max area    above this it is the deck, or the whole scene going wrong.

Watching the arm miss tells you nothing about which of the four is at fault.
This shows the raw proud-height map, every blob it found, and WHY each one was
rejected - so you can see it rather than guess.

    python trimmer/tune_detector.py

    1 / 2   min proud   down / up          5 / 6   min area  down / up
    3 / 4   max proud   down / up          7 / 8   max area  down / up
    m       cycle the view: colour / proud height / accepted mask
    s       print the settings to paste into trimmer_config.py
    q       quit
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
import cv2
import numpy as np

import trimmer_config as tc
from trim_detector import build_pipeline, detect_patches, _row_deck
from camera_session import CameraSession
from keys import Keys

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'detector tuning - the arm does not move'
VIEWS = ('colour', 'proud height', 'accepted mask')


def proud_map(depth):
    """How far each pixel stands above its row's deck level, in metres.

    Per-row rather than one plane for the whole image, because the deck
    recedes with height in the frame. The median across a row is robust: the
    deck is most of the row, so a few blades of grass do not move it.
    """
    deck = _row_deck(depth)
    with np.errstate(invalid='ignore'):
        pr = deck[:, None] - depth              # nearer than the deck = proud
    pr[~np.isfinite(pr)] = 0.0
    return pr


def main():
    s = dict(min_proud=tc.OBJECT_MIN_PROUD_M, max_proud=tc.GRASS_MAX_PROUD_M,
             min_area=float(tc.OBJECT_MIN_AREA_PX),
             max_area=float(tc.OBJECT_MAX_AREA_PX))
    print('\nstarting from trimmer_config.py:')
    for k, v in s.items():
        print(f'   {k:10s} {v}')
    print('\nPut a target in front of the camera. The arm is not connected.\n')

    cam = CameraSession(build_pipeline)
    kb = Keys()
    view = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            g = cam.get()
            if g is None:
                if cam.crashed() and not cam.restart():
                    break
                kb.poll(1)
                if kb.hit(ord('q')):
                    break
                continue
            image = g['rgb'].getCvFrame()
            depth = g['depth'].getFrame().astype(np.float32) / 1000.0
            pr = proud_map(depth)

            # what the detector accepts right now
            good = detect_patches(image, depth, min_proud=s['min_proud'],
                                  max_proud=s['max_proud'],
                                  min_area=s['min_area'], max_area=s['max_area'])
            # and everything it COULD have accepted, so rejections are visible
            loose = detect_patches(image, depth, min_proud=s['min_proud'] * 0.4,
                                   max_proud=s['max_proud'] * 2.5,
                                   min_area=25, max_area=1e9)

            if view == 1:
                v = np.clip(pr / max(1e-6, s['max_proud']), 0, 1)
                disp = cv2.applyColorMap((v * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO)
            elif view == 2:
                m = ((pr >= s['min_proud']) & (pr <= s['max_proud'])).astype(np.uint8)
                disp = cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR)
            else:
                disp = image.copy()

            kept_ids = {(p['u'], p['v']) for p in good}
            for p in loose:
                x, y, w, h = p['bbox']
                if (p['u'], p['v']) in kept_ids:
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (80, 255, 80), 2)
                    cv2.putText(disp, f"{p['area']:.0f}px  {p['z']:.2f}m",
                                (x, max(12, y - 5)), FONT, 0.42, (80, 255, 80), 1)
                else:
                    # say WHICH limit threw it out - that is the whole point
                    why = ('area low' if p['area'] < s['min_area'] else
                           'area high' if p['area'] > s['max_area'] else
                           'proud')
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (90, 90, 200), 1)
                    cv2.putText(disp, why, (x, max(12, y - 5)), FONT, 0.4,
                                (120, 120, 220), 1)

            hdr = [f'view: {VIEWS[view]}   (m to change)',
                   f'proud  {s["min_proud"]*100:5.1f} - {s["max_proud"]*100:5.1f} cm'
                   f'    [1/2  3/4]',
                   f'area   {s["min_area"]:6.0f} - {s["max_area"]:7.0f} px'
                   f'   [5/6  7/8]',
                   f'accepted {len(good)} of {len(loose)} blobs      s = print '
                   f'settings   q = quit']
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 24 * len(hdr) + 8),
                          (16, 16, 16), -1)
            for i, t in enumerate(hdr):
                cv2.putText(disp, t, (10, 20 + i * 23), FONT, 0.48,
                            (255, 220, 90), 1)
            cv2.imshow(WIN, disp)

            kb.poll(1)
            if kb.hit(ord('q')):
                break
            if kb.hit(ord('m')):
                view = (view + 1) % len(VIEWS)
            if kb.tap(ord('1')):
                s['min_proud'] = max(0.002, s['min_proud'] - 0.002)
            if kb.tap(ord('2')):
                s['min_proud'] += 0.002
            if kb.tap(ord('3')):
                s['max_proud'] = max(s['min_proud'] + 0.01, s['max_proud'] - 0.01)
            if kb.tap(ord('4')):
                s['max_proud'] += 0.01
            if kb.tap(ord('5')):
                s['min_area'] = max(25, s['min_area'] - 50)
            if kb.tap(ord('6')):
                s['min_area'] += 50
            if kb.tap(ord('7')):
                s['max_area'] = max(s['min_area'] + 100, s['max_area'] - 2000)
            if kb.tap(ord('8')):
                s['max_area'] += 2000
            if kb.hit(ord('s')):
                print('\n' + '=' * 62)
                print('Paste into core/trimmer_config.py:\n')
                print(f'OBJECT_MIN_PROUD_M = {s["min_proud"]:.3f}')
                print(f'GRASS_MAX_PROUD_M  = {s["max_proud"]:.3f}')
                print(f'OBJECT_MIN_AREA_PX = {int(s["min_area"])}')
                print(f'OBJECT_MAX_AREA_PX = {int(s["max_area"])}')
                print('=' * 62 + '\n')
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
