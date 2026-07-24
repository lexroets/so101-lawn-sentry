"""Lock the demo box's exact colour — click it once, done.

For the quick demo where ONLY a specific tan box should register as poop. Point
the camera at the box on the lawn and click it a few times (different spots).
Each click samples the colour; the script writes a tight HSV band to
poop_box_ref.py, which poop_config.py auto-loads. The detector then matches
that colour and ignores the mower, wheels, QR code, etc.

    python calibrate_box.py      (click the box · u undo · s save · q quit)
"""
import numpy as np
import cv2

from trim_detector import build_pipeline
from camera_session import CameraSession

_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'Click the box to lock its colour'
PATCH = 6              # half-size of the sampled patch
H_MARGIN, S_MARGIN, V_MARGIN = 8, 30, 45
# Hard floors so a band can NEVER match desaturated (white/cream/grey plastic)
# or dark (body/tyres) — the two things that made the mower register. Kept low
# enough that PALE tan cardboard (S~40-60) still passes.
SAT_FLOOR, VAL_FLOOR = 30, 80


def band_from(samples):
    """Robust HSV band around the clicked samples (tight percentiles + margins,
    with hard saturation/brightness floors)."""
    a = np.asarray(samples, np.float32)
    lo = np.percentile(a, 15, axis=0)
    hi = np.percentile(a, 85, axis=0)
    lo = [max(0, lo[0] - H_MARGIN),
          max(SAT_FLOOR, lo[1] - S_MARGIN),
          max(VAL_FLOOR, lo[2] - V_MARGIN)]
    hi = [min(179, hi[0] + H_MARGIN),
          min(255, hi[1] + S_MARGIN),
          min(255, hi[2] + V_MARGIN)]
    return tuple(int(round(x)) for x in lo), tuple(int(round(x)) for x in hi)


def main():
    cam = CameraSession(build_pipeline)
    samples = []
    click = {'pt': None}
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, p:
                         click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    print('Click the box a few times (different spots). s = save, q = quit.')
    try:
        while True:
            g = cam.get()
            if g is None:
                if cam.crashed() and not cam.restart():
                    break
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
                continue
            frame = g['rgb'].getCvFrame()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if click['pt'] is not None:
                x, y = click['pt']; click['pt'] = None
                x0, x1 = max(0, x - PATCH), min(frame.shape[1], x + PATCH + 1)
                y0, y1 = max(0, y - PATCH), min(frame.shape[0], y + PATCH + 1)
                patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
                for px in patch:
                    samples.append(px.tolist())
                med = np.median(hsv[y0:y1, x0:x1].reshape(-1, 3), axis=0)
                print(f'  sampled HSV~{med.astype(int).tolist()}  (total {len(samples)} px)')
            disp = frame.copy()
            msg = f'clicks sampled: {len(samples)} px   '
            if samples:
                lo, hi = band_from(samples)
                msg += f'band {lo}->{hi}'
                # live-preview what would match
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                disp[m > 0] = (0, 90, 230)
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (18, 18, 18), -1)
            cv2.putText(disp, msg[:90], (8, 18), _FONT, 0.5, (0, 255, 255), 1)
            cv2.putText(disp, 'click box · u undo · s save · q quit',
                        (8, disp.shape[0] - 10), _FONT, 0.5, (80, 255, 80), 1)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 255
            if k == ord('q'):
                break
            if k == ord('u') and samples:
                del samples[-max(1, (2 * PATCH + 1) ** 2):]
                print('  undid last click')
            if k == ord('s'):
                if len(samples) < 20:
                    print('  click the box a few more times first.')
                    continue
                lo, hi = band_from(samples)
                with open('poop_box_ref.py', 'w') as f:
                    f.write('"""Locked demo box colour (calibrate_box.py)."""\n')
                    f.write('POOP_DEMO_BOX_MODE = True\n')
                    f.write(f'POOP_BOX_HSV_LO = {lo}\n')
                    f.write(f'POOP_BOX_HSV_HI = {hi}\n')
                print(f'  SAVED band {lo} -> {hi} to poop_box_ref.py')
                print('  Now run:  python launch.py')
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == '__main__':
    main()
