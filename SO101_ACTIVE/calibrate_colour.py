"""Quick battery-ID calibration — teach the system YOUR two battery types
under YOUR lighting and YOUR surface (green mat included).

Lay out a few of each type (e.g. 2 pink + 2 Ryobi). Then, for each battery:

    1. CLICK on it  (the nearest detected box turns white = selected)
    2. Press  R  if it is a RYOBI (black)   or   P  if it is a PINK one

Do each battery 2-3 times, and move one or two of them partway through so
the samples cover a few positions and angles. You need at least 3 of each.

    F = finish and write the calibration      q = abort

Writes colour_calibration.py, which config.py picks up automatically.
Delete that file to go back to the hand-set defaults.

The arm is NOT powered by this script — it is camera-only and safe to run
with the robot switched off.
"""
from collections import defaultdict
from datetime import timedelta
import os
import time
import cv2
import depthai as dai
import numpy as np
from config import *
from detector import detect_candidates

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'colour_calibration.py')
MIN_PER_CLASS = 3

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


def sample(image, bbox):
    """Raw colour evidence for one box — the same pixel tests the runtime
    classifier uses, but returned as components so we can see the split."""
    x, y, w, h = bbox
    ix, iy = int(w * 0.2), int(h * 0.2)
    roi = image[max(0, y + iy):y + h - iy, max(0, x + ix):x + w - ix]
    if roi.size == 0:
        roi = image[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = float(roi.shape[0] * roi.shape[1])
    pink_hued = (((hue >= PINK_HUE[0]) & (hue <= PINK_HUE[1]))
                 | ((hue >= PINK_HUE2[0]) & (hue <= PINK_HUE2[1])))
    pink = int(np.count_nonzero(pink_hued & (sat >= PINK_S_MIN) & (val >= 120)))
    dark = int(np.count_nonzero(val <= DARK_V_MAX))
    white = int(np.count_nonzero((val >= WHITE_V_MIN) & (sat <= WHITE_S_MAX)
                                 & ~pink_hued))
    lime = int(np.count_nonzero((hue >= RYOBI_HUE[0]) & (hue <= RYOBI_HUE[1])
                                & (sat >= COLOUR_S_MIN)))
    # RAW appearance of the COLOURED (non-dark, non-grey) pixels. This is what
    # tells us where the pink battery actually sits in hue space — the last
    # calibration found ZERO pink pixels because the assumed hue band was
    # wrong, and we had no way to see that.
    colourful = (sat >= 20) & (val >= 90)
    if np.count_nonzero(colourful) >= 30:
        hues = hue[colourful].astype(np.float32)
        # Hue is circular: measure around the dominant peak, not the raw mean.
        ang = hues * (2 * np.pi / 180.0)
        peak = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                      % (2 * np.pi)) * 180.0 / (2 * np.pi))
        h_lo = float(np.percentile(hues, 10))
        h_hi = float(np.percentile(hues, 90))
        s_med = float(np.median(sat[colourful]))
        v_med = float(np.median(val[colourful]))
    else:
        peak = h_lo = h_hi = s_med = v_med = float('nan')
    return {'pink': pink, 'dark': dark, 'white': white, 'lime': lime,
            'total': total, 'hue_peak': peak, 'hue_lo': h_lo, 'hue_hi': h_hi,
            'sat_med': s_med, 'val_med': v_med}


def split_point(low_group, high_group, name, fallback):
    """Midpoint between two groups, with an honest report of the gap."""
    if not low_group or not high_group:
        return fallback, f"{name}: not enough data -> kept {fallback}"
    lo_max, hi_min = max(low_group), min(high_group)
    if lo_max < hi_min:
        return (lo_max + hi_min) / 2.0, (
            f"{name}: clean gap {lo_max:.3f} .. {hi_min:.3f} "
            f"-> {(lo_max + hi_min) / 2.0:.3f}")
    # Overlapping: fall back to the midpoint of the two means and say so.
    mid = (float(np.mean(low_group)) + float(np.mean(high_group))) / 2.0
    return mid, (f"{name}: OVERLAP ({lo_max:.3f} >= {hi_min:.3f}) — this "
                 f"signal is weak here; using means midpoint {mid:.3f}")


def main():
    pipeline, queue = build_pipeline()
    pipeline.start()
    print(__doc__)
    samples = defaultdict(list)      # 'pink' / 'ryobi' -> [dict, ...]
    click = {'pt': None}
    selected = None
    message, message_until = '', 0.0
    window = 'SO101 BATTERY-ID CALIBRATION (click, then R or P · F=finish)'

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (mx, my)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    aborted = True
    try:
        while pipeline.isRunning():
            now = time.monotonic()
            group = queue.tryGet()
            if group is not None:
                image = group['rgb'].getCvFrame()
                depth = group['depth'].getFrame().astype(np.float32) / 1000.0
                cands = [c for c in detect_candidates(image, depth)
                         if c['kind'] == 'target']
                if click['pt'] is not None and cands:
                    cx, cy = click['pt']
                    selected = min(cands, key=lambda c: (c['u'] - cx) ** 2
                                   + (c['v'] - cy) ** 2)
                    click['pt'] = None
                elif click['pt'] is not None:
                    click['pt'] = None
                    message, message_until = 'Nothing detected there', now + 2.0
                # Keep the selection alive across frames by position.
                if selected is not None and cands:
                    selected = min(cands, key=lambda c: (c['u'] - selected['u']) ** 2
                                   + (c['v'] - selected['v']) ** 2)
                for c in cands:
                    x, y, w, h = c['bbox']
                    is_sel = selected is not None and c['u'] == selected['u']
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (255, 255, 255) if is_sel else (0, 200, 255),
                                  3 if is_sel else 1)
                    cv2.putText(image,
                                f"{max(c['width_m'], c['height_m']) * 1000:.0f}mm",
                                (x, max(12, y - 6)), 0, 0.45, (0, 200, 255), 1)
                np_, nr_ = len(samples['pink']), len(samples['ryobi'])
                cv2.putText(image,
                            f"PINK={np_}  RYOBI={nr_}  (need {MIN_PER_CLASS} each)",
                            (8, 24), 0, 0.55, (0, 255, 0), 2)
                cv2.putText(image, 'CLICK a battery, then R (ryobi) or P (pink)'
                            + ('  |  F=finish' if min(np_, nr_) >= MIN_PER_CLASS
                               else ''),
                            (8, 48), 0, 0.48, (0, 255, 0), 2)
                if now < message_until:
                    cv2.putText(image, message, (8, 72), 0, 0.5, (0, 0, 255), 2)
                cv2.imshow(window, image)
                last_image, last_sel = image, selected

            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key in (ord('r'), ord('p'), ord('R'), ord('P')):
                if selected is None:
                    message, message_until = 'Click a battery first', now + 2.0
                else:
                    s = sample(last_image, selected['bbox'])
                    if s is None:
                        message, message_until = 'Bad sample', now + 2.0
                    else:
                        s['len_m'] = max(selected['width_m'],
                                         selected['height_m'])
                        cls = 'ryobi' if key in (ord('r'), ord('R')) else 'pink'
                        samples[cls].append(s)
                        ratio = s['pink'] / max(1, s['dark'] + s['white'])
                        print(f"{cls:5s}  len={s['len_m'] * 1000:5.0f}mm  "
                              f"pink={s['pink']:5d} dark={s['dark']:5d} "
                              f"white={s['white']:5d} lime={s['lime']:5d}  "
                              f"ratio={ratio:.2f}  "
                              f"HUE~{s['hue_peak']:.0f} "
                              f"({s['hue_lo']:.0f}-{s['hue_hi']:.0f}) "
                              f"S={s['sat_med']:.0f} V={s['val_med']:.0f}")
                        message = f'{cls.upper()} sample recorded'
                        message_until = now + 1.2
            if key in (ord('f'), ord('F')):
                if min(len(samples['pink']), len(samples['ryobi'])) >= MIN_PER_CLASS:
                    aborted = False
                    break
                message, message_until = (
                    f'Need {MIN_PER_CLASS} of each', now + 2.0)
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass

    if aborted:
        print('\nAborted — nothing written.')
        return

    pink_len = [s['len_m'] for s in samples['pink']]
    ryobi_len = [s['len_m'] for s in samples['ryobi']]
    pink_ratio = [s['pink'] / max(1, s['dark'] + s['white'])
                  for s in samples['pink']]
    ryobi_ratio = [s['pink'] / max(1, s['dark'] + s['white'])
                   for s in samples['ryobi']]
    lime_pink = float(np.mean([s['lime'] / s['total'] for s in samples['pink']]))
    lime_ryobi = float(np.mean([s['lime'] / s['total']
                                for s in samples['ryobi']]))

    print('\n' + '=' * 68)
    # ---- Measured appearance, so a wrong hue band cannot hide again --------
    def _stat(cls, key):
        vals = [s[key] for s in samples[cls] if np.isfinite(s[key])]
        return float(np.median(vals)) if vals else float('nan')

    p_hue, r_hue = _stat('pink', 'hue_peak'), _stat('ryobi', 'hue_peak')
    p_lo = float(np.nanmin([s['hue_lo'] for s in samples['pink']]))
    p_hi = float(np.nanmax([s['hue_hi'] for s in samples['pink']]))
    p_sat = _stat('pink', 'sat_med')
    print(f"MEASURED HUE   pink ~{p_hue:.0f} (band {p_lo:.0f}-{p_hi:.0f}, "
          f"S~{p_sat:.0f})   ryobi ~{r_hue:.0f}")
    print(f"CONFIGURED     PINK_HUE={PINK_HUE} / {PINK_HUE2}, "
          f"PINK_S_MIN={PINK_S_MIN}")
    pink_px_seen = sum(s['pink'] for s in samples['pink'])
    if pink_px_seen == 0:
        print('!! The pink batteries produced ZERO pink pixels with the\n'
              '!! CONFIGURED band. Widening it from the MEASURED hue above.')

    # ---- DARKNESS: the signal that actually separates these two types -----
    pink_dark = [s['dark'] / s['total'] for s in samples['pink']]
    ryobi_dark = [s['dark'] / s['total'] for s in samples['ryobi']]
    dark_split, dark_msg = split_point(pink_dark, ryobi_dark,
                                       'DARK fraction', DARK_SPLIT_FRAC)
    print(dark_msg)
    print(f"   pink  dark_frac {min(pink_dark):.3f} .. {max(pink_dark):.3f}")
    print(f"   ryobi dark_frac {min(ryobi_dark):.3f} .. {max(ryobi_dark):.3f}")

    len_split, len_msg = split_point(pink_len, ryobi_len,
                                     'LENGTH (m)', RYOBI_MIN_LEN_M)
    ratio_split, ratio_msg = split_point(ryobi_ratio, pink_ratio,
                                         'PINK/RYOBI pixel ratio',
                                         PINK_VS_RYOBI_W)
    print(len_msg)
    print(ratio_msg)

    # ---- GUARDS: never write a threshold that makes a class impossible ----
    warnings_out = []
    if not (0.15 <= ratio_split <= 3.0):
        warnings_out.append(
            f'ratio {ratio_split:.3f} out of usable range -> kept 0.45 '
            '(0.0 would make ONE pink pixel decide the sort)')
        ratio_split = 0.45
    if not (0.070 <= len_split <= 0.110):
        warnings_out.append(
            f'length split {len_split:.3f} m implausible for a '
            '115mm vs 80mm pair -> kept 0.090')
        len_split = 0.090
    if not (0.03 <= dark_split <= 0.60):
        warnings_out.append(
            f'dark split {dark_split:.3f} out of usable range -> kept 0.15')
        dark_split = 0.15
    # Length only earns a vote if it genuinely separated the two classes.
    size_usable = bool(pink_len and ryobi_len and max(pink_len) < min(ryobi_len))
    if not size_usable:
        warnings_out.append(
            'LENGTH overlaps between the two types (the detector often '
            'measures a fragment) -> USE_SIZE_OVERRIDE stays False')
    for w in warnings_out:
        print('GUARD: ' + w)
    # Lime is only useful if it actually differs between the two classes.
    lime_useful = abs(lime_ryobi - lime_pink) > 0.05
    print(f"LIME: ryobi {lime_ryobi:.3f} vs pink {lime_pink:.3f} -> "
          + ('usable' if lime_useful else 'NOT usable (green surface) — off'))
    print('=' * 68)

    with open(OUT_PATH, 'w') as f:
        f.write('# Written by calibrate_colour.py '
                + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(f'# {len(samples["pink"])} pink + {len(samples["ryobi"])} '
                'ryobi samples, measured in situ.\n')
        f.write(f'# {len_msg}\n# {ratio_msg}\n')
        for w in warnings_out:
            f.write(f'# GUARD: {w}\n')
        f.write('# DARKNESS is the primary discriminator (Ryobi is black,\n'
                '# the pink cells are pale). Hue proved useless on this rig.\n')
        f.write(f'DARK_SPLIT_FRAC = {dark_split:.3f}\n')
        f.write(f'USE_SIZE_OVERRIDE = {size_usable}\n')
        f.write(f'RYOBI_MIN_LEN_M = {len_split:.4f}\n')
        f.write(f'PINK_VS_RYOBI_W = {ratio_split:.3f}\n')
        f.write(f'LIME_AS_RYOBI = {lime_useful}\n')
        if pink_px_seen == 0 and np.isfinite(p_lo) and np.isfinite(p_hi):
            # The configured band missed the battery entirely — replace it
            # with the MEASURED one (padded), and drop the saturation floor
            # to whatever this battery actually shows.
            lo, hi = int(max(0, p_lo - 8)), int(min(179, p_hi + 8))
            f.write(f'# Measured pink hue ~{p_hue:.0f}; configured band found '
                    'zero pink pixels, so it is replaced here.\n')
            f.write(f'PINK_HUE = ({lo}, {hi})\n')
            f.write(f'PINK_S_MIN = {max(12, int(p_sat * 0.6))}\n')
        else:
            min_pink_px = min(s['pink'] for s in samples['pink'])
            f.write(f'PINK_MIN_PX = {max(40, int(min_pink_px * 0.6))}\n')
            min_pink_frac = min(s['pink'] / s['total'] for s in samples['pink'])
            f.write(f'PINK_MIN_FRAC = {max(0.02, min_pink_frac * 0.6):.3f}\n')
    print(f'\nWrote {OUT_PATH}')
    print('Run main.py — the new thresholds load automatically.')


if __name__ == '__main__':
    main()
