"""Logging for the poop scanner: where was poop found, so the owner avoids it.

Each detection is converted from (base yaw + in-frame bearing + depth range)
into an estimated ground position relative to the arm base, de-duplicated
across scan stations, and written to:

    poop_logs/session_<stamp>/detections.csv    machine-readable, appendable
    poop_logs/session_<stamp>/detections.jsonl  same, one JSON object per line
    poop_logs/session_<stamp>/snap_XXX.jpg       annotated frame per detection
    poop_logs/session_<stamp>/summary.txt        count + top-down ASCII map

Position model (base frame, metres): forward = +Y, right = +X.
    bearing = base_yaw_deg + (u - cx)/W * HFOV
    x = range * sin(bearing),   y = range * cos(bearing)
Range uses the measured depth; it's an estimate (camera tilt not modelled),
good enough to say "poop ~0.4 m ahead and a bit to the right."
"""
import os
import csv
import json
import math
import time

import cv2

import poop_config as cfg

MERGE_DIST_M = 0.12          # detections closer than this = the same poop


def _base_deg(base_raw):
    return (base_raw - cfg.BASE_FORWARD_RAW) / cfg.RAW_PER_DEG


def ground_position(base_raw, u, z):
    """Estimated (x, y, bearing_deg) on the ground, base frame, metres."""
    bearing = _base_deg(base_raw) + (u - cfg.CAMERA_WIDTH / 2.0) \
        / cfg.CAMERA_WIDTH * cfg.CAMERA_HFOV_DEG
    r = float(z) if z is not None else float('nan')
    br = math.radians(bearing)
    return r * math.sin(br), r * math.cos(br), bearing


class PoopLog:
    def __init__(self):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(cfg.POOP_LOG_DIR, f'session_{stamp}')
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, 'detections.csv')
        self.jsonl_path = os.path.join(self.dir, 'detections.jsonl')
        self._csv = open(self.csv_path, 'w', newline='')
        self._w = csv.writer(self._csv)
        self._w.writerow(['time', 'station_raw', 'base_deg', 'u', 'v', 'z_m',
                          'bearing_deg', 'x_m', 'y_m', 'area_px', 'score',
                          'cnn', 'snapshot'])
        self._jsonl = open(self.jsonl_path, 'w')
        self.finds = []          # deduped: list of dicts
        self._snap_n = 0

    def _merge(self, x, y):
        for f in self.finds:
            if math.isfinite(x) and math.isfinite(f['x_m']) \
                    and math.hypot(x - f['x_m'], y - f['y_m']) < MERGE_DIST_M:
                return f
        return None

    def add(self, det, base_raw, image=None):
        """Record one detection. Returns (is_new, find). Merges near-duplicates,
        keeping the higher-scoring observation."""
        x, y, bearing = ground_position(base_raw, det['u'], det['z'])
        existing = self._merge(x, y)
        snap = ''
        if cfg.POOP_SNAPSHOTS and image is not None:
            snap = f'snap_{self._snap_n:03d}.jpg'
            self._snap_n += 1
            frame = image.copy()
            bx, by, bw, bh = det['bbox']
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 90, 230), 2)
            cv2.putText(frame, f"poop {det['score']:.2f}  ~{det['z'] or 0:.2f}m",
                        (bx, max(14, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 90, 230), 2)
            cv2.imwrite(os.path.join(self.dir, snap), frame)
        row = {'time': time.strftime('%H:%M:%S'), 'station_raw': int(base_raw),
               'base_deg': round(_base_deg(base_raw), 1),
               'u': det['u'], 'v': det['v'],
               'z_m': None if det['z'] is None else round(det['z'], 3),
               'bearing_deg': round(bearing, 1),
               'x_m': round(x, 3) if math.isfinite(x) else None,
               'y_m': round(y, 3) if math.isfinite(y) else None,
               'area_px': int(det['area']), 'score': round(det['score'], 3),
               'cnn': round(det['cnn'], 3) if 'cnn' in det else '',
               'snapshot': snap}
        self._w.writerow([row[k] for k in
                          ('time', 'station_raw', 'base_deg', 'u', 'v', 'z_m',
                           'bearing_deg', 'x_m', 'y_m', 'area_px', 'score',
                           'cnn', 'snapshot')])
        self._csv.flush()
        self._jsonl.write(json.dumps(row) + '\n'); self._jsonl.flush()
        if existing is None:
            f = {'x_m': x, 'y_m': y, 'bearing_deg': bearing,
                 'z_m': det['z'], 'score': det['score'], 'best_snap': snap}
            self.finds.append(f)
            return True, f
        if det['score'] > existing['score']:
            existing.update(score=det['score'], best_snap=snap or existing['best_snap'])
        return False, existing

    def _ascii_map(self, w=41, h=17, reach=1.2):
        """Top-down map: base at bottom-centre, forward = up."""
        grid = [[' '] * w for _ in range(h)]
        cx = w // 2
        for j in range(w):
            grid[h - 1][j] = '-'
        grid[h - 1][cx] = 'A'          # arm base
        for f in self.finds:
            if not (math.isfinite(f['x_m']) and math.isfinite(f['y_m'])):
                continue
            gx = int(round(cx + f['x_m'] / reach * (w // 2)))
            gy = int(round((h - 1) - f['y_m'] / reach * (h - 1)))
            if 0 <= gx < w and 0 <= gy < h:
                grid[gy][gx] = '#'
        return '\n'.join(''.join(r) for r in grid)

    def finalize(self):
        path = os.path.join(self.dir, 'summary.txt')
        with open(path, 'w') as f:
            f.write(f"Poop scan summary — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Distinct detections: {len(self.finds)}\n\n")
            for i, d in enumerate(sorted(self.finds, key=lambda d: d['y_m']
                                         if math.isfinite(d['y_m']) else 9)):
                f.write(f"  #{i + 1}: ~{d['y_m']:.2f} m ahead, "
                        f"{abs(d['x_m']):.2f} m {'right' if d['x_m'] >= 0 else 'left'}"
                        f"  (bearing {d['bearing_deg']:+.0f} deg, "
                        f"score {d['score']:.2f})\n")
            f.write("\nTop-down map (A = base, # = poop, forward = up):\n\n")
            f.write(self._ascii_map() + "\n")
        return path

    def close(self):
        try:
            self._csv.close(); self._jsonl.close()
        except Exception:
            pass
