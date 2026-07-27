"""
poop_view.py

Stationary OAK-D poop detector with accurate session logging.

This keeps the same working camera and PoopDetector pipeline as the original
poop_view.py, but adds:

- multi-frame confirmation to reject flickering detections
- robust depth measurement from the centre of each bounding box
- duplicate suppression across the full session
- camera-relative range, bearing and horizontal position
- CSV and JSONL logs
- one annotated snapshot per confirmed object
- live terminal output
- summary.txt when the program closes

IMPORTANT:
The reported positions are relative to the fixed camera:
    range_m       = measured OAK-D depth
    bearing_deg   = horizontal angle from image centre
    right_m       = horizontal displacement from image centre
    forward_m     = forward component of the measured depth

They are not global lawn coordinates because the mower position, camera height
and camera pitch are not currently available.

Keys:
    [       lower score threshold
    ]       raise score threshold
    s       save the current annotated frame
    m       write/print an interim summary
    r       clear active tracking only
    q       quit and write final summary
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

import csv
import json
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

import poop_config as cfg
from trim_detector import build_pipeline
from camera_session import CameraSession
from poop_detector import PoopDetector


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WINDOW = "SO-101 stationary poop detector"

# ---------------------------------------------------------------------------
# Tracking and logging settings
# ---------------------------------------------------------------------------

# Number of detector hits needed before an object is logged.
CONFIRM_HITS = 3

# Confirmation must happen within this period.
CONFIRM_WINDOW_S = 1.5

# Remove an active track if it is not seen for this long.
TRACK_TIMEOUT_S = 2.0

# Maximum image-centre movement for detections to belong to the same track.
TRACK_MATCH_PX = 65.0

# Maximum depth difference when associating detections.
TRACK_MATCH_DEPTH_M = 0.20

# Suppress later detections close to an object already logged.
LOGGED_MATCH_PX = 90.0
LOGGED_MATCH_DEPTH_M = 0.25

# Minimum interval between running expensive logging/confirmation work.
# The detector itself still runs on every received frame.
TRACK_UPDATE_INTERVAL_S = 0.03

# Use the centre portion of the detection box for robust depth measurement.
DEPTH_ROI_FRACTION = 0.45

# Number of valid depth pixels required for a reliable estimate.
MIN_DEPTH_SAMPLES = 12

# Camera-relative offsets. Leave at zero unless measured physically.
CAMERA_RIGHT_OFFSET_M = 0.0
CAMERA_FORWARD_OFFSET_M = 0.0


def finite_number(value):
    """Return True only for a valid finite numeric value."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp(value, low, high):
    return max(low, min(high, value))


def robust_depth_from_bbox(depth_m, bbox, fallback=None):
    """
    Estimate object depth from the central region of its bounding box.

    A median is more robust than reading one centre pixel because stereo depth
    can contain holes and isolated noisy values.
    """
    if depth_m is None or depth_m.size == 0:
        return fallback if finite_number(fallback) else None

    x, y, w, h = [int(v) for v in bbox]
    image_h, image_w = depth_m.shape[:2]

    if w <= 0 or h <= 0:
        return fallback if finite_number(fallback) else None

    roi_w = max(5, int(w * DEPTH_ROI_FRACTION))
    roi_h = max(5, int(h * DEPTH_ROI_FRACTION))

    cx = x + w // 2
    cy = y + h // 2

    x0 = clamp(cx - roi_w // 2, 0, image_w)
    x1 = clamp(cx + roi_w // 2 + 1, 0, image_w)
    y0 = clamp(cy - roi_h // 2, 0, image_h)
    y1 = clamp(cy + roi_h // 2 + 1, 0, image_h)

    roi = depth_m[y0:y1, x0:x1].astype(np.float32, copy=False)

    valid = roi[
        np.isfinite(roi)
        & (roi >= cfg.DETECT_Z_MIN_M)
        & (roi <= cfg.DETECT_Z_MAX_M)
    ]

    if valid.size < MIN_DEPTH_SAMPLES:
        return fallback if finite_number(fallback) else None

    # First median, then remove large outliers using median absolute deviation.
    median = float(np.median(valid))
    absolute_deviation = np.abs(valid - median)
    mad = float(np.median(absolute_deviation))

    if mad > 0:
        filtered = valid[absolute_deviation <= max(3.0 * mad, 0.025)]
        if filtered.size >= MIN_DEPTH_SAMPLES:
            median = float(np.median(filtered))

    return median


def camera_position(u, z_m, image_width):
    """
    Convert image position and depth into camera-relative coordinates.

    The OAK-D horizontal field of view is used to estimate a pinhole-camera
    horizontal bearing.

    Returns:
        bearing_deg, right_m, forward_m
    """
    bearing_deg = (
        (float(u) - image_width / 2.0)
        / float(image_width)
        * float(cfg.CAMERA_HFOV_DEG)
    )

    if not finite_number(z_m):
        return bearing_deg, None, None

    bearing_rad = math.radians(bearing_deg)
    z_m = float(z_m)

    right_m = z_m * math.sin(bearing_rad) + CAMERA_RIGHT_OFFSET_M
    forward_m = z_m * math.cos(bearing_rad) + CAMERA_FORWARD_OFFSET_M

    return bearing_deg, right_m, forward_m


def detection_centre(det):
    """Return detection centre, preferring the detector's centroid."""
    if "u" in det and "v" in det:
        return float(det["u"]), float(det["v"])

    x, y, w, h = det["bbox"]
    return x + w / 2.0, y + h / 2.0


def distance_px(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def depth_compatible(a, b, tolerance):
    """
    Missing depth does not prevent image-space association.
    If both depths exist, require them to be reasonably close.
    """
    if not finite_number(a) or not finite_number(b):
        return True
    return abs(float(a) - float(b)) <= tolerance


class AccuratePoopLogger:
    """
    Temporal tracker and session logger.

    Active tracks confirm detections over several frames. Confirmed detections
    are compared against all previously logged objects to avoid repeatedly
    logging the same stationary object.
    """

    def __init__(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = os.path.join(
            cfg.POOP_LOG_DIR,
            f"stationary_session_{stamp}",
        )
        os.makedirs(self.directory, exist_ok=True)

        self.csv_path = os.path.join(self.directory, "detections.csv")
        self.jsonl_path = os.path.join(self.directory, "detections.jsonl")
        self.summary_path = os.path.join(self.directory, "summary.txt")

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")

        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "id",
                "timestamp",
                "elapsed_s",
                "score",
                "hits",
                "u_px",
                "v_px",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "range_m",
                "bearing_deg",
                "right_m",
                "forward_m",
                "depth_samples",
                "snapshot",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

        self.started_monotonic = time.monotonic()
        self.started_datetime = datetime.now()

        self.active_tracks = []
        self.logged_objects = []

        self.next_track_id = 1
        self.next_object_id = 1
        self.manual_frame_number = 0

    def _new_track(self, det, now):
        u, v = detection_centre(det)

        track = {
            "track_id": self.next_track_id,
            "first_seen": now,
            "last_seen": now,
            "hits": 1,
            "u": float(u),
            "v": float(v),
            "z": det.get("z"),
            "best_det": dict(det),
            "best_score": float(det.get("score", 0.0)),
            "logged": False,
        }

        self.next_track_id += 1
        self.active_tracks.append(track)
        return track

    def _find_active_track(self, det):
        centre = detection_centre(det)
        z = det.get("z")

        best_track = None
        best_distance = float("inf")

        for track in self.active_tracks:
            d_px = distance_px(
                centre,
                (track["u"], track["v"]),
            )

            if d_px > TRACK_MATCH_PX:
                continue

            if not depth_compatible(
                z,
                track["z"],
                TRACK_MATCH_DEPTH_M,
            ):
                continue

            if d_px < best_distance:
                best_distance = d_px
                best_track = track

        return best_track

    def _matches_logged_object(self, track):
        centre = (track["u"], track["v"])

        for obj in self.logged_objects:
            if distance_px(centre, (obj["u"], obj["v"])) > LOGGED_MATCH_PX:
                continue

            if not depth_compatible(
                track["z"],
                obj["range_m"],
                LOGGED_MATCH_DEPTH_M,
            ):
                continue

            return obj

        return None

    def _update_track(self, track, det, now):
        u, v = detection_centre(det)
        score = float(det.get("score", 0.0))

        # Exponential smoothing reduces bounding-box jitter.
        alpha = 0.35
        track["u"] = (1.0 - alpha) * track["u"] + alpha * float(u)
        track["v"] = (1.0 - alpha) * track["v"] + alpha * float(v)

        z = det.get("z")
        if finite_number(z):
            if finite_number(track["z"]):
                track["z"] = (
                    (1.0 - alpha) * float(track["z"])
                    + alpha * float(z)
                )
            else:
                track["z"] = float(z)

        track["hits"] += 1
        track["last_seen"] = now

        if score >= track["best_score"]:
            track["best_score"] = score
            track["best_det"] = dict(det)

    def _write_confirmed_object(self, track, image):
        det = dict(track["best_det"])
        x, y, w, h = [int(value) for value in det["bbox"]]

        u = int(round(track["u"]))
        v = int(round(track["v"]))
        z = float(track["z"]) if finite_number(track["z"]) else None

        bearing_deg, right_m, forward_m = camera_position(
            u,
            z,
            image.shape[1],
        )

        object_id = self.next_object_id
        self.next_object_id += 1

        snapshot_name = f"poop_{object_id:03d}.jpg"
        snapshot_path = os.path.join(self.directory, snapshot_name)

        snapshot = image.copy()

        cv2.rectangle(
            snapshot,
            (x, y),
            (x + w, y + h),
            (0, 90, 230),
            3,
        )

        if z is None:
            position_text = f"bearing {bearing_deg:+.1f} deg"
        else:
            position_text = (
                f"{z:.2f} m range, "
                f"{abs(right_m):.2f} m "
                f"{'right' if right_m >= 0 else 'left'}"
            )

        cv2.putText(
            snapshot,
            f"poop {track['best_score']:.2f}",
            (x, max(18, y - 25)),
            _FONT,
            0.55,
            (0, 90, 230),
            2,
        )

        cv2.putText(
            snapshot,
            position_text,
            (x, max(18, y - 7)),
            _FONT,
            0.48,
            (0, 255, 255),
            2,
        )

        cv2.imwrite(snapshot_path, snapshot)

        now_datetime = datetime.now()
        elapsed_s = time.monotonic() - self.started_monotonic

        row = {
            "id": object_id,
            "timestamp": now_datetime.isoformat(timespec="milliseconds"),
            "elapsed_s": round(elapsed_s, 3),
            "score": round(track["best_score"], 4),
            "hits": track["hits"],
            "u_px": u,
            "v_px": v,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "range_m": None if z is None else round(z, 4),
            "bearing_deg": round(bearing_deg, 3),
            "right_m": None if right_m is None else round(right_m, 4),
            "forward_m": None if forward_m is None else round(forward_m, 4),
            "depth_samples": "robust_bbox_median",
            "snapshot": snapshot_name,
        }

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.jsonl_file.write(json.dumps(row) + "\n")
        self.jsonl_file.flush()

        obj = {
            "id": object_id,
            "u": float(u),
            "v": float(v),
            "range_m": z,
            "bearing_deg": bearing_deg,
            "right_m": right_m,
            "forward_m": forward_m,
            "score": track["best_score"],
            "hits": track["hits"],
            "snapshot": snapshot_name,
        }
        self.logged_objects.append(obj)

        if z is None:
            print(
                f"[POOP #{object_id}] "
                f"bearing={bearing_deg:+.1f} deg, "
                f"pixel=({u}, {v}), "
                f"score={track['best_score']:.2f}, "
                f"confirmed={track['hits']} frames"
            )
        else:
            print(
                f"[POOP #{object_id}] "
                f"range={z:.2f} m, "
                f"bearing={bearing_deg:+.1f} deg, "
                f"{abs(right_m):.2f} m "
                f"{'right' if right_m >= 0 else 'left'}, "
                f"forward-component={forward_m:.2f} m, "
                f"score={track['best_score']:.2f}, "
                f"confirmed={track['hits']} frames"
            )

        print(f"           snapshot: {snapshot_path}")

        return obj

    def update(self, detections, depth_m, image):
        """
        Update tracking from one detector frame.

        Returns objects that became newly confirmed during this update.
        """
        now = time.monotonic()
        newly_logged = []

        # Remove stale unconfirmed tracks.
        self.active_tracks = [
            track
            for track in self.active_tracks
            if now - track["last_seen"] <= TRACK_TIMEOUT_S
        ]

        for original_det in detections:
            det = dict(original_det)

            # Replace a potentially noisy centre-pixel depth with a robust
            # central bounding-box median whenever possible.
            det["z"] = robust_depth_from_bbox(
                depth_m,
                det["bbox"],
                fallback=det.get("z"),
            )

            track = self._find_active_track(det)

            if track is None:
                track = self._new_track(det, now)
            else:
                self._update_track(track, det, now)

            age = now - track["first_seen"]

            if (
                not track["logged"]
                and track["hits"] >= CONFIRM_HITS
                and age <= CONFIRM_WINDOW_S + TRACK_TIMEOUT_S
            ):
                duplicate = self._matches_logged_object(track)

                if duplicate is None:
                    obj = self._write_confirmed_object(track, image)
                    newly_logged.append(obj)

                track["logged"] = True

        return newly_logged

    def annotate_tracks(self, image):
        """Draw confirmation state and logged-object positions."""
        now = time.monotonic()

        for track in self.active_tracks:
            if now - track["last_seen"] > TRACK_TIMEOUT_S:
                continue

            det = track["best_det"]
            x, y, w, h = [int(value) for value in det["bbox"]]

            confirmed = track["logged"]
            colour = (0, 90, 230) if confirmed else (0, 200, 255)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                colour,
                3 if confirmed else 2,
            )

            state = (
                "logged"
                if confirmed
                else f"confirming {track['hits']}/{CONFIRM_HITS}"
            )

            cv2.putText(
                image,
                f"{state}  score={track['best_score']:.2f}",
                (x, max(16, y - 7)),
                _FONT,
                0.48,
                colour,
                2,
            )

            if finite_number(track["z"]):
                bearing, right_m, _ = camera_position(
                    track["u"],
                    track["z"],
                    image.shape[1],
                )

                cv2.putText(
                    image,
                    (
                        f"{float(track['z']):.2f}m  "
                        f"{bearing:+.1f}deg  "
                        f"{abs(right_m):.2f}m "
                        f"{'R' if right_m >= 0 else 'L'}"
                    ),
                    (x, min(image.shape[0] - 8, y + h + 18)),
                    _FONT,
                    0.46,
                    colour,
                    1,
                )

    def save_manual_frame(self, image):
        filename = f"manual_frame_{self.manual_frame_number:03d}.png"
        self.manual_frame_number += 1

        path = os.path.join(self.directory, filename)
        cv2.imwrite(path, image)
        print(f"[saved] {path}")

    def clear_active_tracks(self):
        self.active_tracks.clear()
        print("[tracker] active confirmation tracks cleared")

    def write_summary(self):
        duration_s = time.monotonic() - self.started_monotonic

        with open(self.summary_path, "w", encoding="utf-8") as summary:
            summary.write("Stationary poop detection summary\n")
            summary.write("=" * 36 + "\n")
            summary.write(
                f"Started: {self.started_datetime.isoformat(timespec='seconds')}\n"
            )
            summary.write(
                f"Finished: {datetime.now().isoformat(timespec='seconds')}\n"
            )
            summary.write(f"Duration: {duration_s:.1f} seconds\n")
            summary.write(
                f"Distinct confirmed detections: {len(self.logged_objects)}\n"
            )
            summary.write(
                f"Final score threshold: {cfg.POOP_SCORE_MIN:.2f}\n\n"
            )

            if not self.logged_objects:
                summary.write("No confirmed detections were logged.\n")
            else:
                for obj in self.logged_objects:
                    if finite_number(obj["range_m"]):
                        summary.write(
                            f"#{obj['id']}: "
                            f"range {obj['range_m']:.2f} m, "
                            f"bearing {obj['bearing_deg']:+.1f} deg, "
                            f"{abs(obj['right_m']):.2f} m "
                            f"{'right' if obj['right_m'] >= 0 else 'left'}, "
                            f"score {obj['score']:.2f}, "
                            f"confirmed over {obj['hits']} hits, "
                            f"snapshot {obj['snapshot']}\n"
                        )
                    else:
                        summary.write(
                            f"#{obj['id']}: "
                            f"bearing {obj['bearing_deg']:+.1f} deg, "
                            f"score {obj['score']:.2f}, "
                            f"confirmed over {obj['hits']} hits, "
                            f"snapshot {obj['snapshot']}\n"
                        )

            summary.write(
                "\nCoordinates are relative to the fixed camera, not global "
                "lawn or mower coordinates.\n"
            )

        return self.summary_path

    def close(self):
        try:
            self.csv_file.close()
        finally:
            self.jsonl_file.close()


def main():
    cam = CameraSession(build_pipeline)
    det = PoopDetector()
    logger = AccuratePoopLogger()

    src = (
        "colour+depth+CNN"
        if det.clf.ready
        else "colour+depth (no model yet)"
    )

    print(f"[poop] stationary preview with {src}")
    print(f"[poop] logs: {logger.directory}")
    print(
        "[poop] a detection is logged after "
        f"{CONFIRM_HITS} matching observations"
    )
    print("[keys] [ ] threshold  s save  m summary  r reset tracks  q quit")

    last_tracking_update = 0.0

    try:
        while True:
            frame_group = cam.get()

            if frame_group is None:
                if cam.crashed():
                    print("[camera] stopped, restarting")

                    if not cam.restart():
                        print("[camera] restart failed")
                        break

                if cv2.waitKey(1) & 255 == ord("q"):
                    break

                continue

            image = frame_group["rgb"].getCvFrame()

            depth = (
                frame_group["depth"]
                .getFrame()
                .astype(np.float32)
                / 1000.0
            )

            stats = {}
            detections = det.detect(image, depth, stats)

            now = time.monotonic()

            if now - last_tracking_update >= TRACK_UPDATE_INTERVAL_S:
                logger.update(detections, depth, image)
                last_tracking_update = now

            display = image.copy()
            logger.annotate_tracks(display)

            cv2.rectangle(
                display,
                (0, 0),
                (display.shape[1], 52),
                (18, 18, 18),
                -1,
            )

            cv2.putText(
                display,
                (
                    f"{src}  threshold={cfg.POOP_SCORE_MIN:.2f}  "
                    f"candidates={len(detections)}"
                ),
                (8, 20),
                _FONT,
                0.48,
                (0, 255, 255),
                1,
            )

            cv2.putText(
                display,
                (
                    f"confirmed={len(logger.logged_objects)}  "
                    f"tracks={len(logger.active_tracks)}  "
                    "[ ] threshold  s save  m summary  q quit"
                ),
                (8, 43),
                _FONT,
                0.46,
                (0, 220, 255),
                1,
            )

            cv2.imshow(_WINDOW, display)

            key = cv2.waitKey(1) & 255

            if key == ord("q"):
                break

            if key == ord("["):
                cfg.POOP_SCORE_MIN = max(
                    0.10,
                    cfg.POOP_SCORE_MIN - 0.05,
                )
                print(
                    f"[threshold] POOP_SCORE_MIN={cfg.POOP_SCORE_MIN:.2f}"
                )

            if key == ord("]"):
                cfg.POOP_SCORE_MIN = min(
                    0.95,
                    cfg.POOP_SCORE_MIN + 0.05,
                )
                print(
                    f"[threshold] POOP_SCORE_MIN={cfg.POOP_SCORE_MIN:.2f}"
                )

            if key == ord("s"):
                logger.save_manual_frame(display)

            if key == ord("m"):
                summary = logger.write_summary()
                print(f"[summary] {summary}")

            if key == ord("r"):
                logger.clear_active_tracks()

    finally:
        summary = logger.write_summary()

        print()
        print("=" * 58)
        print("STATIONARY POOP DETECTION COMPLETE")
        print("=" * 58)
        print(f"Confirmed objects : {len(logger.logged_objects)}")
        print(f"CSV log           : {logger.csv_path}")
        print(f"JSONL log         : {logger.jsonl_path}")
        print(f"Summary           : {summary}")
        print(f"Snapshots folder  : {logger.directory}")
        print("=" * 58)

        logger.close()
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == "__main__":
    main()