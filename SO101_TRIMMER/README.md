# SO-101 Lawn Sentry

Autonomous dog-waste detection on an SO-101 arm with an OAK-D Lite camera.
The camera scans a lawn, a trained YOLO detector finds waste on the grass, and
a RYOBI-branded web dashboard plots each target on a live 2D map of the lawn
with a video feed alongside.

## Run the demo

```bash
python launch.py
```

That starts the web dashboard and the live scan together and opens the browser.
Press `q` in the scan window to finish.

First time only — tell it where the lawn is:

```bash
python calibrate/calibrate_lawn_scan.py
```

Scan around and click the four lawn corners; the whole lawn does not need to
fit in one view. Enter the real dimensions when asked (3.5 x 2.1 m by default).

If the video feed does not appear:

```bash
python tools/diag_feed.py
```

## Layout

```
launch.py          the entry point — start here
paths.py           where everything lives; import this instead of using __file__

core/              shared modules imported by everything else
                   poop_config, poop_detector, poop_yolo, ground_track,
                   lawn_shape, camera_session, trim_detector, trimmer_*, grass_*
calibrate/         one-off setup: lawn corners, arm poses, sweep range
run/               the things that drive the robot and camera
                   scan_lawn (the demo), scan_poop, poop_view, main_grass,
                   main_trimmer, run_trimmer
dashboard/         lawn_map_app.py — the Streamlit web dashboard
training/          dataset building, labelling and model training
tools/             diagnostics and tests: diag_feed, debug_range, test_*
docs/              the longer write-ups

data/              INPUTS  — poop_yolo.onnx, lawn_calibration.json, datasets
output/            OUTPUTS — scan_sessions, poop_logs  (regenerated each run)
```

## How the pieces fit

`launch.py` starts `dashboard/lawn_map_app.py` under Streamlit, then runs
`run/scan_lawn.py`. The scan writes each annotated frame to
`output/scan_sessions/session_*/live.jpg` and its findings to `session.json`.
The dashboard serves both to the browser over a small local endpoint on port
8799 — video as an MJPEG stream, data as JSON — so the map and the video
refresh independently of Streamlit.

Detection runs through `core/poop_detector.py`, which uses the trained model in
`data/poop_yolo.onnx` when present and falls back to a colour heuristic when
not. Camera motion is tracked by `core/ground_track.py` so target positions
stay correct on the map while the camera pans.

## Imports and paths

Every script starts with a short block that adds the project folders to
`sys.path`, so flat imports (`import poop_config`) keep working from any
subfolder. Data locations come from `paths.py` rather than `__file__` — that is
what lets a file move between folders without losing track of its data.

To add a new script, copy that block from any existing file and use
`from paths import DATA, OUTPUT` for anything it reads or writes.

## Accuracy notes

- Calibration assumes the marked lawn region is one flat plane.
- Detection position uses the bottom-centre of the bounding box, which
  approximates the ground-contact point.
- Ground tracking keeps world positions correct while the camera pans, but
  re-run calibration if the camera's height or tilt changes.
