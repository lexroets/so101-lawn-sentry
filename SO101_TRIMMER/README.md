# SO-101 Lawn Sentry

Autonomous dog-waste detection on an SO-101 arm with an OAK-D Lite camera. The
camera surveys a lawn, a trained YOLO detector finds waste on the grass, and a
RYOBI-branded web dashboard plots each target on a live 2D map with the video
feed alongside.

Two projects share this repository because they share the hardware — the same
arm, the same camera, the same serial protocol. They live on separate branches
so each can be cloned and run on its own.

| Branch | What it is |
| --- | --- |
| `main` | everything: shared core plus both projects |
| `lawn-sentry` | the waste-detection product only |
| `trimmer` | the earlier grass-trimming robot only |

---

## Run the demo

```bash
python launch.py
```

Starts the dashboard and the live scan together and opens the browser. Press
`q` in the scan window to finish.

## Calibrate

The ground plane is what turns pixels into metres, and it is only true for the
camera pose it was measured at — so every calibrator records the arm's rest
pose alongside it, and `launch.py` drives the arm back there before scanning.

**With a printed marker** — the repeatable one, for anyone who did not build
the system:

```bash
python lawn_sentry/calibrate/calibrate_qr.py --print --paper a3
python lawn_sentry/calibrate/calibrate_qr.py
```

Lay the marker flat on the grass close to the camera, hold still while it
averages, then slide the projected rectangle onto the real lawn with the arrow
keys. Take the marker away afterwards — it is not needed again unless the
camera moves.

**By clicking the corners** — the original method, no marker needed:

```bash
python lawn_sentry/calibrate/calibrate_lawn_scan.py
```

Pan around and click the four corners; the whole lawn does not need to fit in
one view. Ground tracking stitches the views together.

**Which joint does what**, if the arm is remounted:

```bash
python lawn_sentry/calibrate/calibrate_joints.py
```

## When something looks wrong

```bash
python tools/calibrations.py                 # every calibration ever saved
python tools/calibrations.py restore 2       # go back to one that worked
python tools/fix_orientation.py              # is the lawn round the right way?
python tools/diag_feed.py                    # no video feed?
```

Nothing overwrites a calibration without archiving it first, so rolling back is
always available.

---

## Layout

```
launch.py            start the demo
paths.py             where everything lives, and the import bootstrap

core/                shared by both projects
    trimmer_arm.py         SO-101 serial driver
    trimmer_config.py      servo ids, limits, geometry
    camera_session.py      OAK-D pipeline
    frame_source.py        OAK-D / webcam / RTSP / file, one interface
    camera_intrinsics.py   K, from the device's factory calibration
    arm_control.py         pan/tilt roles, survey sweep, go home
    ground_track.py        frame-to-frame ground homography
    marker.py              ArUco detection and printable sheets
    marker_pose.py         ground plane from one marker sighting
    cal_store.py           load, normalise and archive calibrations

lawn_sentry/         the waste-detection product
    scan_lawn.py           the scan loop — detect, track, publish
    lawn_map_app.py        the dashboard (Streamlit + MJPEG)
    poop_detector.py       detector front end
    poop_yolo.py           YOLO inference, tiled
    poop_config.py         thresholds, with the measurements behind them
    lawn_shape.py          green-mask lawn outline
    calibrate/             lawn plane, rest pose, joint roles

trimmer/             the earlier grass-cutting robot
    main_trimmer.py, main_grass.py, grass_detector.py, ...
    calibrate/             its own setup routines

training/            datasets and model training
tools/               diagnostics and one-off fixes
archive/             superseded, kept for reference, imported by nothing
docs/                longer notes
data/                model, calibration, datasets       (inputs)
output/              scan sessions and logs             (generated, not committed)
```

### Imports

Every module starts with a short block that finds `paths.py` and calls
`paths.bootstrap()`, which puts every code folder on `sys.path`. That is why
`import poop_config` works from any depth without packaging the tree, and why
files can move between folders without breaking anything. The folder list is
discovered, not hardcoded — an earlier version named the folders explicitly and
broke every import in a file the moment it moved.

`archive/` is deliberately kept off the import path: it holds older versions of
modules that are still live under other names, and having both importable
invites the wrong one being picked up.
