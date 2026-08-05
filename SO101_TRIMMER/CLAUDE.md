# SO-101 Lawn Sentry / Trimmer — working notes

Two robotics projects on one SO-101 arm with an OAK-D Lite. Read this before
changing anything; most of it is here because something went wrong first.

## Run it

```bash
conda activate sentry                 # NOT base, and not Thonny's Python
python launch.py                      # the lawn demo: dashboard + live scan
python calibrate.py                   # one-run setup (marker, rest pose, joints)
python tools/demo_check.py            # 30-second pre-flight, says GO or what to fix
python trimmer/reach_object.py        # the trimmer arm: reach a target and cut
python trimmer/tune_detector.py       # tune the depth detector, arm never moves
```

Interpreter is `/opt/anaconda3/envs/sentry/bin/python` (3.12). Python 3.14 has
no `onnxruntime` wheel, which forces the detector onto a `cv2.dnn` fallback
several times slower — that is why the env exists.

## Layout and imports

`core/` is shared. `lawn_sentry/` is waste detection. `trimmer/` is the earlier
grass-cutting robot. `archive/` is superseded and imported by nothing.

Every module starts with a block that finds `paths.py` and calls
`paths.bootstrap()`, which puts every code folder on `sys.path`. That is why
`import poop_config` works from any depth without packaging, and why files can
move between folders freely. **The folder list is discovered, not hardcoded** —
an earlier version named the folders and broke every import in a file the
moment it moved. Keep that block when adding a file.

`archive/` is deliberately off the import path: it holds older versions of live
modules and having both importable invites the wrong one being picked up.

## Verifying a change

Always, before saying something works:

```bash
python -m py_compile $(find . -name '*.py' -not -path '*/__pycache__/*')
```

then import every module with the hardware packages stubbed. `py_compile` only
checks syntax — it will happily pass a file with a missing import. Several real
bugs here were found by the import sweep and none by reading the code.

## The calibration model — the thing that took longest

A ground homography is true for **one camera pose only**. The camera is on the
arm, so "camera pose" means "joint angles". Every calibration therefore records
the arm's rest pose alongside the plane, and `launch.py` drives the arm back
there before scanning. Calibrations without a rest pose are only right by luck.

Things that were tried and failed, so they are not tried again:

- **Visual tracking alone** dead-reckons and slides off the lawn. Chained
  frame-to-frame homographies compound: 8 px at 5° pan, 537 px at 45°.
- **Encoder-only geometry** assumed the camera rotates about its optical
  centre. It does not — it is on the END of the arm and swings through an arc,
  12–30 cm of translation at 50° pan. Simulated perfectly, failed on the bench.
- **A 4-point homography from one marker** has eight free parameters all fitted
  from one small square, so half a pixel of corner noise lands in the
  perspective term. 122 cm of error at the far corners with a 25 cm marker.

What works: `solvePnP` with `SOLVEPNP_IPPE_SQUARE` and the OAK-D's known
intrinsics, averaged over 40 frames. Same marker, same noise: 10 cm → 1.7 cm.
`SOLVEPNP_IPPE_SQUARE` is strict about corner order — see
`marker_pose.object_points`, it is in one function for a reason.

Drift is cancelled by re-anchoring whenever the arm passes its origin, twice
per sweep. The note carries an **expiry, not a delay**: acting on a late one
anchors the plane to wherever the sweep has got to since.

Never overwrite a calibration directly. Go through `cal_store.save()`, which
archives the outgoing one. A known-good calibration was lost to a bad one once.

## Sign conventions, which have been wrong twice

`shift_px` reports how the **content** moved. The camera moves the opposite way:

    camera pans RIGHT  -> everything slides LEFT -> dx < 0
    camera tilts DOWN  -> everything slides UP   -> dy < 0

Stated once in `calibrate.py:probe_joints`. A verification step drives each
role and checks the picture agrees — note that when that check used the same
wrong convention it happily confirmed its own error.

## Speeds and units

The survey sweep rate is **degrees per second, computed by the loop** —
`SWEEP_DEG_PER_S` in `arm_control.py`. It used to be a servo speed register,
and the Feetech datasheets disagree about that register's unit by a factor of
fifty, which is how the sweep ended up racing. Do not delegate a rate to the
servo unless the unit is verified.

## Keys and input

`cv2.waitKey` reports which key is DOWN, not when it was pressed, so acting on
the value acts repeatedly while it is held. One tap of ENTER used to toggle
auto-mode five times. Use `keys.Keys` — it edge-detects. `hit()` fires once per
press; `tap()` also repeats while held, for trims you nudge by feel.

Time debounces do not work here: too short and a held key repeats, too long and
a double-tap is swallowed, and one shared timer across keys makes SPACE block
ENTER — which reads as "the buttons are unresponsive".

VS Code's **debug console cannot accept `input()`**. Nine scripts ask
questions. Every launch config sets `"console": "integratedTerminal"`; keep
that in any you add.

## Detector settings

`POOP_YOLO_CONF = 0.25`, `POOP_YOLO_TILES = 3`. Measured on real frames: poop
scores 0.45–0.54, best clutter 0.14, so the threshold sits in the gap. The old
0.45 sliced through the middle of the real detections. Tiling matters most for
the further target: 0.04 untiled, 0.22 at 2×2, 0.45 at 3×3.

Detection runs **off the main loop** (`detect_worker.py`). Inline it stalled
the video — 5.3 fps published vs 27.7. Boxes are carried forward through the
ground tracker to where the objects are now, because async results describe a
frame a few hundred ms old and while panning the boxes trail badly.

## The trimmer arm

`reach_object.py` maps a measured depth to a **taught** arm pose. It only
interpolates; outside the taught range it clamps, which reads as the arm
under-reaching by a fixed amount with nothing to explain it. It now says so.
**Do not extrapolate the table** — a hand-taught pose has no defined behaviour
beyond its points and will drive the head into the bench.

The current table has a mis-taught point: between 0.323 and 0.350 m the
shoulder drops while the elbow opens, so the head moves BACK as the target gets
further away. `reach_table.report()` flags it at startup. Re-teach with
`trimmer/calibrate/calibrate_reach.py`, including a point past the real working
distance.

The OAK-D cannot measure closer than ~0.19 m — below that it wraps and reads
FURTHER. Readings below `MIN_RELIABLE_Z` are rejected.

The cut is: land on top, drop `CUT_DROP_RAW`, then sweep the **base only**. The
head is flat and the base axis is vertical, so the tip holds one height through
the arc — a level cut with no kinematics. `REMEMBER_CLEARED` is off for now.

## Safety

`core/cutter.py` is off unless something actively keeps it on: watchdog on a
keepalive, hard run limit, `atexit`, context manager, and `stop()` never
raises. The failure that matters is not "it did not start", it is "it did not
stop". Backend is `'sim'` until the motor wiring is known.

## Git

`main` has everything. `lawn-sentry` and `trimmer` are `main` minus the other
project. **Never merge a project branch back into `main`** — they are defined
by what they delete, so merging both deleted both projects and gutted main.
Refresh them from main, never the other way.

## CAD

`cad/trimmer_head.py` builds STLs with trimesh + manifold3d. Subtract cutters
**one at a time** — concatenating them makes a self-intersecting mesh and the
boolean returns something that looks right with hundreds of broken faces.
Verify with `is_watertight`, `broken_faces`, and by probing the finished solid;
a bore placed by its centre instead of its floor came out half depth and looked
perfect in the render.
