# SO-101 Poop Patrol — scan a lawn, find dog poop, log where it is

A new mode on the same arm + same OAK-D Lite: swap the trimmer head for a
camera-carrying top piece and the arm becomes a **scanner**. It holds a fixed
"look down at the lawn" pose and rotates the **base** side to side through a
set of yaw stations; at each station it detects dog poop on the grass and logs
the estimated ground position so you (or a mower) can avoid / clean it.

This reuses the proven framework — the same serial driver (`trimmer_arm.py`),
the same OAK-D pipeline (`trim_detector.build_pipeline`), and the same
crash-resilient `camera_session.py`. Nothing from the trimmer or the battery
sorter is touched.

## Detection — works now, sharper after training

Poop on grass is a **compact blob that isn't grass-green**: brown, dark and/or
desaturated, usually standing slightly proud of the blades. `poop_detector.py`
fuses three cheap, robust signals (same recipe as the grass detector):

1. **Colour** — "non-green blob on green" in HSV (brown / dark / desaturated)
2. **Shape** — a compact blob of plausible on-ground size (fills its bbox)
3. **Depth** — sits slightly above the fitted ground plane (optional)

That runs with **no trained model**, so you can practise-run and collect data
today. When `poop_cls.onnx` is present, a MobileNetV2 tile classifier
(poop / grass / other) is fused in to reject brown-but-not-poop distractors
(leaves, sticks, dirt, stones, shadows).

## Fastest path

1. **Aim the camera** — `python calibrate_scan.py`: torque releases, pose the
   arm by hand so the OAK-D looks at the grass, press ENTER. (Skippable — it
   falls back to the trimmer rest pose, but a real look-down view is better.)
2. **Check detection** — `python poop_view.py`. Put a brown object on some
   grass; it should get boxed. Tune the score threshold with `[` `]`.
3. **Scan + log** — `python scan_poop.py`. SPACE starts the sweep. Each find
   prints "~0.4 m ahead, 0.2 m right" and is written to `poop_logs/`.
4. **Train (sharper detection)** — get tiles into `poop_dataset/` (see below),
   zip, run `train_poop_colab.py` in Colab, drop `poop_cls.onnx` +
   `poop_labels.txt` here, verify with `python poop_infer.py`, re-run.

## Getting training data

Three routes into `poop_dataset/{poop,grass,other}/` — mix freely:

- **Synthetic (now, no camera)** — `python make_synth_poop.py`. Procedurally
  renders poop blobs, plain grass, and brown-but-not-poop hard negatives. A
  bootstrap; it won't fully transfer to real lawns.
- **Real public images** — `python fetch_poop_dataset.py` (run on your own
  machine; needs open internet). Pulls the openly-licensed **ScatSpotter**
  dataset (9000+ annotated dog-poop photos, CC-BY-4.0) and crops labelled
  tiles. This is the real-world data.
- **Your own camera** — `python collect_poop.py`. Click tiles to label them
  (`1` poop, `2` grass, `3` other; `f` saves a whole background frame).

Best results: fetch the real set, add a few dozen of your own via
`collect_poop.py`, optionally mix in synthetic, then train.

## What gets logged

`poop_logs/session_<stamp>/`:
- `detections.csv` / `.jsonl` — time, base angle, pixel, depth, estimated
  ground x/y, bearing, confidence, snapshot filename.
- `snap_XXX.jpg` — the annotated frame for each detection.
- `summary.txt` — count, a plain-English list ("~0.4 m ahead, 0.2 m right"),
  and a top-down ASCII map (`A` = base, `#` = poop).

Ground position = base yaw + in-frame bearing + depth range. It's an estimate
(camera tilt isn't modelled) — good enough to steer around. If left/right comes
out mirrored, flip `BASE_DIRECTION_SIGN` handling by negating the base term.

## Key config (`poop_config.py`)

- **Scan**: `SCAN_BASE_MIN/MAX_RAW`, `SCAN_STATIONS`, `SCAN_SETTLE_S`,
  `SCAN_FRAMES_PER_STATION`, `SCAN_LOOPS` (0 = one pass, -1 = patrol forever).
- **Detection**: `GRASS_HUE_*` (green to exclude), `POOP_*` colour bands,
  `POOP_MIN/MAX_AREA_PX`, `POOP_MIN_FILL`, `POOP_SCORE_MIN`, depth `POOP_*_PROUD_M`.
- **Depth window**: `DETECT_Z_MIN_M` (OAK-D dead-zone < 0.18 m), `DETECT_Z_MAX_M`.
- **ML**: `POOP_TILE_PX`, `POOP_CNN_WEIGHT`, `POOP_CNN_CONF`.

## Files

| Script | Purpose |
|--------|---------|
| `calibrate_scan.py` | Teach the camera look-down pose → `poop_scan_pose.py`. |
| `scan_poop.py` | **Main practice run** — sweep, detect, log. |
| `poop_view.py` | Live detector preview, no arm; tune thresholds. |
| `poop_detector.py` | Colour + depth + (optional) CNN → poop detections. |
| `poop_infer.py` | ONNX tile classifier; `python poop_infer.py` = self-test. |
| `poop_log.py` | CSV/JSONL + snapshots + summary map; position estimate. |
| `make_synth_poop.py` | Synthetic bootstrap dataset. |
| `fetch_poop_dataset.py` | Download real ScatSpotter images → tiles (run locally). |
| `collect_poop.py` | Label tiles from your camera. |
| `train_poop_colab.py` | Colab cell to train the classifier. |

Data credit: ScatSpotter / "shitspotter" by Jon Crall (CC-BY-4.0),
<https://github.com/Erotemic/shitspotter>.
