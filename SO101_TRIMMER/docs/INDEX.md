# SO-101 Trimmer — start here

Autonomous lawn/edge trimmer built on the (proven) battery-sorter framework:
**detect → centre the base → reach → act.** Self-contained; does not touch
`SO101_ACTIVE/`. Four controlled joints (base, shoulder, elbow, wrist-pitch);
no roll, no gripper — a 3D-printed trimmer is on the wrist.

## Requirements
`pyserial`, `depthai` (v3), `opencv-python`, `numpy`. `onnxruntime` optional
(auto-used if present; otherwise OpenCV DNN runs the model). Training is done
in Colab, so no `torch` needed locally (Python 3.14 safe).

## Which script does what

| Script | Purpose |
|--------|---------|
| `calibrate_reach.py` | **Teach the flat "cut the top" reach** against real targets (2-step: lock depth, place head). Writes `trimmer_cut.py`. |
| `reach_object.py` | Move the head to a detected object's top. `edge` arg → vertical edging (flat pose + 90° wrist). Live tune (`o/l` height, `k/j` angle), `p` saves. Click to select. |
| `reach_trim.py` / `reach_trim_edge.py` | Persisted fine-trims (written by `p`). |
| `collect_grass.py` | Label grass tiles (uncut/cut/other) from the camera. |
| `make_synth_grass.py` | **No camera?** Procedurally generate a synthetic grass dataset to bootstrap the classifier. Writes `grass_dataset/`. |
| `train_grass_colab.py` | Paste into Colab to train the MobileNetV2 grass classifier. |
| `grass_view.py` | Live preview of grass detection (works with no model yet). |
| `grass_infer.py` | ONNX tile classifier; `python grass_infer.py` = self-test. |
| `grass_detector.py` | Fuses CNN + GLCM texture + depth → uncut regions. |
| `main_grass.py` | **Autonomous grass trimming** — detect → centre → level cut → sweep → next. Camera-crash resilient. |
| `main_trimmer.py` | Older depth-only autonomous loop (superseded by `main_grass.py`). |
| `camera_session.py` | Auto-restart wrapper for the crash-prone OAK-D. |
| `calibrate_trimmer.py` / `run_trimmer.py` | Phase-1: teach + replay a sweep by hand (no camera). |

More detail: `README.md` (phases + level-cut geometry), `GRASS_README.md`
(the three-signal grass recogniser and how it builds on prior research).

## Poop Patrol (camera top piece instead of the trimmer)

A separate mode: the arm sweeps its base side to side, scans the lawn for dog
poop, and logs where it is. Same driver, camera pipeline and crash-recovery.
Start with `POOP_README.md`. Detection runs on colour+depth with no model;
sharpens once `poop_cls.onnx` is trained.

| Script | Purpose |
|--------|---------|
| `calibrate_scan.py` | Teach the camera look-down pose. |
| `calibrate_sweep.py` | Teach the side-to-side base sweep range + station count. |
| `scan_poop.py` | **Sweep + detect + log** dog poop, with a ground-position map. |
| `poop_view.py` | Live detector preview (no arm), threshold tuning. |
| `make_synth_poop.py` / `fetch_poop_dataset.py` / `collect_poop.py` | Training data: synthetic bootstrap / real ScatSpotter download / camera capture. |
| `train_poop_colab.py` / `poop_infer.py` | Train the poop classifier / verify the export. |

## Fastest path to a working trimmer

1. **Reach** — `python calibrate_reach.py`, then `python reach_object.py`;
   tune height/angle live and press `p`. (Edge: `reach_object.py edge`.)
2. **See grass** — `python grass_view.py`. GLCM + depth already find the
   coarse/tall regions with **no training**. Tune the threshold with `[` `]`.
3. **Train (optional, sharper)** — get tiles into `grass_dataset/` either by
   `collect_grass.py` (real, best) **or** `make_synth_grass.py` (synthetic
   bootstrap, no camera needed) → zip → Colab (`train_grass_colab.py`) → drop
   `grass_cls.onnx` + `grass_labels.txt` here → `python grass_infer.py` to
   verify the export → `grass_view.py` again. Add real photos to
   `grass_dataset/` later and retrain to close the synthetic→real gap.
4. **Run it** — `python main_grass.py` (ENTER = auto).

## Key config knobs (`trimmer_config.py`)
- Reach/motion: `MOVE_SPEED/ACC`, `CENTRE_*`, `TRIM_SWEEP_*`, `REACH_TRIM*`,
  `EDGE_WRIST_OFFSET_RAW` (90° = 1024; flip sign if it rotates the wrong way).
- Detection: `DETECT_Z_MIN/MAX_M` (**camera dead-zone < 0.18 m**),
  `GRASS_UNCUT_CONF`, `GRASS_TILE_PX`/`GRASS_STRIDE_PX`, `GRASS_FUSE_DEPTH`.

## Known hardware note
The OAK-D Lite browns out on bare laptop USB and crashes. Use the Y-power
cable or a powered hub. `camera_session.py` auto-restarts it and parks the arm
during the gap, but power is the real fix.
