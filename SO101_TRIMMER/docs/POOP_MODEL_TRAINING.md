# Training a reliable poop detector

Goal: a small, fast model that finds **real dog poop on a real lawn** and
ignores leaves, sticks, mulch, stones and shadows. It runs on the robot and
replaces the colour/box demo heuristic.

## The short version

Open Google Colab → **Runtime → Change runtime type → GPU** → paste all of
**`train_poop_real.py`** into one cell → run it → download `poop_yolo.onnx` →
put it in `SO101_TRIMMER/` → set `POOP_USE_YOLO = True` in `poop_config.py`.

That's it. Everything else below is context and options.

## What the notebook actually does

**1. Real photos with real labels.** It streams the public **ScatSpotter**
dataset (CC-BY-4.0, ~9,000 phone photos of dog poop with hand-drawn polygon
annotations). These are human labels, so they beat any auto-labelling — SAM is
not needed for them.

The dataset also labels the exact things that were causing your false
positives: `leaf`, `stick`, `mulch`, `rock`, `dirt`, `shadow`, `acorn`,
`pine-cone`, `wood-chip`. The notebook keeps only `poop` boxes and leaves those
distractors **unlabelled**, which actively teaches the model *"brown thing on
the ground ≠ poop"*. That is the single biggest reason this will be more
reliable than the colour heuristic.

**2. Your own lawn (optional but recommended).** If you upload
`sam_images.zip` (frames from `capture_frames.py`), **SAM 3.1** auto-labels
them and they're mixed into training, so the model also learns your specific
turf, camera height and lighting. Press Cancel at the upload prompt to skip.

**3. Label check.** Before training it displays 8 labelled images with the
boxes drawn. If those boxes aren't on poop, stop — don't waste the GPU time.

**4. Train + export.** YOLOv8-nano, ~80 epochs, then exports `poop_yolo.onnx`.
Expect **mAP50 > 0.70** for a usable detector.

Knobs at the top of the file: `N_REAL` (how many real images, default 1200 —
raise to 3000+ for a stronger model and longer runtime), `EPOCHS`, `IMG`.

## Deploying it

1. Put `poop_yolo.onnx` in `SO101_TRIMMER/`.
2. `pip install onnxruntime` (optional; ~2× faster than OpenCV DNN).
3. Test it before trusting it:
   ```
   python test_poop_model.py            # runs on real_poop/ or any folder
   ```
   This writes `model_test/` with boxes drawn, plus timing. Look at the images.
4. Set `POOP_USE_YOLO = True` in `poop_config.py`.
5. `python launch.py` — the robot now uses the learned model.

The YOLO detector returns the same boxes as the heuristic, so the lawn map,
dashboard, cleanup route and logs all work unchanged. Set
`POOP_USE_YOLO = False` any time to fall back to the demo detector.

## Tuning after deployment

| Symptom | Fix |
|---|---|
| False detections | raise `POOP_YOLO_CONF` (0.35 → 0.5) |
| Missing real poop | lower `POOP_YOLO_CONF` (0.35 → 0.25) |
| Misses *your* lawn specifically | capture frames with `capture_frames.py`, retrain including them |
| Model won't load | it prints why and falls back to colour automatically; re-download the file |

## Getting your own images

```
python capture_frames.py     # auto-saves a frame every ~0.7 s to real_poop/
python gather_images.py      # collects real photos into sam_images/
zip -r sam_images.zip sam_images
```
Move the subject around between shots — near/far, left/right, different angles
and light. 200–400 varied frames is plenty.

## Files

| File | Runs where | Purpose |
|------|-----------|---------|
| `train_poop_real.py` | **Colab (GPU)** | **The whole training job, one cell.** |
| `capture_frames.py` | Mac | Grab raw camera frames for training. |
| `gather_images.py` | Mac | Collect real photos into `sam_images/`. |
| `test_poop_model.py` | Mac | Verify the trained ONNX on real images. |
| `poop_yolo.py` | Mac | Runs the ONNX on the robot (already wired in). |
| `sam_autolabel.py` / `train_poop_yolo.py` | Colab | Older two-step SAM-only route; `train_poop_real.py` supersedes both. |

Data credit: ScatSpotter / "shitspotter" by Jon Crall, CC-BY-4.0 —
<https://github.com/Erotemic/shitspotter>
