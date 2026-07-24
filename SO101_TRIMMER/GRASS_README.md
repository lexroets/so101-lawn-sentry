# Uncut-grass recognition for the trimmer arm (OAK-D Lite)

Recognises **uncut** grass against a field of **cut** grass so the arm can
target it. Built to be usable immediately and to get better with training.

## How it works — three fused signals (grounded in prior work)

Robotic-mower research converges on three ways to tell uncut from cut grass;
this system fuses all three instead of betting on one:

| Signal | Prior art | Here |
|--------|-----------|------|
| **Texture (classical)** | GLCM cut/uncut discrimination, GA-tuned | `_glcm_contrast` per tile — coarser = uncut. **Needs no training.** |
| **Texture (learned)** | CNN lawn segmentation (Uni Bremen ~7500-img dataset) | MobileNetV2 tile classifier, your proven pipeline |
| **Height** | photo-interrupter grass-height sensors | depth "proud of the mown deck" (already used in `trim_detector`) |

The frame is tiled; each tile is scored uncut by whatever signals are
available, scores are smoothed over time and grouped into regions. Because
GLCM + depth work with no model, **detection runs on day one**; training the
CNN sharpens it.

## Files

- `collect_grass.py` — label tiles (uncut / cut / other) from the camera
- `train_grass_colab.py` — MobileNetV2 trainer (paste into Colab)
- `grass_infer.py` — ONNX tile inference (cv2.dnn, no torch, Py3.14-safe)
- `grass_detector.py` — the three-signal fusion → uncut regions
- `grass_view.py` — live preview + threshold tuning

## Workflow

**1. Try it now (no training).**
```
python grass_view.py
```
GLCM + depth already box the coarser/taller regions as "uncut". Tune the
threshold live with `[` and `]`. This alone may be enough for a first pass.

**2. Collect (sharper).**
```
python collect_grass.py
```
Aim at an all-uncut area, press `1` (bulk-labels the frame); all-cut → `2`;
non-grass → `3`. Press `b` and pan for continuous capture. Target ~600 uncut,
~600 cut, ~400 other, across lighting / grass types / camera angles.

**3. Train (Colab).**
```
zip -r grass_dataset.zip grass_dataset
```
Paste `train_grass_colab.py` into a Colab cell, upload the zip, run. Download
`grass_cls.onnx` + `grass_labels.txt` into this folder. Re-run `grass_view.py`
— it now says `CNN+GLCM+depth`.

## Wiring it into the arm

`GrassDetector().detect(image, depth)` returns the same
`{u, v, z, area, bbox}` targets that `reach_object.py` / `main_trimmer.py`
already consume. To trim grass instead of batteries, swap the detector call:
`detect_patches(...)` → `GrassDetector().detect(...)`. Then it's the proven
loop — detect → centre → reach to level height → sweep — pointed at grass.

## Tuning (all in trimmer_config.py)

- `GRASS_UNCUT_CONF` — how sure a tile must be to count as uncut (also `[`/`]`)
- `GRASS_TILE_PX` / `GRASS_STRIDE_PX` — patch size / overlap (finer vs faster).
  Keep TILE equal between collection and inference.
- `GRASS_FUSE_DEPTH` — require the region to also stand proud of the deck
- `GRASS_MIN_REGION_TILES` — ignore tiny specks
- `GRASS_TEMPORAL` — de-flicker strength

## Notes

- Grass has no up/down, so the trainer uses vertical flips too (unlike the
  battery orientation task) — free extra augmentation.
- Backbone is FROZEN in training (the lesson from the battery model that
  memorised at 100% val). Expect a healthy 0.88–0.96, not 1.000.
- Depth on thin blades is noisy, so it's a soft boost, not a hard gate — the
  texture signals carry most of the decision.
