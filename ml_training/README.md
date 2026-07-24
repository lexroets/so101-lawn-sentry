# MobileNetV2 battery classifier — training workflow

Nothing here touches the demo code. `collect_crops.py` lives in
`SO101_ACTIVE/` so its imports resolve, but it is a standalone script that
**never opens the serial port** — the arm cannot move while it runs.

## What this replaces

Two things that have repeatedly misbehaved, with one small model:

| Currently | After |
|---|---|
| Colour by darkness fraction (works, but fragile — hue proved useless, and it needed a mat-specific calibration) | Learned from your actual batteries |
| Orientation — six hand-written attempts, still not trusted | Learned from your actual batteries |
| Phantom locks on empty mat, patched with a pixel-count gate | Explicit `background` class |

The reason to expect this to work where the heuristics didn't: every failure
so far was "no single hand-designed measurement separates these". That is
exactly the case learned features handle.

## 1. Collect (30–40 min, needs the rig)

```
cd SO101_ACTIVE
python collect_crops.py          # or run it in Thonny
```

Click a battery, press the label key, repeat. Then press `b` and pan across
empty mat and the buckets for a minute to bank background.

| Key | Label |
|---|---|
| `1` | ryobi_upright |
| `2` | ryobi_lying |
| `3` | pink_upright |
| `4` | pink_lying |
| `5` | background |
| `b` | background sweep on/off |
| `u` | undo last burst |
| `q` | quit |

**Targets:** ~150 per battery class, ~300 background.

**Pose variety beats volume.** 150 crops of one battery in one spot teaches
almost nothing. Between bursts: move it, rotate it, change its distance
(25–40 cm), turn it end-on, show the white text side, part-occlude it with
your hand. The end-on lying pose is the one every heuristic got wrong, so
deliberately over-collect it.

Each key press captures a burst of 5 across successive frames, so you get
sensor noise variation for free.

## 2. Train (~5 min)

Google Colab is the path of least resistance — torch is preinstalled, and
you skip the Python 3.14 wheel problem entirely (torch has no 3.14 wheels
at time of writing).

```
zip -r dataset.zip ml_training/dataset
```

New Colab notebook → upload `dataset.zip` → `!unzip -q dataset.zip` →
paste `train_mobilenetv2.py` into a cell → run. Download `battery_cls.onnx`
and `labels.txt`.

Locally instead: a Python 3.11/3.12 venv with `pip install torch torchvision onnx`.

## 3. Read the result honestly

The script prints per-class recall, a confusion matrix, and this:

```
ORIENTATION CONFUSION — the thing this model exists to fix:
  ryobi_upright <-> ryobi_lying: 3.1%  -> SOLVED
  pink_upright <-> pink_lying: 14.8%  -> NOT GOOD ENOUGH
```

Under 5% means the orientation problem is genuinely finished. Over ~12% and
the honest answer is more data on the confusing pose, not a threshold tweak
— that is the mistake I made repeatedly with the hand-written signals.

## 4. Deployment (after the demo)

Host-side is easiest and completely sufficient. Orientation only needs
deciding **once per pick**, during CENTRE — roughly 10–20 crops per cycle,
not every candidate every frame. That is a couple of milliseconds and it
does not touch the DepthAI pipeline at all.

Two options:

- **ONNX Runtime on the Mac** — `pip install onnxruntime`, feed the same
  bbox crop the detector already produces.
- **On-camera** — convert at <https://tools.luxonis.com> for RVC2 and run it
  as a second network. Avoids host dependencies, more setup.

Integration is small and additive: in the CENTRE voting loop, replace
`classify_colour(...)` with a model call returning colour **and**
orientation, and let `background` veto a lock. `AUTO_ORIENT` then flips to
`True` on a signal that is actually trustworthy, and the `L` key stays as a
manual override.

## Notes

- **No vertical flip in augmentation.** Flipping upside-down teaches the
  model that upright and lying look alike, destroying the distinction being
  trained. Horizontal flip is fine.
- The loss is class-weighted, so a large background class won't drown out
  the rarer battery poses.
- `crops.csv` records depth, box size and aspect per crop, if you later want
  to check whether a geometric signal separates the classes after all.
