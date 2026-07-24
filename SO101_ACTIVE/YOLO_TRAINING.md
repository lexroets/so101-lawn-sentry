# Training YOLO to recognise the Ryobi RB4L30 — and which way up it is

Reference product: RB4L30 4V USB Lithium 3.0Ah Battery & Power Bank
(stick-style, ~11-12 cm, lime green/black, USB-C port, fuel-gauge LED):
https://uk.ryobitools.eu/batteries-and-chargers/4v-usb-lithium-batteries-and-battery-chargers/4v-usb-lithium-batteries/rb4l30/rb4l30/

## Why two classes

The arm has two taught grab tables: a side grab for standing batteries and a
vertical top-down grab for lying ones (`lying_calibration.py`). Picking the
wrong one is the difference between a clean grab and driving the gripper into
the battery.

Deciding orientation from **depth** does not work on this hardware. Measured
on 2026-07-20, over five pickups:

| spread | h/w | proud | actually |
|---|---|---|---|
| 22mm | 0.68 | 22mm | lying |
| 67mm | 1.07 | 17mm | lying |
| 203mm | 1.20 | 4mm | lying |
| 240mm | 1.21 | 71mm | lying |
| 8mm | 1.43 | 267mm | **standing** |

Lying batteries spanned 22-240 mm; the standing one gave the smallest value of
all. `proud=267mm` on a 115 mm battery is impossible. The cause is geometric,
not a bad threshold: the bounding box is mostly desk, and the desk's own depth
gradient across the box is larger than a 23 mm-thick battery. Three separate
depth metrics (local desk height, row-median height, self-relative spread) all
failed the same way.

Appearance separates the two cases trivially — a standing RB4L30 shows its
circular top and vertical label, a lying one shows the cylinder side and
horizontal label. So let the network decide. **Label orientation as two
classes and the problem goes away.**

## 1. Collect images (~200-300)

Photograph YOUR battery on YOUR desk under the lighting you demo in.

Aim for a roughly **even split between upright and lying** — if 80% of your
images are upright, the model learns "upright" as the safe guess and you are
back where you started.

Vary within each class:

- **Distance** 25-40 cm, which is what the robot actually sees
- **Angle** around the battery: for lying, include end-on views (the hardest
  case, and the one the depth method most often got wrong); for upright,
  include the USB port toward and away from camera
- **Rotation** of a lying battery across the full 360° of headings on the desk
- Background clutter, and partial occlusion by a hand or another battery
- Include the pink batteries too, so the model learns they are *not* RB4L30

Add ~30 images with no battery at all. Negatives cut false alarms hard.

Screenshots from the OAK feed beat phone photos — same lens, same distortion,
same lighting as runtime. Press a screenshot key while `main.py` runs, or just
photograph the preview window.

## 2. Label in Roboflow (free tier is plenty)

1. roboflow.com → new project → type **Object Detection**
2. Upload, then draw one box per battery using exactly these class names:

   | Class | Use when |
   |---|---|
   | `battery_upright` | standing on its end, either way up |
   | `battery_lying` | resting on its side, any heading |
   | `battery_pink` | the smaller pink cells (optional but useful) |
   | `box` | your drop container (optional) |

   The names matter — `config.py` matches on the `_lying` and `_upright`
   suffixes, so any prefix works but the suffix must be exact.

3. **Borderline cases:** a battery leaning against something is `lying` if the
   gripper would need to come down on it. When in doubt, label `lying` — that
   grab is the more forgiving of the two.
4. Generate a dataset version. Augmentations: brightness ±25%, rotation ±15°,
   slight blur. **Do not enable vertical flip** — flipping upside-down teaches
   the model that upright and lying look alike, which destroys the very
   distinction you are training. Horizontal flip is fine.
5. Train with Roboflow Train, or export and run locally:

       yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=416

Check the confusion matrix before converting. You want upright↔lying confusion
under ~5%. If it is worse, the usual cause is too few end-on lying examples.

## 3. Convert for the OAK camera

The model runs ON the camera. Upload `best.pt` to https://tools.luxonis.com,
input size 416x416, target **RVC2**. You get an NNArchive (`.tar.xz`).

## 4. Wire it in — three lines

Put the archive next to `main.py`, then in `build_pipeline()`:

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        rgb, dai.NNArchive('rb4l30.tar.xz'))

`nn.getClasses()` already feeds `labels` through to `candidates_from_yolo`,
which attaches each detection's class name as `c['label']`. Nothing else in
the detection path changes.

Then in `config.py`:

    AUTO_ORIENT = True      # currently False
    YOLO_CONF = 0.5         # raise it; a specific model is confident

That is the whole integration. `main.py` already checks the label first and
falls back to depth only when there is no class name:

    if any(s in lbl for s in YOLO_LYING_LABELS):     -> lying,   src=yolo
    elif any(s in lbl for s in YOLO_UPRIGHT_LABELS): -> upright, src=yolo
    elif spread is not None:                          -> depth guess
    else:                                             -> aspect guess

YOLO-sourced frames vote unbiased; depth-sourced frames keep `LYING_VOTE_BIAS`.
The `L` key still force-overrides everything, so your manual fallback survives
regardless.

## 5. Verify before trusting it

Run `main.py` and watch the ORIENTATION line. It now names its source:

    ORIENTATION lying=12 upright=0 [src=yolo:battery_lying spread=203mm ...] -> LYING top-down

- `src=yolo:...` means the network decided. Good.
- `src=depth` means no class label arrived — the model is not loaded, the
  class names do not match the suffixes, or confidence is below `YOLO_CONF`.

Test 10 pickups with `AUTO_ORIENT = True` and `L` untouched, half lying and
half standing. If any standing battery gets `-> LYING top-down`, set
`AUTO_ORIENT = False` again and add training images of that pose before
retrying. A wrong top-down grab is the one failure mode worth avoiding.

## Tips

- Keep `CLASSICAL_FALLBACK = True` as a safety net for when YOLO drops a frame.
- The size classifier in `detector.py` still sorts everything into
  target/box by real physical size, so extra classes cost you nothing.
- Retraining is cheap once the Roboflow project exists — add images of any
  pose that fails and regenerate.
