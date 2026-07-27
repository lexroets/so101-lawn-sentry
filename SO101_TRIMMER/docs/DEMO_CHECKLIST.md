# Demo run-sheet — RYOBI Lawn Sentry

## When training finishes (2 commands)

```
python deploy_model.py ~/Downloads/poop_yolo.onnx
python launch.py
```

`deploy_model.py` installs the model, proves it loads and runs, times it,
tries it on your images, and checks the demo prerequisites. It prints
**READY** or tells you exactly what's wrong. No config editing — the robot
picks the model up automatically.

If the file is already in the `SO101_TRIMMER` folder, just `python deploy_model.py`.

## Before the boss walks over (5 min)

1. **Plug the OAK-D into a powered hub / Y-cable.** Bare laptop USB browns out
   mid-demo. This is the single most likely thing to embarrass you.
2. `python deploy_model.py` — confirm it says READY.
3. `python test_poop_model.py` — open `model_test/` and eyeball the boxes.
4. Check the lawn calibration still matches (camera must not have moved since
   `calibrate_lawn_map.py`). If it moved, re-run that — takes 60 seconds.
5. `python launch.py`, do one dry run, then press `q`.

## The demo itself

```
python launch.py
```

- Camera window opens immediately; the dashboard opens in your browser.
- Place a target on the lawn. Within ~2 s a pin drops on the map with a chime
  and a "NEW TARGET DETECTED" banner.
- **The money shot:** click the pin. The inspector shows the actual camera
  capture, coordinates, range from the unit and a confidence meter.
- Click **Route** — it draws the shortest cleanup path with total distance.
  That's the line that sells it: *detection becomes an action plan.*
- Move the target, place a second one — watch the map update live.
- Wrong detection? Hover its row → click **×**. It vanishes and stays gone.
- Press `q` in the camera window to end; the dashboard stays up for questions.

## Talking points

- **Real training data.** Trained on ~1,200 real photos with human-drawn
  labels, including deliberate hard negatives (leaves, sticks, mulch, stones,
  shadows) so it learns "brown ≠ poop".
- **Distilled for the edge.** SAM-class foundation models are the teacher; the
  deployed model is a YOLOv8-nano running in a few ms on the host — roughly
  200× lighter than running the foundation model itself.
- **Metric output, not just boxes.** A homography maps pixels to real lawn
  coordinates, so every detection is an x/y position in metres — which is what
  a mower can actually navigate to.

## If something breaks

| Problem | Fix |
|---|---|
| Camera window frozen / black | USB power. Replug into powered hub; `launch.py` auto-restarts the camera. |
| Too many false detections | raise `POOP_YOLO_CONF` in `poop_config.py` (0.35 → 0.5) |
| Missing obvious targets | lower `POOP_YOLO_CONF` (0.35 → 0.25) |
| Model misbehaving badly | set `POOP_USE_YOLO = False` — instantly back to the colour/box demo that already worked |
| Dashboard didn't open | `pip install streamlit pandas`, or browse to http://localhost:8501 |
| Pins in the wrong place | camera moved — re-run `python calibrate_lawn_map.py` |

**Panic button:** `POOP_USE_YOLO = False` in `poop_config.py` reverts to the
known-good colour detector without touching anything else. Keep that in your
back pocket.
