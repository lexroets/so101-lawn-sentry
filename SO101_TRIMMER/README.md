# Stationary OAK-D Lawn Map

## Install
```bash
pip install -r requirements.txt
```
Keep this folder beside the existing project modules (`poop_config.py`, `poop_detector.py`, `camera_session.py`, `trim_detector.py`). If you copy only these new files into the existing project folder, imports work directly.

## 1. Fix the elevated camera permanently
Do not move it after calibration.

## 2. Measure the lawn rectangle
Measure width and length in metres, then run:
```bash
python calibrate_lawn_map.py --width 4.0 --length 6.0
```
Click lawn corners TL, TR, BR, BL and press S.

## 3. Start scanning
```bash
python scan_lawn.py
```

## 4. Open the web app in another terminal
```bash
streamlit run lawn_map_app.py
```
The app reads `scan_sessions/` and shows the latest and previous sessions.

## Accuracy notes
- Calibration assumes the marked lawn region is one flat plane.
- Detection position uses the bottom-centre of the bounding box, which approximates the ground-contact point.
- A future segmentation mask should replace that point with the lower-centre of the mask.
- Recalibrate whenever the camera moves, zooms, or changes resolution.
