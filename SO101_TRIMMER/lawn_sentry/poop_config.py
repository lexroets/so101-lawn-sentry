"""Config for the SO-101 POOP-SCANNER phase.

Same arm + same OAK-D Lite as the trimmer, but a camera-carrying top piece
replaces the trimmer head. The job is different: instead of moving TO a target
and acting on it, the arm SCANS — it holds a fixed "look down at the lawn"
pose and rotates the BASE side to side, detecting dog poop on the grass at
each station and logging WHERE it is so the owner can avoid / clean it.

Bus, joints, camera and geometry constants are shared with the trimmer, so we
import them rather than duplicate (single source of truth). Everything below
is poop-specific.
"""

# --- put the project's folders on sys.path (see paths.py) --------------------
import sys as _sys, pathlib as _pl
_d = _pl.Path(__file__).resolve().parent
while not (_d / 'paths.py').is_file() and _d != _d.parent:
    _d = _d.parent
if str(_d) not in _sys.path:
    _sys.path.insert(0, str(_d))
import paths as _paths
_paths.bootstrap()
# -----------------------------------------------------------------------------
import os

# --- shared hardware constants (reuse the trimmer's) -----------------------
from trimmer_config import (              # noqa: F401  (re-exported for callers)
    SERIAL_PORT, BAUD_RATE,
    BASE_ID, SHOULDER_ID, ELBOW_ID, WRIST_ID, JOINT_IDS,
    MOVE_SPEED, MOVE_ACC, ARRIVE_TOL_RAW, ARRIVE_TIMEOUT_S, SETTLE_S,
    BASE_RAW_MIN, BASE_RAW_MAX, JOINT_RAW_MIN, JOINT_RAW_MAX,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    RAW_PER_PIXEL, BASE_FORWARD_RAW, CAMERA_LATENCY_S,
    MIN_RELIABLE_Z, REST_POSE,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- scan pose (camera looking down/forward at the lawn) --------------------
# The base sweeps; joints 2/3/4 hold this pose so the camera keeps a steady
# downward-forward view of the grass. Teach it with calibrate_scan.py (writes
# poop_scan_pose.py). Until then we fall back to the trimmer REST_POSE if it
# exists, else a sensible mid pose.
SCAN_POSE = None                          # {2:raw,3:raw,4:raw}; base excluded

# --- base sweep (the side-to-side scan) ------------------------------------
# The scan visits a set of base yaw "stations" spanning a safe arc, pausing at
# each to grab a stable frame. More stations = finer coverage but slower.
SCAN_BASE_MIN_RAW = 1150       # left end of the swept arc (inside BASE limits)
SCAN_BASE_MAX_RAW = 2840       # right end
SCAN_STATIONS = 9              # number of stops across the arc
SCAN_SPEED = 500               # gentle base slew between stations
SCAN_ACC = 18
SCAN_SETTLE_S = 0.6            # let the arm + image stabilise before detecting
SCAN_FRAMES_PER_STATION = 4    # average detections over a few frames (de-noise)
SCAN_LOOPS = 0                 # 0 = one pass then stop; N>0 = repeat N times;
                               # -1 = loop forever (patrol mode)

# --- depth working window (OAK-D Lite near-range dead-zone < 0.18 m) --------
DETECT_Z_MIN_M = 0.3              # floor is 0.19 m (OAK-D can't measure nearer)
DETECT_Z_MAX_M = 5.0              # effectively "no distance filter". This only
                                  # sets how far a RANGE READOUT is trusted —
                                  # map placement uses the homography — so
                                  # widening it cannot create false detections.

# How many pixels OUTSIDE the calibrated lawn quad a detection may still count.
# A poop resting on the far edge has its bounding-box CENTRE above the edge
# line, so a strict test throws it away even though the model saw it. We test
# the box's ground-contact point and allow this margin on top.
POOP_ROI_MARGIN_PX = 60
GROUND_TOL_M = 0.05               # +/- band around the fitted ground plane

# --- poop appearance (heuristic detector, works with NO trained model) -----
# Dog poop is a compact, DULL-BROWN blob on green. Rather than flag anything
# "not green" (which also catches grey stones, pale leaves, bright litter), we
# specifically require a BROWN / dark-brown colour: a warm hue, real colour
# saturation (rejects grey concrete/stones), and a dull mid-low brightness
# (rejects bright objects). Then it must be a compact blob and (optionally)
# stand slightly proud. Tune all of this live in poop_view.py.
#   OpenCV HSV: H 0-179, S 0-255, V 0-255.
GRASS_HUE_LO = 32             # green grass hue band (to EXCLUDE)
GRASS_HUE_HI = 95
GRASS_SAT_MIN = 40           # grass is reasonably saturated green

# What counts as "poop/box-brown". These are deliberately BROWN-BOX-FRIENDLY:
# a real cardboard box is often bright tan (high value) and only moderately
# saturated, so the ceilings are generous. The lawn-polygon ROI + "keep the
# single strongest blob" already stop off-lawn / multiple false hits, so we can
# afford a permissive colour gate here. Tighten POOP_VAL_MAX / raise
# POOP_MIN_BROWN_FRAC later when you switch from a box to real (dull) poop.
POOP_BROWN_HUE_LO = 3        # warm/brown/tan hue band, low end (H)
POOP_BROWN_HUE_HI = 38       # ..high end. Wrap-around reds (H>=168) also count.
POOP_HUE_WRAP = 168          # hues >= this are deep red/brown too
POOP_SAT_MIN = 30            # require SOME colour -> still rejects grey concrete
POOP_VAL_MIN = 18            # reject pure-black holes
POOP_VAL_MAX = 240           # allow bright tan cardboard (was rejecting the box)
POOP_DARK_VAL = 90           # a dark-brown blob may be dimmer but still warm...
POOP_DARK_SAT_MIN = 22       # ...and keep a little saturation (not pure grey)
POOP_MIN_BROWN_FRAC = 0.25   # blob must be at least this fraction brown/tan

POOP_MIN_AREA_PX = 350       # ignore specks (leaves, dirt flecks)
POOP_MAX_AREA_PX = 80000     # a near box is large; raised so it isn't culled
POOP_MIN_FILL = 0.40         # blob must fill this frac of its bbox (compactness)
POOP_FUSE_DEPTH = True       # depth is a BONUS only (never required) — a flat
POOP_MIN_PROUD_M = 0.005     # textureless box gives no stereo depth, so we must
POOP_MAX_PROUD_M = 0.20      # not depend on it to see the object
POOP_SCORE_MIN = 0.45        # min fused score (0..1) to report (lower = easier)
POOP_TEMPORAL_FRAMES = SCAN_FRAMES_PER_STATION   # must persist across frames

# ---- DEMO: match ONE specific object (the tan cardboard box) --------------
# For a quick demo you want ONLY the box to register — not the mower, wheels,
# QR code, etc. In this mode the detector ignores the general brown logic and
# matches a tight HSV band around the box's bright-tan colour. The mower is
# rejected automatically: dark body/tyres fail the brightness floor, white
# panels fail the saturation floor, the green handle is grass-hued.
# Run  calibrate_box.py  to click your real box and lock its exact colour
# (writes poop_box_ref.py). Set POOP_DEMO_BOX_MODE=False to go back to normal.
POOP_DEMO_BOX_MODE = True
POOP_BOX_HSV_LO = (8, 60, 120)    # OpenCV HSV lower bound of the tan box
POOP_BOX_HSV_HI = (30, 190, 245)  # ..upper bound (bright, warm, SATURATED tan)
POOP_BOX_MIN_AREA_PX = 400        # box blob size window (excludes the big mower
POOP_BOX_MAX_AREA_PX = 22000      # and tiny specks)
POOP_BOX_MIN_FILL = 0.50          # a box is a solid rectangle -> fills its bbox
# Hard safety floors applied to WHATEVER band is loaded (incl. a bad
# calibrate_box result). Cardboard tan is saturated and bright; the mower's
# cream/white 3D-printed parts are DESATURATED, its body/tyres are DARK — these
# floors reject both no matter what colour got saved.
POOP_BOX_SAT_FLOOR = 30           # rejects cream/white plastic (S~5-25) while
                                  # keeping PALE tan cardboard (S~40-60) — 55
                                  # was too strict and dropped the real box
POOP_BOX_VAL_FLOOR = 80           # never match dark body/tyres/shadow
POOP_BOX_MIN_RECT = 0.62          # blob must be rectangular (area / min-rect area);
                                  # the box is a clean rectangle, the mower isn't

# Keep only the strongest N detections per frame.
#   0 = UNLIMITED — find every poop on the lawn  <-- what you want now
#   1 = only the single strongest (was used for the one-cardboard-box demo;
#       leaving it at 1 makes the system look like it can only see one poop)
# The lawn-polygon ROI already discards anything off the grass, and the
# confidence threshold controls quality, so there is no need to cap the count.
POOP_MAX_TARGETS = 0

# ---- learned detector (distilled from SAM 3.1 -> YOLOv8) ------------------
# Once you've trained poop_yolo.onnx (sam_autolabel.py -> train_poop_yolo.py),
# flip POOP_USE_YOLO on. The detector then uses the LEARNED model instead of
# the colour/box heuristic, but returns the same boxes so everything
# downstream (map, dashboard, route) is unchanged. Falls back to the heuristic
# automatically if the ONNX file is missing.
# 'auto'  = use the trained model IF poop_yolo.onnx is present (recommended)
# True    = force the learned model (falls back to colour if it won't load)
# False   = force the colour/box demo heuristic
POOP_USE_YOLO = 'auto'
POOP_YOLO_MODEL = 'poop_yolo.onnx'
POOP_YOLO_IMG = 640          # must match the imgsz used at export
# Trained model v1 measured: mAP50 0.550, precision 0.626, recall 0.531.
#
# 0.45 was set from those training-set numbers, and on THIS lawn it was far too
# high. Measured on the 11 snapshots from the 29 Jul run, with 3x3 tiling:
#
#     real poop, near      0.49 - 0.54
#     real poop, further   0.45          <- sat exactly on the old threshold
#     clutter, best case   0.14          <- cork tiles, paper, the robot
#
# So the model does see the poop; the cut was slicing through the middle of the
# real detections, which is why one got pinned and the one beside it did not.
# The gap between real (0.45+) and clutter (0.14) is wide and empty, so the
# threshold belongs in the middle of it rather than at the top.
#
# Counting boxes per frame across all 11 snapshots shows where the noise starts:
#     0.45 -> 0.6/frame    0.30 -> 0.9    0.25 -> 0.9    0.20 -> 1.1
#     0.15 -> 1.8/frame  <- clutter floods in here
# 0.25 sits on the flat part, well clear of both edges. scan_lawn.py still
# requires CONFIRM_HITS=3 sightings at the same LAWN COORDINATE, which is what
# makes a low threshold safe: a stray box does not land on the same patch of
# grass three times, and a real poop does.
#   0.45 = the old setting, misses roughly half of them
#   0.25 = balanced                                    (current)
#   0.15 = catches everything, expect stray pins
POOP_YOLO_CONF = 0.25
POOP_YOLO_IOU = 0.50         # NMS IoU threshold

# --- tiled inference: finds SMALL / DISTANT poop ---------------------------
# The model was trained mostly on close-up phone photos where poop fills much
# of the frame, so distant poop (~15-20 px across your lawn) is easily missed.
# With tiling the frame is ALSO scanned as an NxN grid of overlapping crops,
# each blown up to the full model input — a 2x2 grid roughly doubles the
# apparent size of a distant target.
#   0 or 1 = off (1 forward pass) — misses the far end of the lawn
#   2      = full frame + 2x2 grid  (5 passes, ~65 fps)
#   3      = full frame + 3x3 grid  (10 passes, ~37 fps)  <-- IN USE
# Verified with debug_range.py on the real lawn: tiling finds far-end targets
# that the single pass misses. Same model, same confidence threshold — the
# targets simply become large enough to detect. Set back to 0 to revert.
#
# How much it matters, measured on one frame from the 29 Jul run with two real
# poops about 20 px across:
#
#                       no tiling   2x2      3x3
#     near poop            0.12     0.54     0.49
#     further poop         0.04     0.22     0.45
#
# The further one is the whole argument. Untiled it is invisible; at 2x2 it
# reaches 0.22, which the old 0.45 cut threw away; at 3x3 it is as strong as
# its neighbour. Ten passes instead of five costs frame rate the scan can
# afford, and DETECT_EVERY_N in scan_lawn already spreads the load.
#
# It is not free, though — timed on one frame: 1x1 75 ms, 2x2 369 ms,
# 3x3 580 ms. That is 1.6x the detector cost of the old setting. If the live
# feed starts feeling heavy again, the fallback that keeps most of the benefit
# is TILES = 2 with POOP_YOLO_CONF = 0.20: it still clears the 0.22 the
# further poop reaches at 2x2, and still sits above the 0.14 clutter floor.
POOP_YOLO_TILES = 3
POOP_YOLO_TILE_OVERLAP = 0.20   # crop overlap so poop on a seam isn't split

# --- ML tile classifier (poop / grass / other) -----------------------------
# Same proven pipeline as the grass/battery MobileNetV2 tile classifier. When
# poop_cls.onnx is present the detector fuses its per-blob probability with the
# colour+depth score; absent, colour+depth alone run the show.
POOP_MODEL = 'poop_cls.onnx'
POOP_LABELS_FILE = 'poop_labels.txt'
POOP_TILE_PX = 96             # classify a square crop around each blob
POOP_CNN_WEIGHT = 0.6         # fuse: score = w*cnn + (1-w)*colour_depth
POOP_CNN_CONF = 0.55          # min model 'poop' prob to count as a vote

# --- dataset (for training the classifier) ---------------------------------
from paths import DATA as _DATA, OUTPUT as _OUT
POOP_DATASET_DIR = str(_DATA / 'poop_dataset')   # <label>/*.png
POOP_LABELS = ('poop', 'grass', 'other')

# --- logging (kept simple for now; expand later) ---------------------------
POOP_LOG_DIR = str(_OUT / 'poop_logs')
POOP_SNAPSHOTS = True         # save an annotated frame per detection
# Estimated ground position uses base yaw + in-frame bearing + depth range.
CAMERA_HFOV_DEG = 69.0        # OAK-D Lite RGB horizontal FOV
RAW_PER_DEG = 4096.0 / 360.0  # Feetech raw per degree of base yaw

# --- taught scan pose loads here once calibrate_scan.py has run -------------
try:
    from poop_scan_pose import *          # noqa: F401,F403  (SCAN_POSE)
except ImportError:
    pass
# --- taught sweep range loads here once calibrate_sweep.py has run ----------
try:
    from poop_scan_range import *         # noqa: F401,F403  (SCAN_BASE_MIN/MAX_RAW,
except ImportError:                        # SCAN_STATIONS)
    pass
# --- demo box colour loads here once calibrate_box.py has run ---------------
try:
    from poop_box_ref import *            # noqa: F401,F403  (POOP_BOX_HSV_LO/HI)
except ImportError:
    pass
