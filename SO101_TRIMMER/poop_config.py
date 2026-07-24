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
DETECT_Z_MIN_M = 0.5              # TESTING range 0.5-2.0 m (floor is 0.19 m —
DETECT_Z_MAX_M = 2.0              # the OAK-D can't measure nearer than ~0.18 m;
                                  # stereo gets noisier past ~1.5 m but fine to test)
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

# Keep only the strongest N blobs per frame. 1 = "just the single brown box"
# (what you want while testing with one object); set 0 for unlimited (real
# lawns with several poops). The lawn-polygon ROI already discards anything off
# the grass, so with 1 you get exactly the box in front of the camera.
POOP_MAX_TARGETS = 1

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
POOP_DATASET_DIR = os.path.join(_HERE, 'poop_dataset')   # <label>/*.png
POOP_LABELS = ('poop', 'grass', 'other')

# --- logging (kept simple for now; expand later) ---------------------------
POOP_LOG_DIR = os.path.join(_HERE, 'poop_logs')
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
