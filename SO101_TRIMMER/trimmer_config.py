"""Config for the SO-101 TRIMMER arm (grass-cutting phase).

The gripper (servo 6) and wrist-roll (servo 5) have been removed and a 3D
printed trimmer is mounted on the wrist-pitch (servo 4). So this arm has FOUR
controlled joints only:

    1  base      (rotation / yaw)
    2  shoulder  (lift)
    3  elbow
    4  wrist     (pitch — carries the trimmer)

Everything here is independent of the battery-sorter config; this folder does
not import from SO101_ACTIVE, so the old system is untouched.
"""

# --- bus -------------------------------------------------------------------
SERIAL_PORT = "/dev/cu.usbmodem5B790152071"   # same USB port as before
BAUD_RATE = 1_000_000

# --- joints (NO roll, NO gripper) ------------------------------------------
BASE_ID = 1
SHOULDER_ID = 2
ELBOW_ID = 3
WRIST_ID = 4
JOINT_IDS = (BASE_ID, SHOULDER_ID, ELBOW_ID, WRIST_ID)

# --- motion profiles -------------------------------------------------------
# GENTLER than the battery arm: the trimmer adds mass and leverage at the
# wrist, so a softer acceleration keeps the motion natural and avoids
# whipping. Tune these once you see it move.
MOVE_SPEED = 800        # steps/s for repositioning moves
MOVE_ACC = 30           # acceleration ramp (x100 steps/s^2); low = smooth
SWEEP_SPEED = 600       # the cutting sweep itself — slower, deliberate
SWEEP_ACC = 22

# --- blended replay --------------------------------------------------------
# Move to the NEXT waypoint once within BLEND_TOL of the current one, instead
# of stopping dead at each. This is what makes the sweep flow continuously
# rather than stop-start. Set BLEND_TOL small (or 0) to stop at every point.
BLEND_TOL_RAW = 90
ARRIVE_TOL_RAW = 15     # "arrived" (used for REST / final settle)
ARRIVE_TIMEOUT_S = 4.0
SETTLE_S = 0.20

# --- safety ----------------------------------------------------------------
# Base yaw range verified on the original arm. Keep the trimmer sweeping
# within a safe arc; widen only after checking clearance.
BASE_RAW_MIN = 981
BASE_RAW_MAX = 2946
# Per-joint full travel; taught poses are trusted because you demonstrated
# them physically. A hard clamp still protects against a bad command.
JOINT_RAW_MIN = 0
JOINT_RAW_MAX = 4095

# --- camera (OAK-D Lite) ---------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
# ~69 deg horizontal FOV over 640 px -> raw-per-pixel for base centring.
# 4096 raw = 360 deg, so (69/360)*4096/640 ~= 1.23.
RAW_PER_PIXEL = 1.23
BASE_DIRECTION_SIGN = 1        # flip to -1 if the base yaws AWAY from the patch
BASE_FORWARD_RAW = 1995        # base yaw that points the camera straight ahead
CAMERA_LATENCY_S = 0.17        # measured on the original arm

# --- grass-patch detection (depth-proud of the flat deck) ------------------
# The mown deck is flat + level, so uncut grass stands PROUD of it in depth,
# exactly like a battery stood proud of the desk. A pixel is "grass" if it is
# nearer than the deck (its image row's median depth) by more than this.
GRASS_MIN_PROUD_M = 0.020      # >= this much taller than the deck = uncut
GRASS_MAX_PROUD_M = 0.25       # ignore anything implausibly tall (noise/objects)
PATCH_MIN_AREA_PX = 500        # ignore specks; a real patch is a big blob
PATCH_MAX_AREA_PX = 120000

# HARD near-range floor. The OAK-D Lite stereo cannot measure nearer than
# ~0.18m; below that the depth WRAPS and reads FURTHER as the object gets
# closer. There is no software fix — you must keep targets beyond this
# distance. Any reading below it is treated as invalid/out of range.
MIN_RELIABLE_Z = 0.19
DETECT_Z_MIN_M = MIN_RELIABLE_Z  # working depth window the arm can reach
DETECT_Z_MAX_M = 0.42            # tightened: batteries sit at 0.20-0.35m, so
                                 # anything beyond this is desk edge / wall /
                                 # background that was being chased and lost.

# --- desk-test objects (batteries standing proud of the desk) --------------
# For bench testing the reach with a Ryobi battery instead of grass: a battery
# is a smaller, more solid blob than a grass patch.
OBJECT_MIN_PROUD_M = 0.020     # object stands at least this far proud of desk.
                               # Raised from 12mm: desk texture / noise was
                               # registering as proud and creating phantom
                               # targets that stalled the centring loop.
OBJECT_MIN_AREA_PX = 800       # real batteries measured 2500-3700 px; 800
                               # kills the near-range specks (area ~190-270)
                               # without missing a genuine battery.
OBJECT_MAX_AREA_PX = 40000
# Constant offsets added to every taught reach pose, tuned LIVE in
# reach_object.py and pasted here to persist:
#   joint 2 (SHOULDER) = HEIGHT  — o/l keys raise/lower the tip
#   joint 4 (WRIST)    = ANGLE   — k/j keys tilt the head to keep it FLAT
# The shoulder is the real height lever; the wrist only sets the head angle.
REACH_TRIM = {2: 0, 3: 0, 4: 0}
HEIGHT_TRIM_STEP = 10          # raw per keypress for live trimming
APPROACH_HOVER_FRAC = 0.6      # go to this fraction of the descent first
                               # (a safe hover), then the rest — no lunging.

# --- base centring on a patch (PD tracker, smoothed + rate-limited) --------
CENTRE_GAIN = 0.9             # proportional gain on the smoothed pixel error
TRACK_ERR_SMOOTH = 0.4        # EMA on the noisy patch centroid (lighter = faster)
TRACK_MIN_CMD_INTERVAL_S = 0.09   # quicker corrections -> faster lock
GOAL_UPDATE_MIN_RAW = 10
CENTRE_DEADBAND_PX = 12       # the reach is blind + the head is wide, so
                              # pixel-perfect centring is unnecessary; a wider
                              # band locks MUCH faster.
CENTRE_STABLE_FRAMES = 2
CENTRE_SPEED = 1600           # snappy base profile while centring (the gentle
CENTRE_ACC = 80               # MOVE_ACC=30 made small corrections crawl)

# --- level trim motion -----------------------------------------------------
# Base-only yaw sweep about the (vertical) base axis keeps the trimmer tip at
# CONSTANT height over the flat deck, i.e. a level cut, with NO inverse
# kinematics. Half-amplitude and pass count are taught / set here.
TRIM_SWEEP_HALF_RAW = 220     # yaw each side of the set point (~19 deg)
TRIM_PASSES = 2               # back-and-forth passes per patch
TRIM_SWEEP_SPEED = 500        # slow, deliberate cut
TRIM_SWEEP_ACC = 20

# --- ML grass recognition (uncut vs cut vs other) --------------------------
# A tile classifier: the frame is chopped into patches, each scored uncut /
# cut / other by a MobileNetV2 model (same pipeline as the battery classifier).
# Uncut tiles that also stand proud of the ground (depth) form the target.
GRASS_MODEL = 'grass_cls.onnx'      # trained ONNX, sits next to the scripts
GRASS_LABELS_FILE = 'grass_labels.txt'
GRASS_TILE_PX = 96          # patch size collected AND classified (keep equal!)
GRASS_STRIDE_PX = 48        # runtime tile stride (overlap = finer localisation)
GRASS_UNCUT_CONF = 0.60     # min model prob to call a tile 'uncut'
GRASS_BG_MIN_CONF = 0.80    # 'other'/'cut' must be this sure to veto uncut
GRASS_MIN_REGION_TILES = 3  # a target region needs at least this many uncut tiles
GRASS_FUSE_DEPTH = True     # also require the region to stand proud of the deck
GRASS_TEMPORAL = 0.5        # EMA on the per-tile uncut score (0..1) to de-flicker

# Taught poses load from here once the calibrators have been run.
REST_POSE = None
SWEEP = None
CUT_POINTS = None             # [(z, {sid: raw}), ...] flat "cut the top" table
EDGE_POINTS = None            # (legacy) separate edging table — no longer needed
# EDGE is DERIVED from the flat reach: same shoulder/elbow, wrist rotated 90 deg
# so the trimmer stands vertical. 90 deg = 4096/4 = 1024 raw. Either sign keeps
# it perpendicular to the ground; flip to -1024 if it rotates the wrong way or
# hits a joint limit.
EDGE_WRIST_OFFSET_RAW = 1024
REACH_TRIM_EDGE = {2: 0, 3: 0, 4: 0}   # persisted edge FINE trim (o/l/k/j + p)
try:
    from trimmer_poses import *      # noqa: F401,F403
except ImportError:
    pass
try:
    from trimmer_cut import *        # noqa: F401,F403  (REST_POSE, CUT_POINTS,
except ImportError:                  # TRIM_SWEEP_HALF_RAW overrides)
    pass
try:
    from reach_trim import *         # noqa: F401,F403  persisted REACH_TRIM,
except ImportError:                  # saved with the 'p' key in reach_object
    pass
try:
    from trimmer_edge import *       # noqa: F401,F403  EDGE_POINTS
except ImportError:
    pass
try:
    from reach_trim_edge import *    # noqa: F401,F403  persisted REACH_TRIM_EDGE
except ImportError:
    pass
