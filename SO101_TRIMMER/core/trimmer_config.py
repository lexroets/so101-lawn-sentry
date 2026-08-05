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

# --- bus -------------------------------------------------------------------
SERIAL_PORT = "/dev/cu.usbmodem5B790152071"   # same USB port as before
BAUD_RATE = 1_000_000

# --- joints (NO roll, NO gripper) ------------------------------------------
BASE_ID = 1
SHOULDER_ID = 2
ELBOW_ID = 3
WRIST_ID = 4
# The joint the trimmer HEAD is bolted to - wrist roll. It sets which way the
# cutting plane faces, which is the whole difference between mowing a top and
# edging a side, so it cannot stay unconfigured.
HEAD_ID = 5

# The four that must be there for the arm to be usable at all. The head is
# listed separately because it is a later addition: a rig without it should
# still run rather than refusing to start.
# --- head joint travel -------------------------------------------------------
# The head does NOT have a normal min/max. Its travel is an arc that runs
# through the encoder's zero:
#
#     1000  ....  anticlockwise limit
#       |         counts DOWN
#       0 / 4095  flat - the trimmer head's rest orientation, on the wrap
#       |         counts DOWN
#     3000  ....  clockwise limit
#
# So the ALLOWED values are  0..1000  and  3000..4095,
# and 1001..2999 - the part containing 2048 - is the FORBIDDEN half where the
# head is at a dangerous angle and the cable gets wound.
#
# Written as a plain range this is nonsense: min=3000, max=1000 is empty, and
# 0..4095 permits the forbidden half. Everything therefore works in ARC
# COORDINATES (see pathplay.to_arc), counting from the clockwise limit up
# through the wrap to the anticlockwise one, where it is a plain interval.
HEAD_ARC_LO = 1000          # clockwise limit; arc counts UP from here...
HEAD_ARC_HI = 3100          # ...through 4095/0, to the anticlockwise limit
HEAD_FLAT_RAW = 2048           # flat, on the wrap, mid-arc

# Crossing the wrap IS allowed, because each step is commanded and then read
# back to confirm the servo went the short way - see pathplay.step_head.
HEAD_ALLOW_WRAP_CROSSING = True

HEAD_MAX_STEP_RAW = 1200    # about 105 degrees in one waypoint

REQUIRED_IDS = (BASE_ID, SHOULDER_ID, ELBOW_ID, WRIST_ID)
OPTIONAL_IDS = (HEAD_ID,)
JOINT_IDS = REQUIRED_IDS + OPTIONAL_IDS

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
# Full servo travel. The old 981..2946 window was set for the trimmer, whose
# head could hit things; a camera on top has no such constraint and needs to
# look wherever the lawn is.
BASE_RAW_MIN = 0
BASE_RAW_MAX = 4095
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

# The head angle that suits APPROACHING a patch is not the angle that suits
# CUTTING it. Reaching, the head is level and hovering on top; cutting, it has
# dropped through the grass and the geometry has changed, so the same wrist
# value leaves it skewed. This offset applies ONLY once the cut starts, on top
# of REACH_TRIM, and is tuned live with [ and ] and saved with p.
CUT_WRIST_TRIM = 0
CUT_WRIST_STEP = 8             # raw per keypress - finer than the height step,
                               # because head angle is the fussier of the two

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


# --- camera ----------------------------------------------------------------
# A property of the OAK-D Lite itself, so it lives with the hardware config
# rather than in either project. Used only as a fallback: camera_intrinsics
# asks the device for its factory calibration first.
CAMERA_HFOV_DEG = 69.0        # OAK-D Lite RGB horizontal field of view


# --- travelling between paths ------------------------------------------------
# The move from the end of the top pass to the start of a side pass is a
# TRAVEL move, not a cut. Interpolated straight it goes through the box,
# because a straight line in joint space between two points either side of an
# obstacle does not care that the obstacle is there.
#
# So a travel move arcs: the shoulder opens on the way across and closes again
# on arrival, peaking in the middle. Both ends are untouched, so it still
# starts and finishes exactly on the taught poses.
TRAVEL_LIFT_RAW = 220        # about 19 deg of extra clearance at the peak
TRAVEL_LIFT_MIN_RAW = 350    # only arc when the move is at least this big


# --- holding the head flat ---------------------------------------------------
# On this arm the shoulder, elbow and wrist all pitch about parallel axes, so
# the head's angle to the ground is their SUM and nothing else. Measured on the
# taught top pass, that sum varies by 215 raw - 18.9 degrees of the head
# tipping about while the pass is cut, which is the fluctuation that looks bad.
#
# It does not have to be taught at all. Fix the sum and the wrist follows from
# the other two: wrist = target - shoulder - elbow. The head is then flat
# everywhere by construction, and one fewer thing has to be got right by hand.
#
# Only the TOP pass by default. The side pass legitimately changes pitch as it
# works down the face - levelling it would need 556 raw of wrist, which is a
# different motion, not a tidy-up. Add 'side_1' here if you want it anyway.
LEVEL_PITCH_PATHS = ('top',)
LEVEL_PITCH_TARGET = {}     # path -> sum; absent means "median of that pass"
LEVEL_PITCH_TRIM = 0        # nudge the locked angle, raw (+ tips one way)


# --- cutting at ONE height ---------------------------------------------------
# With the pitch locked, the head's height above the table depends only on the
# shoulder and the elbow - the base just points the arm, it does not raise or
# lower it. So "hold one cutting plane" is a single curve through
# (shoulder, elbow), and every pose on that curve is at the same height.
#
# A pass dragged by hand does not sit on that curve; it scatters either side of
# it. Measured on the taught top pass the scatter is 135 raw of elbow, and that
# scatter IS the uneven cut. Fitting the curve and putting every waypoint back
# onto it makes the height constant by construction.
#
# The fit is checked before it is used: if the taught pass does not describe a
# consistent plane the correction is refused rather than applied blind.
LEVEL_HEIGHT_PATHS = ('top',)
LEVEL_HEIGHT_DEGREE = 2      # quadratic; the locus is smooth over this range
# -133 keeps the AVERAGE height that was taught and only removes the
# unevenness. Set to 0 to cut on the surface that was actually touched -
# but the taught pass sits 133 raw of elbow away from it, so that is a
# real move, not a tidy-up. Change it in steps and watch.
LEVEL_HEIGHT_TRIM = -133
LEVEL_HEIGHT_MAX_FIX = 200
# Over how much shoulder travel the correction fades to nothing at the
# edge of the calibrated range, so the uncalibrated ends of a pass join
# the levelled middle smoothly instead of stepping into it.
LEVEL_HEIGHT_FADE_RAW = 150   # refuse if any waypoint needs more than this


# --- the arm's real geometry -------------------------------------------------
# From the published SO-101 URDF (TheRobotStudio/SO-ARM100, so101_new_calib),
# NOT fitted. Fitting the lengths from touch points looked reasonable and came
# out 42% wrong on the ratio - and the level curve of the wrong arm is not the
# level curve of this one, which is why a tilt survived every correction.
#
# These two are the STOCK part of the arm - shoulder to elbow, elbow to wrist
# pitch. Everything past the wrist pitch is modified on this rig and is NOT
# used here: with the head's pitch locked, its vertical contribution is a
# constant, and holding a constant height does not care about constants.
ARM_L1_M = 0.1160        # shoulder_lift -> elbow_flex
ARM_L2_M = 0.1350        # elbow_flex    -> wrist_flex

# The cut height, in MILLIMETRES above the plane taught in level_plane.py.
# It used to be in elbow ticks, which was a mistake: a fixed elbow offset is
# not a fixed height, so the units themselves introduced a tilt.
# 25 mm is the average height the pass was ALREADY cutting at, so this
# changes the evenness without changing the height. Lower it to cut
# closer, raise it to cut higher - it is millimetres now, so it means
# what it says at every reach.
LEVEL_HEIGHT_TRIM_MM = 0.0

# Lower the CLOSE end of the sweep by this much, fading to nothing at the far
# end. The plane the arm cuts and the plane the work actually sits on are not
# always the same: the box may not be level, the table may not be, or the head
# may load differently folded in than stretched out. Rather than re-teach the
# plane to absorb that, tilt the cut.
#
# Positive lowers the near rows. The far end is left exactly where
# LEVEL_HEIGHT_TRIM_MM puts it, so a sweep that is already right at the far
# end stays right.
LEVEL_HEIGHT_NEAR_DROP_MM = 0.0

# ...and the same for the FAR end. Two independent ends rather than one tilt,
# because in practice only one of them is ever wrong: whichever end is right
# should not move while the other is being fixed.
LEVEL_HEIGHT_FAR_DROP_MM = 0.0


# Subtract the overshoot measured by trimmer/check_level.py. The arm droops
# under gravity and sticks against friction, and neither can be predicted from
# the pose - but both repeat, to 0.2 mm across two measured runs, so they can
# simply be measured and taken off the command.
SAG_COMPENSATE = True

# Learn automatically. After a path is played in run_box, the arm walks it
# again a pose at a time, reads where it actually landed, and folds the
# difference into the correction - so every run is closer than the last with
# nothing to remember to do. The learning walk is slower than the pass and
# only happens when this is on.
AUTO_LEARN = False
AUTO_LEARN_PASSES = 1        # learning walks per play; it stops early anyway
