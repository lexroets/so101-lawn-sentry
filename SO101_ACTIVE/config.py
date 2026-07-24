# SO-101 autonomous architecture — actuation ENABLED
AUTONOMOUS_ACTUATION_ENABLED = True
DRY_RUN_ONLY = False

SERIAL_PORT = "/dev/cu.usbmodem5B790152071"
BAUD_RATE = 1_000_000
BASE_SERVO_ID = 1
BASE_RAW_MIN = 981
BASE_RAW_MAX = 2946
BASE_FORWARD_RAW = 1995

# Set to -1 if the base turns AWAY from the pen cap instead of toward it.
BASE_DIRECTION_SIGN = 1
# Fast motion profile: the servo drives itself to the goal at this speed.
GOAL_MOVE_SPEED = 2400  # steps/second (0 = flat out; lower if too violent)
GOAL_MOVE_ACC = 80      # acceleration ramp (x100 steps/s^2)

# Pixel -> raw conversion for the one-shot move to the locked point.
# 4096 raw = 360 deg; OAK RGB ~69 deg over 640 px  =>  ~1.23 raw per pixel.
# If the base consistently OVERSHOOTS the target, lower this; if it stops
# SHORT, raise it.
RAW_PER_PIXEL = 1.23
ARRIVE_TOLERANCE_RAW = 4

# ---- Closed-loop tracking (goal updated live while the base moves) ----------
# The target's PIXEL POSITION is noisy (the depth centroid of a small, close
# battery jumps +-20px frame to frame), so the trick to smooth tracking is
# NOT derivative control (which amplifies that noise) but:
#   1. EMA-smooth the error so the base chases a STABLE reference, and
#   2. RATE-LIMIT the goal commands so each move actually executes instead of
#      being interrupted every frame (that interruption WAS the glitchiness).
CLOSED_LOOP_GAIN = 0.8        # proportional gain on the SMOOTHED error
TRACK_ERR_SMOOTH = 0.5        # EMA weight on the previous error (0..1). Higher
                              # = smoother reference but laggier; 0 = raw noisy
                              # error (old behaviour).
TRACK_MIN_CMD_INTERVAL_S = 0.13   # minimum time between base goal commands.
                              # Lets the servo execute a move before the next
                              # one — the single biggest anti-jitter lever.
GOAL_UPDATE_MIN_RAW = 12      # ignore goal changes smaller than this.
TRACK_SETTLE_DEADBAND_PX = 2  # inside this SMOOTHED error, hold (no new goal).
                              # Must stay below IMAGE_DEADBAND_PX (3).
TRACK_MAX_STEP_RAW = 9999     # slew limit effectively OFF — keeps the big
                              # opening move snappy.

# ---- IDLE SCAN: sweep the base to find targets outside the current view ------
# The camera only sees ~69 deg. When AUTO SORT finds nothing, instead of
# declaring "no more batteries" the base now sweeps slowly side to side,
# pausing at each step so the detector gets clean, motion-free frames.
SCAN_ENABLED = True
SCAN_START_S = 2.5        # seconds of empty IDLE before scanning begins
SCAN_STEP_RAW = 220       # base movement per scan step (~19 deg, a bit under
                          # half the FOV, so views overlap and nothing hides
                          # in the seam between two steps)
SCAN_SETTLE_S = 0.55      # pause at each step for the image to stop blurring
SCAN_SPEED = 700          # slow, controlled sweep (vs GOAL_MOVE_SPEED 2400)
SCAN_ACC = 30
SCAN_MIN_RAW = 1300       # outer sweep limits. The ACTUAL arc is narrowed at
SCAN_MAX_RAW = 2700       # runtime to stay clear of the buckets (below).
# KEEP-OUT around each taught bucket. The buckets are battery-shaped clutter
# sitting at known base angles (green 2482, red 1462), and sweeping across
# them made the robot try to pick its own buckets up. The scan arc is clamped
# to the region BETWEEN the buckets, and target locking is refused entirely
# while the base points within this many raw of either one.
BUCKET_KEEPOUT_RAW = 170  # NARROWED from 300. At 300 the scan arc stopped
                          # 300 raw short of each bucket AND batteries placed
                          # in front of a bucket were unreachable. The colour-
                          # evidence gate (COLOUR_MIN_EVIDENCE_PX) is what
                          # actually stops the bucket itself being locked;
                          # this only keeps the sweep from staring into one.
SCAN_MAX_SWEEPS = 3       # give up after this many full passes with no target
SCAN_FIRST_DIR = +1       # ALWAYS sweep RIGHT first. Raw increases to the
                          # right (the green bucket sits at 2482, right of the
                          # 1995 forward centre), so +1 = right. This is now
                          # re-applied every time a scan starts — previously
                          # the direction carried over from the last sweep, so
                          # it often set off leftwards.
LOCK_MATCH_RADIUS_PX = 80     # (legacy) association radius without prediction
ASSOC_RADIUS_PX = 45          # tight radius around the PREDICTED target
                              # position — keeps the lock off lookalikes
ASSOC_Z_TOL_M = 0.06          # candidate depth must match the lock's depth

# ---- Object detection ---------------------------------------------------------
REQUIRE_GREEN_PLANE = False   # False = detect on ANY desk (depth-based);
                              # True = old green-plane gating
DESK_PROUD_M = 0.010          # object must stand this far proud of the desk.
                              # 12mm missed lying batteries (only ~23mm tall);
                              # 7mm let stereo noise (+-3-5mm at 25cm) into the
                              # mask, which merged objects into sprawling blobs
                              # that failed the solidity test — most clicks then
                              # landed on nothing. 10mm is the working middle.
FX_PIXELS = 465.0             # focal length in px (640 px / ~69 deg HFOV)
CAP_WIDTH_M = (0.004, 0.125)  # (legacy, unused by the battery gate below)
CAP_HEIGHT_M = (0.004, 0.125)

# TARGET gate tightened to the RB4L30 battery's dimensions (~115x40x23 mm):
# in ANY orientation its smaller image dimension is ~18-50 mm and its larger
# ~35-130 mm. Far more specific than independent width/height ranges.
# NOTE: detection_calibration.py overrides these at the end of this file.
# Change them THERE, not here.
BATT_MIN_DIM_M = (0.014, 0.058)
BATT_MAX_DIM_M = (0.028, 0.140)

# Ignore anything outside the arm's taught reach range — it's background and
# unreachable anyway. Range comes from the calibration (REACH_NEAR/FAR_M).
REACH_RANGE_MARGIN_M = 0.04
# NEAR side needs a MUCH bigger margin than the far side. The nearest taught
# reach point is 0.268 m, so a 0.04 margin discarded everything closer than
# 0.228 m — and batteries were logged at 0.220/0.232/0.235 m, i.e. sitting on
# the cutoff. They were rejected before they could ever be highlighted.
# The arm reaches nearer than its nearest taught point, so open this up.
REACH_NEAR_MARGIN_M = 0.16   # NEAR objects were the ones being thrown away.
                             # Nearest taught reach is 0.268m, so this puts the
                             # floor at ~0.108m — essentially the stereo
                             # camera's own minimum range with extended
                             # disparity on. Nothing close is rejected now.

# ---- Pick-and-place: drop box + carry/drop behaviour --------------------------
BOX_WIDTH_M = (0.14, 0.40)    # the DROP BOX: real size 14-40 cm (must not
BOX_HEIGHT_M = (0.05, 0.40)   # overlap the target range above)
DROP_POSE_FRAC = 0.8          # drop pose = this fraction of a full reach, so
                              # the gripper ends above the box rim, not in it
DROP_PAUSE_S = 0.6            # pause after opening before retracting
CARRY_TIMEOUT_S = 3.0         # max wait per carry step (arm home / base swing)
PLACE_BLIND_S = 3.0           # if the box isn't re-detected, place blind after this
CAP_ASPECT = (0.12, 4.0)      # lower bound must pass a LYING battery: on its
                              # side it is ~115x23mm -> h/w = 0.20, which the
                              # old 0.25 floor rejected as aspect_rej
CAP_SOLIDITY_MIN = 0.35       # loose blob-solidity sanity check
MIN_PLANE_AREA_PX = 15000     # the green plane must be at least this big
PLANE_EDGE_DILATE_PX = 21     # accept objects this close to the plane boundary

# ---- Base tracking latency compensation --------------------------------------
# The image is ~2 frames old by the time it's processed; goals are computed
# against where the base WAS at capture time, which kills overshoot at speed.
CAMERA_LATENCY_S = 0.12

# ---- Debugging ---------------------------------------------------------------
DEBUG = True                  # write debug_log.txt next to main.py
DEBUG_STATS_PERIOD_S = 1.0    # how often to log detector reject statistics

# ---- Physical reach calibration (run calibrate.py once!) ---------------------
# calibrate.py lets you move the limp arm BY HAND onto the cap at a near and a
# far position and records the exact servo poses. Those taught poses replace
# all the IK/angle guesswork above. Defaults below are inactive placeholders.
REACH_ABSOLUTE = False
REACH_POSE_NEAR = None
REACH_POSE_FAR = None
REACH_POINTS = None       # multi-point calibration: [(z_m, {sid: raw}), ...]
CAL_REST_POSE = None

# ---- Other SO-101 joints (Feetech STS3215, same serial bus) -----------------
SHOULDER_ID = 2
ELBOW_ID = 3
WRIST_ID = 4
ROLL_ID = 5               # wrist ROLL — controlled: records and reproduces
                          # the roll (essential for lying top-down grabs)
ARM_JOINT_IDS = (SHOULDER_ID, ELBOW_ID, WRIST_ID, ROLL_ID)
# Arm joints: fast but a notch below the base swing.
ARM_MOVE_SPEED = 900
ARM_MOVE_ACC = 60

# Reach poses = raw offsets from the pose the arm is in when the program
# STARTS (start the program with the arm in its rest position!).
# TWO calibration poses; the reach interpolates between them by target depth.
# Calibrate physically: put the cap at ~REACH_NEAR_M, lock, and tune the NEAR
# deltas until the gripper is at the cap; repeat at ~REACH_FAR_M for FAR.
# If a joint moves the WRONG WAY, flip its sign. 0 = don't move that joint.
REACH_NEAR_M = 0.20
REACH_DELTAS_NEAR = {
    SHOULDER_ID: -250,   # slight forward lean for a close target
    ELBOW_ID: 300,
    WRIST_ID: 120,
}
REACH_FAR_M = 0.60
REACH_DELTAS_FAR = {
    SHOULDER_ID: -500,   # full extension for a far target
    ELBOW_ID: 600,
    WRIST_ID: 220,
}
MAX_JOINT_DELTA = 1200   # hard safety envelope (raw, ~105 deg) around the start pose

# ---- SO-101 geometry (open-source SO-ARM101 URDF values, metres) ------------
UPPER_ARM_M = 0.11257      # shoulder axis -> elbow axis
FOREARM_M = 0.1349         # elbow axis -> wrist-flex axis
WRIST_TO_TIP_M = 0.108     # wrist-flex axis -> gripper tip
SHOULDER_HEIGHT_M = 0.11   # shoulder axis above the desk — MEASURE yours
SHOULDER_FORWARD_M = 0.03  # shoulder axis forward of the base rotation axis

# ---- Camera mounting relative to the ARM BASE (MEASURE these three!) --------
CAM_HEIGHT_M = 0.25          # lens height above the desk (m)
CAM_PITCH_DEG = 25.0         # camera downward tilt from horizontal (deg)
CAM_OFFSET_FORWARD_M = -0.05 # camera forward(+) / behind(-) the arm base (m)

# ---- Joint angle calibration --------------------------------------------------
# What angle each joint is at in the START/REST pose, in these conventions:
#   shoulder: upper-arm angle from horizontal-forward, up = positive
#   elbow   : interior angle between upper arm and forearm (180 = straight)
#   wrist   : gripper angle relative to the forearm direction (0 = inline)
# Eyeball these against your rest pose — a few degrees off is fine, then tune.
REST_ANGLES_DEG = {SHOULDER_ID: 100.0, ELBOW_ID: 30.0, WRIST_ID: 0.0}
JOINT_SIGNS = {SHOULDER_ID: 1, ELBOW_ID: 1, WRIST_ID: 1}  # flip if a joint
                                                          # moves the wrong way
GRIPPER_APPROACH_DEG = -75.0  # approach angle (steep = from above; right for
                              # close-in desk targets)
USE_IK = True                 # False = fall back to the NEAR/FAR pose reach

# The cap always stands ON the desk, so its height is KNOWN — this removes
# the camera-pitch calibration from the maths entirely. Only CAM_HEIGHT_M and
# CAM_OFFSET_FORWARD_M still matter (measure with a ruler).
TARGET_ON_DESK = True
CAP_CENTER_HEIGHT_M = 0.02    # grab-point height above the desk (m)

# One-time direction check: at startup each arm joint moves +15 deg for a
# moment then returns. Watch it — any joint moving the WRONG WAY (up/back
# instead of forward/down) needs its JOINT_SIGNS entry flipped. Set False
# once directions are confirmed.
JOINT_TEST_ON_START = True
IK_UPDATE_S = 0.30            # limb goal refresh rate while on target
Z_MEDIAN_SAMPLES = 7          # depth smoothing window (kills z spikes)
ENTER_DEBOUNCE_S = 0.4        # ignore repeated ENTER within this window

STABLE_FRAMES = 8
MAX_CENTRE_JITTER_PX = 7.0
MAX_DEPTH_JITTER_M = 0.012
IMAGE_DEADBAND_PX = 3   # "on target" when within ±3 px of the centreline
                        # (raise to 5 if it hunts side-to-side, lower for exactness)
TARGET_TIMEOUT_S = 0.5
FRAME_TIMEOUT_S = 1.0
MAX_PROPOSED_STEP_RAW = 8

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480   # full 4:3 sensor = ~30% more vertical FOV than 16:9
                      # (horizontal FOV is fixed by the lens at ~69 deg)
CAMERA_FPS = 15   # higher fps = tighter closed loop at the higher speeds

# ---- YOLO (runs ON the OAK camera — nothing installed on the Mac) -----------
USE_YOLO = True
YOLO_MODEL = "yolov6-nano"    # DepthAI Zoo slug; downloaded once, then cached.
                              # Swap for a custom pen-cap model slug later.
YOLO_CONF = 0.35              # minimum detection confidence
CLASSICAL_FALLBACK = True     # use the HSV/contour detector when YOLO is silent

# ---- Gripper (servo id 6) -----------------------------------------------------
GRIPPER_ID = 6
GRIPPER_SPEED = 400
GRIPPER_ACC = 30
GRIP_DELAY_S = 0.3        # settle time AFTER the arm physically arrives
ARM_ARRIVE_TOL_RAW = 40   # all arm joints within this of the reach pose = arrived
GRIP_TIMEOUT_S = 5.0      # close anyway if the arm hasn't arrived by then

# ---- Mouse selection + centring confirmation ----------------------------------
CLICK_RADIUS_PX = 70      # click snaps to a detection within this; otherwise
                          # the clicked pixel itself is used (depth measured there)
CENTRE_STABLE_FRAMES = 3  # consecutive centred+settled frames before staging
VERIFY_ALIGN_FRAMES = 2   # consecutive aligned frames before the final reach
CENTRE_BLIND_S = 1.5      # clicked point never re-detected: proceed once the
                          # base has reached its goal for this long

# ---- Staged reach with mid-flight verification --------------------------------
REACH_STAGE_FRAC = 0.7    # stage 1 moves this fraction of the way to the cap
STAGE_TIMEOUT_S = 3.0     # max wait for a stage to physically arrive
VERIFY_S = 1.2            # pause at stage 1: re-look at the cap, correct base
VERIFY_CORR_INTERVAL_S = 0.30  # min gap between base corrections — must exceed
                               # the measured 0.17s camera latency so each
                               # correction is observed before the next fires
GRIP_STEP_RAW = 25        # closing step per cycle (smaller = gentler)
GRIP_RELEASE_FRAC = 0.5   # gripper counts as "released" once this far toward
                          # open — the arm must NOT retract before this
GRIP_STALL_RAW = 45       # position lag that means "fingers on the object"
GRIP_SQUEEZE_RAW = 20     # extra squeeze after contact, for a firm hold
# Open/closed raw positions come from calibrate_gripper.py (run it once):
GRIPPER_OPEN_RAW = None
GRIPPER_CLOSED_RAW = None

# ---- Precision placement into the sorting sleeve (run calibrate_place.py) ----
USE_PLACE_SEQUENCE = False  # True = fixed taught sleeve sequence;
                            # False = click-to-place anywhere (default)
PLACE_CARRY = None        # taught poses (absolute raw) — None until taught
PLACE_HOVER = None
PLACE_INSERT = None
PLACE_BASE_RAW = None     # base rotation facing the sleeve
INSERT_SPEED = 250        # slow, controlled descent (sleeve AND click-place)
INSERT_ACC = 20
BUCKET_DIP_RAW = 160      # lower this far (shoulder) INTO the bucket mouth
                          # before releasing — placing, not dropping
RETURN_SPEED = 600        # gentle profile for the fold-arm-then-swing return
RETURN_ACC = 40
PLACE_SETTLE_S = 0.4      # settle at the insert pose before opening
PLACE_LOW_FRAC = 0.95     # click-place: lower to this fraction of a full
                          # reach before releasing (just above the surface)

# ---- 5-slot battery holster (run calibrate_slots.py) --------------------------
USE_SLOTS = False         # set True by slots_calibration.py once taught
SLOTS = None              # [{'base': raw, 'hover': {...}, 'insert': {...}}, ...]
USABLE_SLOTS = (1, 3, 5)  # only auto-fill these (holster too compact for the
                          # gripper between adjacent slots) — demo setting

# Grab height: <1.0 stops the grab descent early, gripping the battery
# HIGHER UP (nearer its top) along the taught approach line. 1.0 = exactly
# the taught pose (which was taught near the battery's base).
GRAB_POSE_FRAC = 0.93

# Upright-grab wrist trim: tilts the wrist (id 4) at the final grab pose so
# the gripper enters PARALLEL to the floor. Tune on the robot: each 12 raw
# is about 1 degree; flip the sign if it tilts the wrong way.
UPRIGHT_WRIST_TRIM_RAW = 0

# ---- Two-bucket colour sorting (run calibrate_buckets.py) ----------------------
USE_BUCKETS = False       # set True by buckets_calibration.py once taught
BUCKET_GREEN = None       # {'base': raw, 'pose': {...}} — Ryobi batteries here
BUCKET_RED = None         # pink batteries here
RYOBI_HUE = (30, 55)      # lime ACCENTS on the otherwise black Ryobi
PINK_HUE = (130, 179)     # pink batteries (wide magenta band)
PINK_HUE2 = (0, 6)        # coral wrap band — NARROW (warm whites leak here)
COLOUR_S_MIN = 60         # min saturation for LIME pixels
PINK_S_MIN = 25           # pastel pink is pale: much lower saturation bar
PINK_MIN_PX = 150         # pink needs real pixel MASS — the white text side
                          # of a Ryobi produced 79 stray "pink" pixels
# The Ryobi's WHITE text side: bright + unsaturated = ryobi evidence.
WHITE_S_MAX = 40
WHITE_V_MIN = 170
WHITE_MIN_FRAC = 0.30
COLOUR_MIN_FRAC = 0.08    # (legacy)
# The Ryobi battery body is BLACK — classified by darkness, not hue:
DARK_V_MAX = 90           # pixel value (brightness) below this = dark
DARK_MIN_FRAC = 0.25      # this fraction of dark pixels = black battery
NOT_DARK_MAX_FRAC = 0.15  # UNDER this dark fraction = definitely not the
                          # black Ryobi -> classified pink (2-type demo)
                          # (narrow 15-25% gap = far fewer 'unknown' reads)
PINK_MIN_FRAC = 0.10      # this fraction of pink pixels = pink battery
# Pink wins when its pixel count exceeds this fraction of the TOTAL ryobi
# evidence (dark + lime + white). Set from measured runs, not guesswork:
#
#   ryobi_px  pink_px   ratio   truth
#     1443      332      0.23   ryobi
#     2286     1684      0.74   pink
#      888      693      0.78   pink
#
# Clean gap between 0.23 and 0.74, so 0.45 sits in the middle with room on
# both sides. Do NOT weight dark/lime/white separately — comparing pink to
# the combined total is what these numbers actually support.
PINK_VS_RYOBI_W = 0.45

# Count LIME pixels as Ryobi evidence? FALSE when the batteries sit on a
# GREEN MAT — the mat floods the lime hue band and every battery, pink
# included, reads as a Ryobi. The Ryobi is identified by its BLACK body and
# WHITE text anyway; the lime accents were only ever a weak bonus signal.
# Set True again if you go back to a neutral (non-green) surface.
LIME_AS_RYOBI = False

# SIZE OVERRIDE — colour-independent, so the green mat cannot affect it.
# The RB4L30 is ~115 mm; the pink cells are ~30% shorter (~80 mm). Anything
# measuring shorter than this CANNOT be a Ryobi, whatever the pixels say.
# Raise toward 0.100 if a pink one still reads ryobi; lower toward 0.085 if
# a Ryobi ever gets called pink.
RYOBI_MIN_LEN_M = 0.090
# The size override only applies to a PLAUSIBLE battery length. Without this
# floor, noise fragments measuring 28mm and 42mm were being force-labelled
# "pink battery" and carried to the red bucket.
PLAUSIBLE_LEN_M = (0.060, 0.140)

# SIZE OVERRIDE IS OFF. Measured 2026-07-21: Ryobi lengths came out
# 29/30/33/67/80..99mm and pink 29..93mm — total overlap, because the
# detector often measures a FRAGMENT rather than the whole battery.
# Length cannot separate these two types on this rig.
USE_SIZE_OVERRIDE = False

# ---- DARKNESS: the discriminator that actually works ---------------------
# Measured over 43 in-situ samples:
#     Ryobi (black):  dark pixels 539 .. 2215
#     Pink  (pale):   dark pixels   6 ..   61
# A ~10x gap, and it is immune to the green mat (not dark), to lighting
# (relative measure), and to orientation (a lying black battery is still
# black). Hue was useless here — both types measured hue 80-107 with
# saturation 20-50, i.e. too pale to carry a stable colour.
DARK_SPLIT_FRAC = 0.15    # >= this fraction of dark pixels = RYOBI

# Minimum total colour evidence before ANY verdict is given. Without this,
# "not dark" was read as "pink", so a featureless patch of green mat near a
# bucket locked as a pink battery with dark=0 pink=0 white=0 — pure phantom.
# Below this, classify_colour returns 'unknown' and auto-sort skips it.
COLOUR_MIN_EVIDENCE_PX = 150

# ---- Partially-visible targets ----------------------------------------------
# A battery close to the camera gets clipped by the frame edge. Those contours
# used to be thrown away outright (edge_rej), so a battery you could plainly
# see went undetected. Now they are kept, provided the visible part is big and
# solid enough to be a real object rather than a sliver of clutter.
# ---- Performance ------------------------------------------------------------
# Detection only runs in the phases that actually READ the candidate list.
# During carry / place / return the result was computed and thrown away,
# which was a large slice of the per-frame cost for no benefit.
DETECT_PHASES = frozenset((
    'IDLE', 'SEL_BOX', 'CENTRE', 'VERIFY', 'STAGE1', 'FULL',
))
# The orientation metrics (proud height, depth spread) are DIAGNOSTIC ONLY
# while AUTO_ORIENT is False, and depth_spread runs a percentile over every
# candidate's ROI each frame. Skip them unless they can change a decision.
COMPUTE_ORIENT_METRICS = True   # needed for the top-edge orientation signal;
                                # it is one median pair per candidate, far
                                # cheaper than the old percentile spread.
HUD_SHOW_FPS = True

ALLOW_PARTIAL_TARGETS = True
PARTIAL_MIN_AREA_PX = 900   # a clipped-but-close battery is LARGE in frame,
                            # so demand real area — this is what keeps mat
                            # edges and the arm itself out.
AUTO_RELOCK_DELAY_S = 1.2 # settle time after a cycle before locking the next
                          # target (the base is still swinging home)
PINK_MAX_LEN_M = 0.095    # pink batteries are ~30% smaller: anything longer
                          # than this CANNOT be pink -> forced Ryobi
DETECT_Z_MAX_M = 0.45     # back to 0.45 — the FAR cap was never the problem,
                          # and raising it only invited distant clutter.
AUTO_IDLE_TIMEOUT_S = 4.0 # auto-sort ends after no batteries seen this long

# ---- Lying-down batteries (run calibrate_lying.py to teach the grabs) --------
# Orientation logic is INVERTED: UPRIGHT must be proven, otherwise LYING.
# An upright RB4L30 has an unmistakable signature — a tall narrow silhouette
# (~115mm tall x ~23-40mm wide) and it stands far proud of the desk. A lying
# battery can look like almost anything depending on which way it points and
# how the depth contour fragments, so it is the default.
UPRIGHT_MIN_TALL = 1.9    # aspect is a WEAK signal — your data showed lying
                          # at h/w=1.31 and upright at h/w=0.89, so only a
                          # clearly tall silhouette counts as upright now.
LYING_MAX_PROUD_M = 0.045 # (legacy — desk-relative height proved unreliable)

# Depth-based orientation spread (measured, but NOT trusted — see below).
UPRIGHT_MIN_SPREAD_M = 0.040

# AUTOMATIC orientation detection: OFF.
#
# Measured on 2026-07-20, lying batteries gave depth spreads of 22/67/203/240mm
# while a STANDING one gave 8mm — uncorrelated, and it drove a vertical grab
# onto an upright battery. The bbox contains far more desk than battery, so
# every depth-derived metric measures the table's gradient, not the object.
#
# With this False, grabs are UPRIGHT unless the operator presses L. That is
# the safe default: a side-grab on a lying battery misses harmlessly, whereas
# a top-down grab on a standing one drives the gripper into it.
#
# Set True only if you have trained the two-class YOLO (see YOLO_TRAINING.md),
# which classifies orientation from appearance instead of depth.
AUTO_ORIENT = True

# Top-edge residual below this = LYING. A standing RB4L30 puts its top
# 60-115mm proud of the desk; lying, only its 23mm diameter shows.
# 45mm sits between the two with room either side.
UPRIGHT_MIN_TOP_M = 0.045
# LYING must WIN by this margin, not merely tie. The two failures are not
# equal: a side-grab on a lying battery misses harmlessly, a top-down grab
# on a standing one drives the gripper into it. So when the signal is
# marginal we deliberately fall back to the upright side-grab.
LYING_VOTE_MARGIN = 4

# YOLO class names that state orientation directly. When a detection carries
# one of these labels it OVERRIDES the depth estimate completely — the network
# is looking at the battery's appearance, which is the reliable cue.
# Matching is case-insensitive substring, so "rb4l30_lying" matches "_lying".
YOLO_LYING_LABELS = ('_lying', 'battery_lying')
YOLO_UPRIGHT_LABELS = ('_upright', 'battery_upright')
LYING_ASPECT = 1.6        # (legacy, unused by the inverted test)
LYING_MIN_LEN_M = 0.045
LYING_VOTE_BIAS = 2       # each lying frame counts this much (a lying battery
                          # is unambiguous; an upright one never looks flat)
LYING_POINTS = None       # taught top-down grab poses [(z, {sid: raw}), ...]

# ---- Detection fusion + stability ---------------------------------------------
MERGE_RADIUS_PX = 40      # YOLO + depth detections this close = same object
STABLE_RADIUS_PX = 40     # a candidate must reappear within this across
                          # frames to count (kills background flicker)

# ==============================================================================
# PRODUCTION / RELIABILITY (production.py). Every flag here defaults to the
# SAFE choice; set the ..._ENABLED flags to False to get the exact demo
# behaviour back. Nothing here changes anything until enabled.
# ==============================================================================

# ---- Failed-grab detection + retry -------------------------------------------
# The gripper closes to GRIPPER_CLOSED_RAW (~1666) on EMPTY air; a held
# battery stops the fingers higher (~1740-1810). So a settled position at or
# just above fully-closed means the grab MISSED. Rather than carry nothing to
# a bucket, reopen and try again; after too many misses, cool that spot down
# and move on so it can't loop forever.
GRAB_RETRY_ENABLED = True
GRAB_EMPTY_MARGIN = 45        # settled <= closed+this = empty grip
GRAB_MAX_RETRIES = 2          # re-attempts before giving up on a spot
GRAB_COOLDOWN_S = 8.0         # ignore a repeatedly-failing spot this long
GRAB_COOLDOWN_RADIUS_PX = 45  # ...within this pixel radius

# ---- Watchdog + error recovery -----------------------------------------------
# Backstop for unattended running. If any active (non-idle) phase runs longer
# than this, something hung — recover to rest and resume. Set high enough that
# no NORMAL phase ever hits it (a full cycle is ~16s; a single phase is <4s).
WATCHDOG_ENABLED = True
WATCHDOG_PHASE_MAX_S = 25.0
# Tolerate transient serial/camera errors instead of crashing: log and keep
# going. After this many CONSECUTIVE hard errors, stop for safety.
WATCHDOG_MAX_CONSEC_ERRORS = 8

# ---- Startup self-check ------------------------------------------------------
# Verify the rig before the arm moves. Missing servos or camera abort the run
# (can't operate safely); missing OPTIONAL calibrations only warn.
SELFCHECK_ENABLED = True
SELFCHECK_ABORT_ON_FAIL = False  # warn-only by default so a false positive
                                 # can never block a run. Flip to True for
                                 # true unattended operation once trusted.

# ---- Session metrics ---------------------------------------------------------
# Append per-run stats (attempts, placed, empty grips, per-colour) to a file
# and print a summary on exit. Useful for reviewing unattended runs.
METRICS_ENABLED = True

# ---- Motion smoothing (EXPERIMENTAL — needs hardware tuning) ------------------
# Overlaps the retreat/return phases and scales acceleration to move distance
# for faster, less robotic motion. OFF by default: it changes the motion path
# and MUST be tuned/verified on the actual arm before trusting it. Enable only
# when you can watch the arm and are ready to tune the lead fractions below.
SMOOTH_MOTION_ENABLED = False
SMOOTH_RETREAT_LEAD = 0.6     # start the base swing when the arm retreat is
                              # this fraction complete (1.0 = old sequential)
CARRY_ACC_SCALE = 0.7         # gentler acceleration while carrying a battery
EMPTY_ACC_SCALE = 1.0         # normal acceleration when empty-handed

# ==============================================================================
# CALIBRATION OVERRIDES — MUST STAY AT THE VERY END OF THIS FILE.
# Taught values override everything above. Anything defined after these lines
# would silently cancel the calibration — that bug already happened once.
# ==============================================================================
try:
    from reach_calibration import *
except ImportError:
    pass
try:
    from gripper_calibration import *
except ImportError:
    pass
try:
    from base_calibration import *      # RAW_PER_PIXEL / SIGN / LATENCY,
except ImportError:                     # measured by calibrate_full.py
    pass
try:
    from place_calibration import *     # taught sleeve-insertion sequence
except ImportError:
    pass
try:
    from slots_calibration import *     # taught 5-slot holster sequence
except ImportError:
    pass
try:
    from buckets_calibration import *   # taught two-bucket sorting demo
except ImportError:
    pass
try:
    from detection_calibration import * # measured battery sizes + colours
except ImportError:
    pass
try:
    from lying_calibration import *     # taught lying-battery grab poses
except ImportError:
    pass
try:
    from colour_calibration import *    # measured pink-vs-ryobi thresholds
except ImportError:                     # (calibrate_colour.py) — LAST so the
    pass                                # in-situ measurement wins
