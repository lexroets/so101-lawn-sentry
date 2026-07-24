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
# Gentle fixed motion profile for each bounded step.
GOAL_MOVE_SPEED = 300   # steps/second (0 = max; keep low)
GOAL_MOVE_ACC = 20      # 0 = instant, higher = smoother ramp

STABLE_FRAMES = 8
MAX_CENTRE_JITTER_PX = 7.0
MAX_DEPTH_JITTER_M = 0.012
IMAGE_DEADBAND_PX = 14
TARGET_TIMEOUT_S = 0.5
FRAME_TIMEOUT_S = 1.0
MAX_PROPOSED_STEP_RAW = 8

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10
