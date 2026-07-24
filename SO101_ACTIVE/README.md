# SO-101 autonomous base centring — ACTIVE

Self-contained copy of the original architecture with the motors live.
`detector.py`, `planner.py`, `safety.py` and `config.py` are unchanged from
your original code. `actuator.py` drives the base servo directly over
pyserial (already bundled with Thonny) — **nothing to install**.

## Run

1. Power the arm, plug in the USB serial board and the OAK camera.
   **Put the arm in its rest position first** — that pose is captured at
   startup as the home the arm returns to.
2. Open `main.py` from THIS folder in Thonny and press Run.
3. Torque enables after a 3-second countdown. The feed highlights detected
   candidates in yellow — when the box is on the pen cap, press **ENTER**.
   The pixel error is converted to a raw goal once and the base drives
   straight there at speed. The locked point is fixed: detection flicker or
   box movement afterwards is ignored. **ENTER** again unlocks.

Keys (click the video window first): **ENTER** = lock & go / release+rest ·
**B** = when gripped, bring the object back to base (gripper stays closed) ·
**E**/**Esc** = emergency stop (cuts torque, restart to re-run) · **q** = quit.

Pick-and-place cycle (phases shown on screen and tagged in the log):
IDLE (targets yellow, boxes magenta) → ENTER locks the target → SEL_BOX
(ENTER confirms the box, X = grab only) → CENTRE → STAGE1 → VERIFY → FULL →
GRIP → CARRY (lifts, swings to the box) → PLACE_CENTRE (re-centres on the
box visually) → PLACE_REACH (lowers to DROP_POSE_FRAC above the rim) →
DROP → back to rest. Nothing moves until both selections are made. The base
never tracks after the arm deploys, so it can't chase its own gripper.

Detection classifies by REAL size: targets 4-75 mm (battery, cap...), boxes
8-35 cm (`BOX_WIDTH_M`/`BOX_HEIGHT_M` in config). For true named-object
recognition (e.g. "Ryobi battery"), train a custom model and set
`YOLO_MODEL` to its zoo slug — everything else stays the same.

IMPORTANT for a new object (battery): re-run `calibrate.py` teaching the
gripper ON THE BATTERY at 3-4 distances — grab height and grip differ from
the pen cap. Tune `DROP_POSE_FRAC` so the drop pose clears your box rim.

Once the base arrives, the shoulder/elbow/wrist (servo ids 2/3/4) reach
toward the target, scaled by its measured depth. ENTER again returns the
whole arm to the captured rest pose and the base to the forward centre;
`q` also parks at rest before cutting torque. **E/Esc cuts torque on every
servo — the arm goes limp, so be ready to catch it.**

Detection: YOLO runs ON the OAK camera (model auto-downloaded from the
DepthAI Zoo on first run — needs internet once, then cached). COCO models
have no pen-cap class, so detections are gated by real pen-cap size
(5–30 mm × 10–70 mm via depth), and the classical HSV detector fills in
whenever YOLO is silent. Each box is tagged `yolo:` or `hsv` on screen.
Swap `YOLO_MODEL` in `config.py` for a custom-trained pen-cap model later.

## Gripper — RUN `calibrate_gripper.py` ONCE (30 seconds, terminal only)

Releases the gripper so you can move the fingers by hand; records fully-open
and fully-closed. After that, each cycle is: lock → gripper opens → base
centres → arm reaches → 1.5 s settle → gripper closes until the fingers
stall on the cap (stall-detected, so it grips without crushing) → ENTER
releases and returns to rest. Tune `GRIP_STALL_RAW` in `config.py` if it
stops too early (raise) or squeezes too hard (lower).

## Reach calibration — RUN `calibrate.py` FIRST (multi-point)

Arm torque stays off; move the limbs by hand. Teach the gripper touching the
cap at AS MANY distances as you like — 3-5 spread across your working range
is ideal; the reach interpolates between the two nearest taught depths, so
more points = more accurate distancing. F finishes; G also calibrates the
gripper in the same session (recommended). It writes:

- `reach_calibration.py` / `gripper_calibration.py` — used automatically
- `calibration_export.txt` — UPLOAD THIS TO CHAT to finalise the system

Delete `reach_calibration.py` to revert.

Full-limb reach (`USE_IK = True`, only without calibration): once the base has centred the target, a
2-link inverse-kinematics solve (open-source SO-ARM101 link lengths) positions
the shoulder, elbow and wrist so the gripper tip reaches the target's computed
3D position, refreshed while locked. To make it accurate, MEASURE and set in
`config.py`:
1. `CAM_HEIGHT_M`, `CAM_PITCH_DEG`, `CAM_OFFSET_FORWARD_M` — where the camera
   sits relative to the arm base (ruler + eyeball is fine).
2. `SHOULDER_HEIGHT_M` — desk to the shoulder axis.
3. `REST_ANGLES_DEG` — the approximate joint angles of your rest pose
   (conventions in `kinematics.py`). A few degrees off is fine.
4. First lock: any joint driving the WRONG WAY → flip it in `JOINT_SIGNS`.
The `IK ...` lines in debug_log.txt show the computed target position and
angles — paste them if the reach lands wrong. Set `USE_IK = False` to fall
back to the simple NEAR/FAR pose reach.
- Base closed-loop hunts side-to-side → lower `CLOSED_LOOP_GAIN`; converges
  too slowly → raise it toward 1.0.
- Joint goals can never leave ±`MAX_JOINT_DELTA` (600 raw ≈ 53°) around the
  captured rest pose.

Debugging: every run appends events to `debug_log.txt` (LOCK / TRACK /
ARRIVED / REACH / UNLOCK plus once-per-second detector statistics). Paste
that file into chat to help tune the system.

## If something is off

- Base turns AWAY from the pen cap → set `BASE_DIRECTION_SIGN = -1` in `config.py`.
- "could not open /dev/cu.usbmodem…" → check the port name in `config.py`
  (Thonny shell: `import serial.tools.list_ports; [p.device for p in serial.tools.list_ports.comports()]`)
  and make sure nothing else has the port open.
- "no response from servo id 1" → arm power / cable.

Motion only happens while there is exactly one stable target (8-frame gate,
jitter and timeout checks). Every command is clamped to the verified raw range
[981, 2946] and to ≤ 8 raw steps from the servo's actual position.
