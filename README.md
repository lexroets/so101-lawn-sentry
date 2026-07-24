# SO-101 autonomous architecture — actuation disabled

This folder contains the full software architecture around autonomous base centring, but deliberately contains no code that enables torque or writes a goal position.

Included:
- DepthAI v3 synchronised RGB/depth pipeline
- pen-cap candidate detection
- exactly-one-target requirement
- 8-frame stability gate
- pixel/depth jitter checks
- target and camera timeouts
- latched E/Escape software stop
- verified raw-range check
- small bounded proposed relative steps
- disabled actuator interface preserving the integration boundary

Run `main_dry_run.py` in Thonny. The screen shows proposed raw base steps and explicitly states that no command is sent.

`actuator_disabled.py` is intentionally non-operational. It contains no torque-enable or goal-position register implementation.

---

## ACTIVE MODE (base now physically moves)

`actuator.py` + `main_active.py` add the real integration. The base servo rotates
in small bounded steps to centre a single stable pen-cap target.

### One-time setup (Thonny)
1. Tools > Manage packages > install **`feetech-servo-sdk`** (provides the
   `scservo_sdk` module). `depthai`, `opencv-python` and `numpy` must already be
   installed for the dry run.
2. Confirm `SERIAL_PORT` in `config.py` matches your board
   (Feetech/Waveshare port, e.g. `/dev/cu.usbmodem…`). Close LeRobot or any other
   program holding the port.
3. Power the arm.

### Run
Open **`main_active.py`** in Thonny and press Run. Torque enables after a 3-second
countdown, then the base centres on the target.

- **E** or **Esc** — latched software emergency stop (cuts torque; restart to re-run)
- **q** — quit cleanly (also cuts torque and closes the port)

### Safety notes
- Motion only happens while the supervisor reports one stable, verified target
  (8-frame gate, jitter/timeout checks). Loss of target holds position.
- Every command is clamped twice inside `actuator.py`: to the verified raw range
  `[BASE_RAW_MIN, BASE_RAW_MAX]`, and to at most `MAX_PROPOSED_STEP_RAW` from the
  servo's actual present position.
- If the base turns **away** from the pen cap, set `BASE_DIRECTION_SIGN = -1` in
  `config.py` and re-run.
- To go back to the non-moving version, run `main_dry_run.py` (unchanged).
