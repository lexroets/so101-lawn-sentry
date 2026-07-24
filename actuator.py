"""Operational base actuator for the SO-101 (single base servo).

Talks to the SO-101 base joint — a Feetech STS3215 serial-bus servo — over the
Feetech protocol using scservo_sdk. It exposes the same small interface the rest
of the application already expects (enable / read_position /
move_one_bounded_step / emergency_stop), but every command passes through two
hard safety layers that the caller cannot bypass:

  1. the raw goal is clamped to the verified [BASE_RAW_MIN, BASE_RAW_MAX] range;
  2. the goal is limited to at most MAX_PROPOSED_STEP_RAW away from the servo's
     *actual* present position, so a bad proposal can never cause a large jump.

This module replaces actuator_disabled.py. That file is left in place so the
non-actuating dry run (main_dry_run.py) still works.
"""

import time

from scservo_sdk import sms_sts, PortHandler
try:
    from scservo_sdk import COMM_SUCCESS
except Exception:  # some SDK builds don't re-export the constant
    COMM_SUCCESS = 0

from config import (
    SERIAL_PORT,
    BAUD_RATE,
    BASE_SERVO_ID,
    BASE_RAW_MIN,
    BASE_RAW_MAX,
    MAX_PROPOSED_STEP_RAW,
    AUTONOMOUS_ACTUATION_ENABLED,
    GOAL_MOVE_SPEED,
    GOAL_MOVE_ACC,
)

# --- Feetech STS3215 register addresses (verified) ---------------------------
ADDR_TORQUE_ENABLE = 40      # 1 byte : 1 = on, 0 = off (limp)
ADDR_GOAL_ACC = 41           # 1 byte : acceleration ramp
ADDR_GOAL_POSITION = 42      # 2 bytes: writing this triggers the move
ADDR_GOAL_SPEED = 46         # 2 bytes: profile speed
ADDR_PRESENT_POSITION = 56   # 2 bytes: current position, 0..4095


class ActuatorError(RuntimeError):
    pass


class BaseActuator:
    """Real, torque-capable base actuator with built-in clamping."""

    def __init__(self):
        if not AUTONOMOUS_ACTUATION_ENABLED:
            raise ActuatorError(
                "AUTONOMOUS_ACTUATION_ENABLED is False in config.py — refusing "
                "to open the servo bus."
            )

        self.stop_latched = False
        self.torque_on = False

        self.port = PortHandler(SERIAL_PORT)
        self.servo = sms_sts(self.port)

        if not self.port.openPort():
            raise ActuatorError(
                f"could not open serial port {SERIAL_PORT} — check the cable and "
                f"that no other program (e.g. LeRobot) has it open."
            )
        if not self.port.setBaudRate(BAUD_RATE):
            self.port.closePort()
            raise ActuatorError(f"could not set baud rate {BAUD_RATE}")

        _model, comm, _err = self.servo.ping(BASE_SERVO_ID)
        if comm != COMM_SUCCESS:
            self.port.closePort()
            raise ActuatorError(
                f"no response from servo id {BASE_SERVO_ID} on {SERIAL_PORT} — "
                f"check power and wiring."
            )

        # Program a gentle, fixed motion profile once. Every subsequent write to
        # the goal-position register moves at this speed/acceleration.
        self.servo.write1ByteTxRx(BASE_SERVO_ID, ADDR_GOAL_ACC, int(GOAL_MOVE_ACC))
        self.servo.write2ByteTxRx(BASE_SERVO_ID, ADDR_GOAL_SPEED, int(GOAL_MOVE_SPEED))

    # -- feedback -------------------------------------------------------------
    def read_position(self):
        pos, comm, _err = self.servo.read2ByteTxRx(BASE_SERVO_ID, ADDR_PRESENT_POSITION)
        if comm != COMM_SUCCESS:
            raise ActuatorError("failed to read present position from servo")
        return int(pos)

    # -- torque ---------------------------------------------------------------
    def enable(self):
        if self.stop_latched:
            raise ActuatorError("cannot enable torque: emergency stop is latched")
        _r, comm, _err = self.servo.write1ByteTxRx(BASE_SERVO_ID, ADDR_TORQUE_ENABLE, 1)
        if comm != COMM_SUCCESS:
            raise ActuatorError("failed to enable torque")
        self.torque_on = True

    def emergency_stop(self):
        """Latch a stop and cut torque. The servo goes limp and will not move
        again until the program is restarted."""
        self.stop_latched = True
        self.torque_on = False
        try:
            self.servo.write1ByteTxRx(BASE_SERVO_ID, ADDR_TORQUE_ENABLE, 0)
        except Exception:
            pass

    # -- motion ---------------------------------------------------------------
    def move_one_bounded_step(self, target_raw):
        """Command a single, clamped, bounded step toward target_raw.

        Returns the raw goal actually sent after safety clamping."""
        if self.stop_latched:
            raise ActuatorError("stop latched; refusing to move")
        if not self.torque_on:
            raise ActuatorError("torque not enabled; call enable() first")

        present = self.read_position()

        # Layer 1: never leave the verified raw range.
        target = max(BASE_RAW_MIN, min(BASE_RAW_MAX, int(target_raw)))

        # Layer 2: never move more than one bounded step from the ACTUAL position.
        delta = target - present
        if delta > MAX_PROPOSED_STEP_RAW:
            target = present + MAX_PROPOSED_STEP_RAW
        elif delta < -MAX_PROPOSED_STEP_RAW:
            target = present - MAX_PROPOSED_STEP_RAW

        # Re-clamp in case the present position sat just outside the range.
        target = max(BASE_RAW_MIN, min(BASE_RAW_MAX, target))

        _r, comm, _err = self.servo.write2ByteTxRx(BASE_SERVO_ID, ADDR_GOAL_POSITION, target)
        if comm != COMM_SUCCESS:
            raise ActuatorError("failed to write goal position")
        return target

    # -- lifecycle ------------------------------------------------------------
    def close(self):
        try:
            self.emergency_stop()
        finally:
            try:
                self.port.closePort()
            except Exception:
                pass
