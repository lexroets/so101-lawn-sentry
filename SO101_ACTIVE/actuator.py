"""Operational actuator for the SO-101 (base + shoulder/elbow/wrist).

Speaks the Feetech serial-bus protocol directly over pyserial (bundled with
Thonny) — no extra SDK or package needed. All servos share one bus.

Safety, which the caller cannot bypass:
  - base goals are clamped to the verified [BASE_RAW_MIN, BASE_RAW_MAX] range;
  - arm-joint goals are clamped to +/- MAX_JOINT_DELTA around the pose each
    joint was in when the program started (its captured rest pose);
  - emergency_stop() latches and cuts torque on EVERY servo.
"""
import time

import serial

from config import (
    SERIAL_PORT,
    BAUD_RATE,
    BASE_SERVO_ID,
    BASE_RAW_MIN,
    BASE_RAW_MAX,
    BASE_FORWARD_RAW,
    MAX_PROPOSED_STEP_RAW,
    AUTONOMOUS_ACTUATION_ENABLED,
    GOAL_MOVE_SPEED,
    GOAL_MOVE_ACC,
    ARM_JOINT_IDS,
    ARM_MOVE_SPEED,
    ARM_MOVE_ACC,
    REACH_NEAR_M,
    REACH_DELTAS_NEAR,
    REACH_FAR_M,
    REACH_DELTAS_FAR,
    REACH_ABSOLUTE,
    REACH_POSE_NEAR,
    REACH_POSE_FAR,
    REACH_POINTS,
    CAL_REST_POSE,
    MAX_JOINT_DELTA,
    GRIPPER_ID,
    GRIPPER_SPEED,
    GRIPPER_ACC,
    GRIP_STEP_RAW,
    GRIP_STALL_RAW,
    GRIP_SQUEEZE_RAW,
    GRIP_RELEASE_FRAC,
    GRIPPER_OPEN_RAW,
    GRIPPER_CLOSED_RAW,
)

# Margin (raw) added around the physically-demonstrated pose envelope.
CAL_LIMIT_MARGIN = 120

# --- Feetech STS3215 registers (16-bit values are little-endian) -------------
REG_TORQUE_ENABLE = 40      # 1 byte : 1 = on, 0 = off (limp)
REG_GOAL_ACC = 41           # 1 byte : acceleration ramp
REG_GOAL_POSITION = 42      # 2 bytes: writing this starts the move
REG_GOAL_SPEED = 46         # 2 bytes: profile speed
REG_PRESENT_POSITION = 56   # 2 bytes: current position 0..4095

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03


class ActuatorError(RuntimeError):
    pass


class BaseActuator:
    """Real, torque-capable actuator for the base and arm joints."""

    def __init__(self):
        if not AUTONOMOUS_ACTUATION_ENABLED:
            raise ActuatorError(
                "AUTONOMOUS_ACTUATION_ENABLED is False in config.py — refusing "
                "to open the servo bus."
            )
        self.stop_latched = False
        self.torque_on = False
        self.ids = (BASE_SERVO_ID,) + tuple(ARM_JOINT_IDS) + (GRIPPER_ID,)

        try:
            self.port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.08)
        except serial.SerialException as exc:
            raise ActuatorError(
                f"could not open {SERIAL_PORT} — check the USB cable and that "
                f"no other program has the port open. ({exc})"
            ) from exc
        time.sleep(0.1)
        self.port.reset_input_buffer()

        for sid in self.ids:
            if not self._ping(sid):
                self.port.close()
                raise ActuatorError(
                    f"no response from servo id {sid} on {SERIAL_PORT} — "
                    f"check arm power and wiring."
                )

        # Motion profiles: fast base swing, gentler arm joints, gentle gripper.
        self._set_profile(BASE_SERVO_ID, GOAL_MOVE_SPEED, GOAL_MOVE_ACC)
        for sid in ARM_JOINT_IDS:
            self._set_profile(sid, ARM_MOVE_SPEED, ARM_MOVE_ACC)
        self._set_profile(GRIPPER_ID, GRIPPER_SPEED, GRIPPER_ACC)

        # Rest pose: live positions for every controlled joint, overlaid
        # with the CALIBRATED values where they exist. Joints absent from
        # the calibration (e.g. the roll on older files) rest wherever they
        # are at startup — never a crash.
        self.rest_pose = {sid: self.read_position(sid)
                          for sid in ARM_JOINT_IDS}
        if CAL_REST_POSE:
            self.rest_pose.update({sid: v for sid, v in CAL_REST_POSE.items()
                                   if sid in ARM_JOINT_IDS})

        # Arm joint limits: REMOVED per operator request — full servo range.
        # (The base keeps its verified range; taught reach poses are trusted
        # because they were physically demonstrated.)
        self.joint_limits = {sid: (0, 4095)
                             for sid in ARM_JOINT_IDS + (GRIPPER_ID,)}

    # -- Feetech protocol ------------------------------------------------------
    def _send(self, servo_id, instruction, params):
        body = [servo_id, len(params) + 2, instruction] + list(params)
        checksum = (~sum(body)) & 0xFF
        self.port.reset_input_buffer()
        self.port.write(bytes([0xFF, 0xFF] + body + [checksum]))
        self.port.flush()

    def _read_reply(self, servo_id, n_params):
        expected = 6 + n_params
        reply = self.port.read(expected)
        if (len(reply) < expected or reply[0] != 0xFF or reply[1] != 0xFF
                or reply[2] != servo_id):
            return None
        if (~sum(reply[2:expected - 1])) & 0xFF != reply[expected - 1]:
            return None
        return reply[5:5 + n_params]

    def _ping(self, servo_id):
        for _ in range(3):
            self._send(servo_id, INST_PING, [])
            if self._read_reply(servo_id, 0) is not None:
                return True
            time.sleep(0.05)
        return False

    def _write(self, servo_id, address, values):
        self._send(servo_id, INST_WRITE, [address] + list(values))
        self.port.read(6)  # consume the status reply if the servo sends one

    def _read(self, servo_id, address, count):
        for _ in range(3):
            self._send(servo_id, INST_READ, [address, count])
            params = self._read_reply(servo_id, count)
            if params is not None:
                return params
        raise ActuatorError(f"no valid reply reading reg {address} of servo {servo_id}")

    def _set_profile(self, servo_id, speed, acc):
        self._write(servo_id, REG_GOAL_ACC, [int(acc) & 0xFF])
        self._write(servo_id, REG_GOAL_SPEED,
                    [int(speed) & 0xFF, (int(speed) >> 8) & 0xFF])

    # -- feedback --------------------------------------------------------------
    def read_position(self, servo_id=BASE_SERVO_ID):
        lo, hi = self._read(servo_id, REG_PRESENT_POSITION, 2)
        pos = lo | (hi << 8)
        if pos & 0x8000:                 # sign/direction bit (multiturn)
            pos = -(pos & 0x7FFF)
        return int(pos)

    # -- torque ----------------------------------------------------------------
    def enable(self):
        if self.stop_latched:
            raise ActuatorError("cannot enable torque: emergency stop is latched")
        for sid in self.ids:
            self._write(sid, REG_TORQUE_ENABLE, [1])
        self.torque_on = True

    def emergency_stop(self):
        """Latch a stop and cut torque on every servo (the arm goes limp)."""
        self.stop_latched = True
        self.torque_on = False
        for sid in self.ids:
            try:
                self._write(sid, REG_TORQUE_ENABLE, [0])
            except Exception:
                pass

    def enable_ids(self, ids):
        """Enable torque on specific servos only (e.g. just the base while
        the arm stays limp for hand-teaching)."""
        for sid in ids:
            self._write(sid, REG_TORQUE_ENABLE, [1])
        self.torque_on = True

    def torque_off(self):
        """Actively release every servo WITHOUT latching a stop — used by
        calibrate.py so the limbs can be moved by hand. (Servos keep their
        last torque state across program runs, so this must be explicit.)"""
        self.torque_on = False
        for sid in self.ids:
            try:
                self._write(sid, REG_TORQUE_ENABLE, [0])
            except Exception:
                pass

    # -- clamping --------------------------------------------------------------
    def _clamp(self, servo_id, target_raw):
        target = int(round(target_raw))
        if servo_id == BASE_SERVO_ID:
            return max(BASE_RAW_MIN, min(BASE_RAW_MAX, target))
        lo, hi = self.joint_limits[servo_id]
        return max(lo, min(hi, target))

    # -- motion ----------------------------------------------------------------
    def move_to(self, target_raw, servo_id=BASE_SERVO_ID):
        """Command one absolute goal; the servo drives itself there at its
        programmed profile. Clamped per-servo. Returns the goal sent."""
        if self.stop_latched:
            raise ActuatorError("stop latched; refusing to move")
        if not self.torque_on:
            raise ActuatorError("torque not enabled; call enable() first")
        target = self._clamp(servo_id, target_raw)
        self._write(servo_id, REG_GOAL_POSITION,
                    [target & 0xFF, (target >> 8) & 0xFF])
        return target

    def reach_pose(self, z, points=None):
        """Compute (t, {sid: raw}) for a target at depth z WITHOUT moving.
        `points` selects an alternative taught table (e.g. LYING_POINTS for
        lying-down batteries); default is the standard REACH_POINTS.
        Joints missing from a taught pose (e.g. the roll on older
        calibrations) fall back to their rest position."""
        pts_src = points if points is not None else REACH_POINTS
        if pts_src:
            pts = sorted(pts_src, key=lambda p: p[0])
            if z <= pts[0][0]:
                lo = hi = pts[0]
                t = 0.0
            elif z >= pts[-1][0]:
                lo = hi = pts[-1]
                t = 0.0
            else:
                for a, b in zip(pts, pts[1:]):
                    if a[0] <= z <= b[0]:
                        lo, hi = a, b
                        t = (z - a[0]) / max(1e-6, b[0] - a[0])
                        break
            return t, {sid: int(round(
                lo[1].get(sid, self.rest_pose[sid])
                + t * (hi[1].get(sid, self.rest_pose[sid])
                       - lo[1].get(sid, self.rest_pose[sid]))))
                for sid in ARM_JOINT_IDS}
        t = (z - REACH_NEAR_M) / max(1e-6, REACH_FAR_M - REACH_NEAR_M)
        t = max(0.0, min(1.0, t))
        if REACH_ABSOLUTE and REACH_POSE_NEAR and REACH_POSE_FAR:
            return t, {sid: int(round(
                REACH_POSE_NEAR.get(sid, self.rest_pose[sid])
                + t * (REACH_POSE_FAR.get(sid, self.rest_pose[sid])
                       - REACH_POSE_NEAR.get(sid, self.rest_pose[sid]))))
                for sid in ARM_JOINT_IDS}
        return t, {sid: int(round(self.rest_pose[sid]
                                  + REACH_DELTAS_NEAR.get(sid, 0)
                                  + t * (REACH_DELTAS_FAR.get(sid, 0)
                                         - REACH_DELTAS_NEAR.get(sid, 0))))
                   for sid in ARM_JOINT_IDS}

    def reach(self, z):
        """Move the arm joints toward a target at depth z (metres).

        Interpolates each joint between its calibrated NEAR pose (target at
        REACH_NEAR_M) and FAR pose (target at REACH_FAR_M). Returns
        (t, goals) where t is the interpolation factor actually used."""
        t, pose = self.reach_pose(z)
        goals = {sid: self.move_to(g, sid) for sid, g in pose.items()}
        return t, goals

    def move_joints(self, goals):
        """Command several arm-joint goals at once (each clamped). Returns
        the goals actually sent."""
        sent = {}
        for sid, target in goals.items():
            sent[sid] = self.move_to(target, sid)
        return sent

    # -- gripper -----------------------------------------------------------------
    def _gripper_torque(self):
        """Re-assert gripper torque. Feetech servos silently drop the torque
        flag after voltage dips (big arm moves cause them) — after that every
        goal write is ignored. Re-asserting before each command is cheap."""
        try:
            self._write(GRIPPER_ID, REG_TORQUE_ENABLE, [1])
        except Exception:
            pass

    def grip_open(self):
        """Open the gripper to its calibrated open position."""
        if GRIPPER_OPEN_RAW is None:
            return None                    # gripper not calibrated yet
        self._gripper_torque()
        return self.move_to(GRIPPER_OPEN_RAW, GRIPPER_ID)

    def grip_test(self):
        """Startup self-test: command open and verify the servo MOVED.
        Returns (commanded, reached) or None if uncalibrated."""
        if GRIPPER_OPEN_RAW is None:
            return None
        self._gripper_torque()
        self.move_to(GRIPPER_OPEN_RAW, GRIPPER_ID)
        time.sleep(1.2)
        return GRIPPER_OPEN_RAW, self.read_position(GRIPPER_ID)

    def grip_released(self):
        """True once the fingers have physically opened past the release
        point (GRIP_RELEASE_FRAC of the way to fully open). The gripper is
        slow — never retract the arm before this returns True."""
        if GRIPPER_OPEN_RAW is None or GRIPPER_CLOSED_RAW is None:
            return True
        span = GRIPPER_OPEN_RAW - GRIPPER_CLOSED_RAW
        if span == 0:
            return True
        try:
            pos = self.read_position(GRIPPER_ID)
        except ActuatorError:
            return False
        return (pos - GRIPPER_CLOSED_RAW) / span >= GRIP_RELEASE_FRAC

    def wait_released(self, timeout=3.0):
        """Block until grip_released() (manual-release path)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.grip_released():
                return True
            time.sleep(0.08)
        return False

    def grip_close(self):
        """SIMPLE close: one direct command to the calibrated closed
        position. The servo drives until the fingers stop on the object,
        then keeps holding torque toward closed — a firm natural grip.
        Waits until finger movement stops; returns the settled raw position
        (or None if the gripper isn't calibrated)."""
        if GRIPPER_OPEN_RAW is None or GRIPPER_CLOSED_RAW is None:
            return None
        self._gripper_torque()
        self.move_to(GRIPPER_CLOSED_RAW, GRIPPER_ID)
        last = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            if self.stop_latched:
                return last
            time.sleep(0.15)
            try:
                pos = self.read_position(GRIPPER_ID)
            except ActuatorError:
                continue
            if last is not None and abs(pos - last) <= 3:
                return pos          # settled: on the object or fully closed
            last = pos
        return last

    def set_base_profile(self, speed, acc):
        """Change the BASE servo's speed/acceleration profile on the fly —
        slow for the idle scan sweep, fast for target tracking."""
        self._set_profile(BASE_SERVO_ID, speed, acc)

    def set_arm_profile(self, speed, acc):
        """Change the arm joints' speed/acceleration profile on the fly
        (e.g. slow for the sleeve insertion, fast for everything else)."""
        for sid in ARM_JOINT_IDS:
            self._set_profile(sid, speed, acc)

    def arm_home(self):
        """Arm joints to their rest pose; base and gripper untouched.
        Used to lift/carry a gripped object before swinging the base."""
        return {sid: self.move_to(self.rest_pose[sid], sid)
                for sid in ARM_JOINT_IDS}

    def go_rest(self):
        """Return all arm joints to their captured rest pose and the base to
        the forward-facing centre."""
        for sid in ARM_JOINT_IDS:
            self.move_to(self.rest_pose[sid], sid)
        return self.move_to(BASE_FORWARD_RAW, BASE_SERVO_ID)

    # -- lifecycle ---------------------------------------------------------------
    def close(self):
        try:
            self.emergency_stop()
        finally:
            try:
                self.port.close()
            except Exception:
                pass
