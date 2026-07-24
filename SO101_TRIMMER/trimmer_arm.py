"""Slim 4-joint serial driver for the SO-101 trimmer arm.

Same Feetech STS3215 protocol as the battery arm's actuator.py, but for the
four remaining joints only (base, shoulder, elbow, wrist) — no gripper, no
roll. Self-contained: imports only trimmer_config, so the old system is
untouched.
"""
import time
import serial

from trimmer_config import (
    SERIAL_PORT, BAUD_RATE, JOINT_IDS, BASE_ID,
    MOVE_SPEED, MOVE_ACC, BASE_RAW_MIN, BASE_RAW_MAX,
    JOINT_RAW_MIN, JOINT_RAW_MAX, ARRIVE_TOL_RAW, ARRIVE_TIMEOUT_S,
)

REG_TORQUE_ENABLE = 40
REG_GOAL_ACC = 41
REG_GOAL_POSITION = 42
REG_GOAL_SPEED = 46
REG_PRESENT_POSITION = 56
INST_PING, INST_READ, INST_WRITE = 0x01, 0x02, 0x03


class TrimmerError(RuntimeError):
    pass


class TrimmerArm:
    def __init__(self):
        self.stop_latched = False
        self.torque_on = False
        self.ids = tuple(JOINT_IDS)
        try:
            self.port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.08)
        except serial.SerialException as exc:
            raise TrimmerError(
                f"could not open {SERIAL_PORT} — check the USB cable and that "
                f"no other program has the port open. ({exc})") from exc
        time.sleep(0.1)
        self.port.reset_input_buffer()
        for sid in self.ids:
            if not self._ping(sid):
                self.port.close()
                raise TrimmerError(
                    f"no response from servo id {sid} — check arm power/wiring. "
                    f"(Joints 5/6 are expected to be GONE; JOINT_IDS={JOINT_IDS})")
        for sid in self.ids:
            self._set_profile(sid, MOVE_SPEED, MOVE_ACC)
        self.start_pose = {sid: self.read_position(sid) for sid in self.ids}

    # -- Feetech protocol ----------------------------------------------------
    def _send(self, sid, inst, params):
        body = [sid, len(params) + 2, inst] + list(params)
        checksum = (~sum(body)) & 0xFF
        self.port.reset_input_buffer()
        self.port.write(bytes([0xFF, 0xFF] + body + [checksum]))
        self.port.flush()

    def _read_reply(self, sid, n):
        expected = 6 + n
        reply = self.port.read(expected)
        if (len(reply) < expected or reply[0] != 0xFF or reply[1] != 0xFF
                or reply[2] != sid):
            return None
        if (~sum(reply[2:expected - 1])) & 0xFF != reply[expected - 1]:
            return None
        return reply[5:5 + n]

    def _ping(self, sid):
        for _ in range(3):
            self._send(sid, INST_PING, [])
            if self._read_reply(sid, 0) is not None:
                return True
            time.sleep(0.05)
        return False

    def _write(self, sid, addr, values):
        self._send(sid, INST_WRITE, [addr] + list(values))
        self.port.read(6)

    def _read(self, sid, addr, count):
        for _ in range(3):
            self._send(sid, INST_READ, [addr, count])
            params = self._read_reply(sid, count)
            if params is not None:
                return params
        raise TrimmerError(f"no reply reading reg {addr} of servo {sid}")

    def _set_profile(self, sid, speed, acc):
        self._write(sid, REG_GOAL_ACC, [int(acc) & 0xFF])
        self._write(sid, REG_GOAL_SPEED,
                    [int(speed) & 0xFF, (int(speed) >> 8) & 0xFF])

    def set_profile(self, speed, acc, ids=None):
        for sid in (ids or self.ids):
            self._set_profile(sid, speed, acc)

    # -- feedback ------------------------------------------------------------
    def read_position(self, sid):
        lo, hi = self._read(sid, REG_PRESENT_POSITION, 2)
        pos = lo | (hi << 8)
        if pos & 0x8000:
            pos = -(pos & 0x7FFF)
        return int(pos)

    def read_pose(self):
        return {sid: self.read_position(sid) for sid in self.ids}

    # -- torque --------------------------------------------------------------
    def enable(self):
        if self.stop_latched:
            raise TrimmerError("cannot enable: emergency stop latched")
        for sid in self.ids:
            self._write(sid, REG_TORQUE_ENABLE, [1])
        self.torque_on = True

    def torque_off(self):
        self.torque_on = False
        for sid in self.ids:
            self._write(sid, REG_TORQUE_ENABLE, [0])

    def emergency_stop(self):
        self.stop_latched = True
        self.torque_on = False
        for sid in self.ids:
            try:
                self._write(sid, REG_TORQUE_ENABLE, [0])
            except Exception:
                pass

    # -- motion --------------------------------------------------------------
    def _clamp(self, sid, raw):
        if sid == BASE_ID:
            return max(BASE_RAW_MIN, min(BASE_RAW_MAX, int(raw)))
        return max(JOINT_RAW_MIN, min(JOINT_RAW_MAX, int(raw)))

    def move_to(self, sid, raw):
        if self.stop_latched:
            raise TrimmerError("stop latched; refusing to move")
        target = self._clamp(sid, raw)
        self._write(sid, REG_GOAL_POSITION,
                    [target & 0xFF, (target >> 8) & 0xFF])
        return target

    def move_pose(self, pose):
        """Command several joints at once (each clamped). pose = {sid: raw}."""
        return {sid: self.move_to(sid, raw) for sid, raw in pose.items()}

    def arrived(self, pose, tol=ARRIVE_TOL_RAW):
        try:
            return all(abs(self.read_position(sid) - raw) <= tol
                       for sid, raw in pose.items())
        except TrimmerError:
            return False

    def wait_arrived(self, pose, tol=ARRIVE_TOL_RAW, timeout=ARRIVE_TIMEOUT_S):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.arrived(pose, tol):
                return True
            time.sleep(0.02)
        return False

    def close(self):
        try:
            self.port.close()
        except Exception:
            pass
