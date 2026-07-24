"""Planar kinematics for the SO-101 arm (open-source SO-ARM101 geometry).

Once the base has centred the target on the centreline, the arm, the camera
ray and the target all lie in one vertical plane — so a 2-link planar IK
(shoulder lift + elbow) plus a wrist orientation solve gives the full reach.

Angle conventions (degrees):
    shoulder : upper-arm angle from horizontal-forward, up = positive
    elbow    : interior angle between upper arm and forearm (180 = straight)
    wrist    : gripper angle relative to the forearm direction (0 = inline)

Link lengths come from the open-source SO-ARM101 URDF (config.py).
"""
import math

from config import (
    CAMERA_HEIGHT,
    FX_PIXELS,
    CAM_HEIGHT_M,
    CAM_PITCH_DEG,
    CAM_OFFSET_FORWARD_M,
    UPPER_ARM_M,
    FOREARM_M,
    WRIST_TO_TIP_M,
    SHOULDER_HEIGHT_M,
    SHOULDER_FORWARD_M,
    GRIPPER_APPROACH_DEG,
    CAP_CENTER_HEIGHT_M,
    SHOULDER_ID,
    ELBOW_ID,
    WRIST_ID,
    REST_ANGLES_DEG,
    JOINT_SIGNS,
)

_RAW_PER_DEG = 4096.0 / 360.0


def target_from_camera(v_px, z_m):
    """Project (pixel row, depth) to (forward, height) relative to the ARM
    BASE on the desk, using the measured camera mounting geometry."""
    yc = (v_px - CAMERA_HEIGHT / 2.0) * z_m / FX_PIXELS   # metres below axis
    ray = math.hypot(z_m, yc)
    ang_below = math.radians(CAM_PITCH_DEG) + math.atan2(yc, z_m)
    forward = CAM_OFFSET_FORWARD_M + ray * math.cos(ang_below)
    height = CAM_HEIGHT_M - ray * math.sin(ang_below)
    return forward, height


def target_on_desk(z_m):
    """Forward distance + height of a target KNOWN to stand on the desk,
    from depth alone. Needs only CAM_HEIGHT_M (lens above desk) and
    CAM_OFFSET_FORWARD_M — no camera pitch calibration at all: the vertical
    drop from lens to cap is known, so Pythagoras gives the horizontal
    distance."""
    drop = CAM_HEIGHT_M - CAP_CENTER_HEIGHT_M
    horiz = math.sqrt(max(0.0, z_m * z_m - drop * drop))
    return CAM_OFFSET_FORWARD_M + horiz, CAP_CENTER_HEIGHT_M


def solve_arm(x_target, z_target):
    """Planar IK to put the gripper TIP at (x_target, z_target), approaching
    at GRIPPER_APPROACH_DEG. Returns (shoulder_deg, elbow_deg, wrist_deg,
    clamped) — clamped=True means the point was out of reach and the nearest
    reachable pose was used instead."""
    g = math.radians(GRIPPER_APPROACH_DEG)
    # The wrist-flex axis must sit WRIST_TO_TIP_M back from the target along
    # the approach direction.
    wx = x_target - WRIST_TO_TIP_M * math.cos(g)
    wz = z_target - WRIST_TO_TIP_M * math.sin(g)
    # Relative to the shoulder axis.
    dx = wx - SHOULDER_FORWARD_M
    dz = wz - SHOULDER_HEIGHT_M
    r = math.hypot(dx, dz)
    clamped = False
    r_max = UPPER_ARM_M + FOREARM_M - 0.005
    r_min = abs(UPPER_ARM_M - FOREARM_M) + 0.005
    if r > r_max:
        r, clamped = r_max, True
    elif r < r_min:
        r, clamped = r_min, True
    # Interior elbow angle (law of cosines; 180 deg = straight).
    cos_e = (UPPER_ARM_M ** 2 + FOREARM_M ** 2 - r * r) / (2 * UPPER_ARM_M * FOREARM_M)
    elbow = math.acos(max(-1.0, min(1.0, cos_e)))
    # Elbow-up shoulder solution.
    alpha = math.atan2(dz, dx)
    cos_gamma = (UPPER_ARM_M ** 2 + r * r - FOREARM_M ** 2) / (2 * UPPER_ARM_M * r)
    gamma = math.acos(max(-1.0, min(1.0, cos_gamma)))
    shoulder = alpha + gamma
    # Forearm absolute angle, then the wrist offset that points the gripper
    # along the approach direction.
    forearm_abs = shoulder - (math.pi - elbow)
    wrist = g - forearm_abs
    return (math.degrees(shoulder), math.degrees(elbow),
            math.degrees(wrist), clamped)


def angles_to_raw(rest_pose, shoulder_deg, elbow_deg, wrist_deg):
    """Convert solved angles to servo raw goals, referenced to the captured
    rest pose and the user-calibrated REST_ANGLES_DEG / JOINT_SIGNS."""
    goals = {}
    for sid, angle in ((SHOULDER_ID, shoulder_deg),
                       (ELBOW_ID, elbow_deg),
                       (WRIST_ID, wrist_deg)):
        goals[sid] = int(round(
            rest_pose[sid]
            + JOINT_SIGNS[sid] * (angle - REST_ANGLES_DEG[sid]) * _RAW_PER_DEG
        ))
    return goals
