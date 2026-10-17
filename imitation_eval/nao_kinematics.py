"""
nao_kinematics.py — NAO V4 arm forward kinematics
=================================================
Turns NAO arm joint angles into 3D shoulder / elbow / wrist positions so the
robot's movement can be put through exactly the same joint-angle code as the
child's camera data (the paper captured both with the same marker set).

Frame: NAO torso frame — x forward, y left, z up, metres.
Link lengths: Aldebaran NAO V4 (H25) documentation.
"""

import numpy as np

SHOULDER_OFFSET_Y = 0.098
SHOULDER_OFFSET_Z = 0.100
UPPER_ARM_LENGTH = 0.105
ELBOW_OFFSET_Y = 0.015
LOWER_ARM_LENGTH = 0.05595
# Pseudo hip centre used to define the trunk's vertical axis
HIP_CENTER = np.array([0.0, 0.0, -0.085])

# Joint limits (rad) for NAO V4, left arm; right arm is mirrored (see mirror_limits)
LEFT_ARM_LIMITS = {
    "LShoulderPitch": (-2.0857, 2.0857),
    "LShoulderRoll": (-0.3142, 1.3265),
    "LElbowYaw": (-2.0857, 2.0857),
    "LElbowRoll": (-1.5446, -0.0349),
    "LWristYaw": (-1.8238, 1.8238),
    "LHand": (0.0, 1.0),
}


def _mirror_limits(left_limits):
    """Right-arm limits: roll / yaw joints have their ranges negated."""
    out = {}
    for name, (lo, hi) in left_limits.items():
        rname = "R" + name[1:]
        if name.endswith(("ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw")):
            out[rname] = (-hi, -lo)
        else:
            out[rname] = (lo, hi)
    return out


JOINT_LIMITS = {**LEFT_ARM_LIMITS, **_mirror_limits(LEFT_ARM_LIMITS)}

# The four motors that determine shoulder / elbow / wrist positions
ARM_JOINTS = ("ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll")
SENSED_JOINTS = tuple(f"{s}{j}" for s in "LR" for j in ARM_JOINTS)


def clamp(name, value):
    lo, hi = JOINT_LIMITS[name]
    return float(min(max(value, lo), hi))


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1),
                     np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1),
                     np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1),
                     np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def arm_fk(side, shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll):
    """
    Forward kinematics for one arm. Angles are scalars or (T,) arrays (rad).

    Returns (shoulder, elbow, wrist), each (T, 3) in the torso frame.
    """
    s = 1.0 if side == "L" else -1.0
    sp, sr, ey, er = (np.atleast_1d(np.asarray(a, dtype=float))
                      for a in (shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll))
    T = sp.shape[0]

    shoulder = np.tile([0.0, s * SHOULDER_OFFSET_Y, SHOULDER_OFFSET_Z], (T, 1))
    R_sh = _rot_y(sp) @ _rot_z(sr)
    elbow = shoulder + R_sh @ np.array([UPPER_ARM_LENGTH, s * ELBOW_OFFSET_Y, 0.0])
    R_el = R_sh @ _rot_x(ey) @ _rot_z(er)
    wrist = elbow + R_el @ np.array([LOWER_ARM_LENGTH, 0.0, 0.0])
    return shoulder, elbow, wrist


def robot_points(angles):
    """
    angles: dict joint name -> (T,) array, containing all SENSED_JOINTS.
    Returns dict of (T, 3) arrays with the same keys the human pose code uses.
    """
    pts = {}
    for side in "LR":
        sh, el, wr = arm_fk(side, *(angles[f"{side}{j}"] for j in ARM_JOINTS))
        k = side.lower()
        pts[f"{k}_shoulder"], pts[f"{k}_elbow"], pts[f"{k}_wrist"] = sh, el, wr
    T = pts["l_shoulder"].shape[0]
    pts["mid_hip"] = np.tile(HIP_CENTER, (T, 1))
    return pts
