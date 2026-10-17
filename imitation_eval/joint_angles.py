"""
joint_angles.py — the paper's four arm angles from 3D joint positions
====================================================================
Fears et al. (2023) compared robot and participant on, per arm:
    shoulder flexion/extension, shoulder abduction/adduction,
    shoulder internal/external rotation, elbow flexion/extension.

The same function is used for NAO (points from forward kinematics) and for the
child (points from RealSense + MediaPipe), so both are measured identically.

Shoulder angles use a Cardan sequence of the humerus relative to the trunk —
flexion/extension first, then abduction/adduction, then axial rotation — the
usual clinical-biomechanics (Visual3D-style) decomposition. With d the unit
upper-arm vector (shoulder -> elbow) in the trunk frame (out, forward, up):

    shoulder_flex_ext   atan2(d.forward, -d.up): 0 = hanging, +90 = forward
                        horizontal, 180 = overhead, negative = extension
    shoulder_abd_add    asin(d.out): elevation out of the sagittal plane,
                        +90 = straight out to the side, negative = adduction
    shoulder_rotation   forearm direction about the humerus, relative to the
                        Cardan-rotated forward axis; 0 = forearm forward with the
                        arm hanging, positive = external rotation
    elbow_flex_ext      0 = straight arm, positive = flexion

All angles in degrees and identical in meaning for both arms. Frames where an
angle is undefined (rotation with a straight elbow; flexion and rotation with
the arm straight out to the side = gimbal lock) are NaN; scoring interpolates
over them.
"""

import numpy as np

ANGLE_NAMES = ("shoulder_flex_ext", "shoulder_abd_add", "shoulder_rotation", "elbow_flex_ext")

MAX_ABDUCTION_SIN = np.sin(np.radians(80.0))   # beyond this flexion/rotation are gimbal-locked
MIN_FOREARM_PERP = np.sin(np.radians(20.0))    # rotation needs >= 20 deg elbow flexion


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def trunk_frame(l_shoulder, r_shoulder, mid_hip, fallback_up=None):
    """
    Returns (forward, left, up) unit vectors, each (T, 3).

    up is mid-hip -> mid-shoulder. If mid_hip is missing (NaN) for a frame and
    fallback_up is given, that direction is used instead (e.g. camera "up").
    """
    mid_sh = (l_shoulder + r_shoulder) / 2.0
    up = _normalize(mid_sh - mid_hip)
    if fallback_up is not None:
        bad = ~np.isfinite(up).all(axis=-1)
        up[bad] = np.asarray(fallback_up, dtype=float) / np.linalg.norm(fallback_up)
    left = l_shoulder - r_shoulder
    left = _normalize(left - _dot(left, up)[:, None] * up)
    forward = np.cross(left, up)
    return forward, left, up


def arm_angles(side, shoulder, elbow, wrist, frame):
    """
    side: "L" or "R"; shoulder/elbow/wrist: (T, 3); frame: (forward, left, up).
    Returns dict angle name -> (T,) array in degrees (NaN where undefined).
    """
    forward, left, up = frame
    out = left if side == "L" else -left       # lateral, away from the body
    upper = _normalize(elbow - shoulder)
    fore = _normalize(wrist - elbow)

    d_f, d_o, d_u = _dot(upper, forward), _dot(upper, out), _dot(upper, up)
    not_locked = np.abs(d_o) <= MAX_ABDUCTION_SIN

    # 1) flexion about the lateral axis; range [-90, 270) so overhead doesn't wrap
    theta = np.arctan2(d_f, -d_u)
    flex = np.degrees(theta)
    flex = np.where(flex < -90.0, flex + 360.0, flex)
    flex = np.where(not_locked, flex, np.nan)
    # 2) abduction about the flexed forward axis
    abd = np.degrees(np.arcsin(np.clip(d_o, -1.0, 1.0)))

    elbow_flex = np.degrees(np.arccos(np.clip(_dot(upper, fore), -1.0, 1.0)))

    # 3) axial rotation: forearm vs the forward axis after flexion + abduction
    #    (cos(theta) forward + sin(theta) up; abduction rotates about it, so it is unchanged)
    ref = (np.cos(theta)[:, None] * forward + np.sin(theta)[:, None] * up)
    fore_perp = fore - _dot(fore, upper)[:, None] * upper
    perp_len = np.linalg.norm(fore_perp, axis=-1)
    fore_perp = _normalize(fore_perp)
    sign = -1.0 if side == "L" else 1.0        # makes external rotation positive on both arms
    rot = np.degrees(np.arctan2(sign * _dot(np.cross(ref, fore_perp), upper),
                                _dot(ref, fore_perp)))
    rot = np.where((perp_len >= MIN_FOREARM_PERP) & not_locked, rot, np.nan)

    return {
        "shoulder_flex_ext": flex,
        "shoulder_abd_add": abd,
        "shoulder_rotation": rot,
        "elbow_flex_ext": elbow_flex,
    }


def all_arm_angles(points, fallback_up=None):
    """
    points: dict with (T, 3) arrays l_/r_shoulder, l_/r_elbow, l_/r_wrist, mid_hip.
    Returns {"L": {angle: (T,)}, "R": {angle: (T,)}}  (anatomical sides).
    """
    frame = trunk_frame(points["l_shoulder"], points["r_shoulder"], points["mid_hip"],
                        fallback_up=fallback_up)
    return {
        side: arm_angles(side, points[f"{k}_shoulder"], points[f"{k}_elbow"],
                         points[f"{k}_wrist"], frame)
        for side, k in (("L", "l"), ("R", "r"))
    }
