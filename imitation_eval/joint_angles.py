"""
joint_angles.py — the paper's four arm angles from 3D joint positions
====================================================================
Fears et al. (2023) compared robot and participant on, per arm:
    shoulder flexion/extension, shoulder abduction/adduction,
    shoulder internal/external rotation, elbow flexion/extension.

The same function is used for NAO (points from forward kinematics) and for the
child (points from RealSense + MediaPipe), so both are measured identically.

Conventions (degrees, anatomical, identical for left and right arms):
    shoulder_flex_ext   0 = arm hanging, +90 = forward horizontal, 180 = overhead,
                        negative = extension behind the trunk  (sagittal-plane projection)
    shoulder_abd_add    0 = arm hanging, +90 = out to the side,  180 = overhead,
                        negative = adduction across the body    (frontal-plane projection)
    shoulder_rotation   0 = forearm pointing forward with the arm hanging,
                        positive = external rotation (forearm turns outward)
    elbow_flex_ext      0 = straight arm, positive = flexion

Frames where an angle is geometrically undefined (e.g. rotation with a straight
elbow, flexion with the arm straight out to the side) are returned as NaN; the
scoring step interpolates over them.
"""

import numpy as np

ANGLE_NAMES = ("shoulder_flex_ext", "shoulder_abd_add", "shoulder_rotation", "elbow_flex_ext")

# Minimum projected length (as a fraction of a unit vector) for an angle to be defined
MIN_PROJECTION = 0.25          # ~14 deg away from the plane's normal
MIN_FOREARM_PERP = np.sin(np.radians(20.0))   # rotation needs >= 20 deg elbow flexion
MAX_OVERHEAD_COS = -0.9        # rotation reference is undefined with the arm straight up


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


def _projected_angle(along, across, normal_component):
    """atan2 in a plane, NaN when the vector is close to the plane's normal.
    Range is [-90, 270) so that overhead (180) does not wrap to -180."""
    ang = np.degrees(np.arctan2(along, across))
    ang = np.where(ang < -90.0, ang + 360.0, ang)
    in_plane = np.sqrt(along ** 2 + across ** 2)
    return np.where(in_plane >= MIN_PROJECTION, ang, np.nan)


def _minimal_rotation(v, src, dst):
    """Rotate v by the smallest rotation taking unit src onto unit dst (row-wise)."""
    axis = np.cross(src, dst)
    sin_t = np.linalg.norm(axis, axis=-1)
    cos_t = _dot(src, dst)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = axis / sin_t[:, None]
    k = np.where(sin_t[:, None] > 1e-9, k, 0.0)
    return (v * cos_t[:, None]
            + np.cross(k, v) * sin_t[:, None]
            + k * _dot(k, v)[:, None] * (1.0 - cos_t[:, None]))


def arm_angles(side, shoulder, elbow, wrist, frame):
    """
    side: "L" or "R"; shoulder/elbow/wrist: (T, 3); frame: (forward, left, up).
    Returns dict angle name -> (T,) array in degrees (NaN where undefined).
    """
    forward, left, up = frame
    out = left if side == "L" else -left       # lateral, away from the body
    upper = _normalize(elbow - shoulder)
    fore = _normalize(wrist - elbow)

    u_f, u_o, u_z = _dot(upper, forward), _dot(upper, out), _dot(upper, up)
    flex = _projected_angle(u_f, -u_z, u_o)
    abd = _projected_angle(u_o, -u_z, u_f)

    elbow_flex = np.degrees(np.arccos(np.clip(_dot(upper, fore), -1.0, 1.0)))

    # Rotation: compare the forearm with "forward", carried along with the upper
    # arm by the minimal rotation from the hanging position (swing-twist).
    ref = _normalize(_minimal_rotation(forward, -up, upper))
    fore_perp = fore - _dot(fore, upper)[:, None] * upper
    perp_len = np.linalg.norm(fore_perp, axis=-1)
    fore_perp = _normalize(fore_perp)
    sign = -1.0 if side == "L" else 1.0        # makes external rotation positive on both arms
    rot = np.degrees(np.arctan2(sign * _dot(np.cross(ref, fore_perp), upper),
                                _dot(ref, fore_perp)))
    rot_ok = (perp_len >= MIN_FOREARM_PERP) & (_dot(upper, up) < -MAX_OVERHEAD_COS)
    rot = np.where(rot_ok, rot, np.nan)

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
