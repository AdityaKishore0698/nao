"""
simulate.py — synthetic "child" for dry runs and tests (no camera needed)
========================================================================
Builds a ChildRecording from a RobotRecording as a mirror-imitating child
would produce it, expressed in RealSense camera coordinates, so the full
child pipeline (trunk frame, angles, mirroring, DTW) is exercised:

    robot 3D points --reflect left/right (mirror imitation)--> child body
    --scale to child size, delay, slow down, add noise--> camera frame
"""

import numpy as np

from .nao_kinematics import robot_points
from .pose3d import ChildRecording, POINT_NAMES

# child body frame (x fwd, y left, z up) -> camera (x right, y down, z away),
# child standing facing the camera: fwd -> -z, left -> +x, up -> -y
BODY_TO_CAMERA = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0]])


def simulate_child(robot_rec, delay_s=0.5, time_scale=1.1, noise_m=0.01, scale=3.0,
                   distance_m=2.5, dropout=0.0, tail_s=2.0, fs=30.0, seed=0,
                   amplitude=1.0, poor=False):
    """
    amplitude < 1 shrinks the child's movement (less accurate imitation).
    poor=True replaces the arm movement with unrelated random motion.
    """
    rng = np.random.default_rng(seed)
    pts = robot_points(robot_rec.as_dict())
    # mirror imitation: child's right arm does what the robot's left arm does
    mirrored = {}
    for k, v in pts.items():
        v = v.copy()
        v[:, 1] *= -1.0
        if k.startswith("l_"):
            k = "r_" + k[2:]
        elif k.startswith("r_"):
            k = "l_" + k[2:]
        mirrored[k] = v

    # shrink the upper-arm movement towards hanging; the forearm moves with the elbow
    if amplitude != 1.0:
        rest_dir = np.array([0.0, 0.0, -1.0])
        for side in "lr":
            sh = mirrored[f"{side}_shoulder"]
            elbow = mirrored[f"{side}_elbow"]
            forearm = mirrored[f"{side}_wrist"] - elbow
            rel = elbow - sh
            n = np.linalg.norm(rel, axis=1, keepdims=True)
            d = (1 - amplitude) * rest_dir + amplitude * rel / n
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            mirrored[f"{side}_elbow"] = sh + d * n
            mirrored[f"{side}_wrist"] = mirrored[f"{side}_elbow"] + forearm

    t_r = robot_rec.t - robot_rec.t[0]
    t_c = np.arange(0.0, t_r[-1] + tail_s, 1.0 / fs)
    src_t = np.clip((t_c - delay_s) / time_scale, 0.0, t_r[-1])

    hip = mirrored["mid_hip"]
    out = np.full((len(t_c), len(POINT_NAMES), 3), np.nan)
    for i, name in enumerate(POINT_NAMES):
        if name in ("l_hip", "r_hip"):
            base = hip + np.array([0.0, 0.05 if name == "l_hip" else -0.05, 0.0])
        else:
            base = mirrored[name]
        traj = np.stack([np.interp(src_t, t_r, base[:, d]) for d in range(3)], axis=1)
        if poor and name.endswith(("elbow", "wrist")):
            walk = np.cumsum(rng.normal(0, 0.01, size=traj.shape), axis=0)
            traj = traj[0] + walk
        body = traj * scale
        cam = body @ BODY_TO_CAMERA.T + np.array([0.0, 0.0, distance_m])
        out[:, i] = cam + rng.normal(0.0, noise_m, size=cam.shape)
    if dropout > 0:
        mask = rng.random(out.shape[:2]) < dropout
        out[mask] = np.nan
    return ChildRecording(robot_rec.t[0] + t_c, out)
