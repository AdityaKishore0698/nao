"""
scoring.py — robot vs child joint-angle DTW, per arm and per angle
=================================================================
Pipeline for one trial (one gesture, one robot arm or both):

    robot sensor angles --FK--> 3D points --+
                                            +--> 4 anatomical angles per arm
    child RealSense/MediaPipe 3D points ----+
        -> resample to a common rate -> fill short gaps -> 6 Hz low-pass
        -> z-normalise -> DTW(robot arm, mirrored child arm) -> cost per angle

Mirror imitation (paper's instruction): when NAO moves its LEFT arm the child
moves their RIGHT arm, so robot "L" is compared with child "R" by default.
For unimanual gestures only the moving arm pair is scored (paper, 2.6).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .camera import CAMERA_UP
from .dtw import dtw_znorm
from .joint_angles import ANGLE_NAMES, all_arm_angles
from .nao_kinematics import robot_points
from .signal_utils import fill_nans, lowpass, resample_uniform

MIRROR = {"L": "R", "R": "L"}


@dataclass
class ScoringConfig:
    fs: float = 30.0                 # common sample rate (Hz)
    cutoff_hz: float = 6.0           # low-pass cutoff
    min_valid_fraction: float = 0.6  # below this the angle is not scored (NaN cost)
    mirror: bool = True
    dtw_window_s: float = None       # optional Sakoe-Chiba band (seconds)


def robot_angle_series(rec):
    """RobotRecording -> (t, {side: {angle: (T,)}})"""
    return rec.t, all_arm_angles(robot_points(rec.as_dict()))


def child_angle_series(rec):
    """ChildRecording -> (t, {side: {angle: (T,)}})"""
    return rec.t, all_arm_angles(rec.as_points(), fallback_up=CAMERA_UP)


def _prepare(t, x, t0, t1, cfg):
    _, xu = resample_uniform(t, x, cfg.fs, t0, t1)
    filled, frac = fill_nans(xu)
    if frac == 0.0:
        return filled, 0.0
    return lowpass(filled, cfg.fs, cfg.cutoff_hz), frac


def score_trial(robot_rec, child_rec, robot_arms: List[str], cfg: ScoringConfig = None,
                tail_s: float = 2.0) -> List[Dict]:
    """
    robot_arms: arms NAO moved, e.g. ["L"] or ["L", "R"].
    The child's window runs from the robot's first sample to its last + tail_s
    (children lag the demonstrator).
    Returns one row per (robot arm, angle).
    """
    cfg = cfg or ScoringConfig()
    t_r, robot = robot_angle_series(robot_rec)
    t_c, child = child_angle_series(child_rec)
    r0, r1 = t_r[0], t_r[-1]
    window = None if cfg.dtw_window_s is None else int(cfg.dtw_window_s * cfg.fs)

    rows = []
    for r_arm in robot_arms:
        c_arm = MIRROR[r_arm] if cfg.mirror else r_arm
        for ang in ANGLE_NAMES:
            r_sig, r_frac = _prepare(t_r, robot[r_arm][ang], r0, r1, cfg)
            c_sig, c_frac = _prepare(t_c, child[c_arm][ang], r0, r1 + tail_s, cfg)
            row = {"robot_arm": r_arm, "child_arm": c_arm, "angle": ang,
                   "robot_valid_frac": round(r_frac, 3), "child_valid_frac": round(c_frac, 3),
                   "n_robot": len(r_sig), "n_child": len(c_sig),
                   "robot_range_deg": float(np.nanmax(r_sig) - np.nanmin(r_sig)) if r_frac else np.nan,
                   "child_range_deg": float(np.nanmax(c_sig) - np.nanmin(c_sig)) if c_frac else np.nan}
            if (r_frac < cfg.min_valid_fraction or c_frac < cfg.min_valid_fraction
                    or len(r_sig) < 2 or len(c_sig) < 2):
                row.update(dtw_cost=np.nan, dtw_cost_per_step=np.nan, path_length=0,
                           scored=False)
            else:
                row.update(dtw_znorm(r_sig, c_sig, window=window), scored=True)
            rows.append(row)
    return rows
