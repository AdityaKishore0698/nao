"""
gestures.py — the six gestures of Fears et al. (2023), Appendix B, for NAO V4
=============================================================================
    Bump       one arm extending forward, fist closed, palm down        (unimanual)
    Give       one arm extending forward, open hand, palm up            (unimanual)
    Wave       arm up and to the side, elbow out-in-out-in-out, 40 deg  (unimanual)
    Celebrate  both arms up and out, open hands, palms forward          (bimanual)
    Hug        both arms out and forward, palms in                      (bimanual)
    What       both arms out and upward, open hands, palms up           (bimanual)

Poses are written for the LEFT arm; the right arm is the mirror image.
Unimanual gestures are run once with each arm; bimanual with both arms.

Angle values are starting points derived from NAO kinematics — check each one
on the real robot with `scripts/demo_gestures.py` and adjust here. Palm
orientation (WristYaw / Hand) is cosmetic: it does not change the four angles
that are scored.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .nao_kinematics import clamp

Pose = Dict[str, float]          # left-arm joint name without the "L" -> angle (rad)

# Neutral pose between gestures: arms down, forearms slightly forward (NAO "Init")
READY: Pose = {"ShoulderPitch": 1.45, "ShoulderRoll": 0.15, "ElbowYaw": -1.20,
               "ElbowRoll": -0.50, "WristYaw": 0.0, "Hand": 0.3}

# TUNE ON ROBOT: wrist yaw that turns the palm down / up / forward with the arm raised
PALM_DOWN, PALM_UP, PALM_FORWARD = -1.5, 1.5, 0.0


def _pose(**kw) -> Pose:
    p = dict(READY)
    p.update(kw)
    return p


@dataclass
class Gesture:
    name: str
    description: str
    bimanual: bool
    cycle: List[Tuple[float, Pose]]          # one repetition: (seconds, target pose)
    prep: Optional[Pose] = None               # pose held before / between repetitions
    notes: str = ""


_BUMP_PREP = _pose(ShoulderPitch=0.80, ShoulderRoll=0.05, ElbowYaw=-1.0, ElbowRoll=-1.40,
                   WristYaw=PALM_DOWN, Hand=0.0)
_GIVE_PREP = _pose(ShoulderPitch=0.85, ShoulderRoll=0.05, ElbowYaw=-1.0, ElbowRoll=-1.35,
                   WristYaw=PALM_UP, Hand=1.0)
_WAVE_IN = _pose(ShoulderPitch=-0.25, ShoulderRoll=0.60, ElbowYaw=-1.55, ElbowRoll=-1.20,
                 WristYaw=PALM_FORWARD, Hand=1.0)
_WAVE_OUT = dict(_WAVE_IN, ElbowRoll=-1.20 + 0.70)          # 40 deg elbow extension
_HUG_OPEN = _pose(ShoulderPitch=0.25, ShoulderRoll=1.10, ElbowYaw=-1.0, ElbowRoll=-0.25,
                  WristYaw=0.0, Hand=1.0)

GESTURES: Dict[str, Gesture] = {
    "bump": Gesture(
        "bump", "One arm extending forward, fist closed, palm facing down", False,
        prep=_BUMP_PREP,
        cycle=[(1.5, dict(_BUMP_PREP, ShoulderPitch=0.0, ShoulderRoll=0.0, ElbowRoll=-0.05)),
               (1.5, _BUMP_PREP)]),
    "give": Gesture(
        "give", "One arm extending forward, open hand, palm facing up", False,
        prep=_GIVE_PREP,
        cycle=[(1.5, dict(_GIVE_PREP, ShoulderPitch=0.15, ShoulderRoll=0.0, ElbowRoll=-0.05)),
               (1.5, _GIVE_PREP)]),
    "wave": Gesture(
        "wave", "One arm up and to the side, moving back and forth, open hand, palm forward",
        False,
        # Paper: raise arm, elbow out-in-out-in-out (40 deg), return; 7.5 s in total
        cycle=[(1.5, _WAVE_IN),
               (0.9, _WAVE_OUT), (0.9, _WAVE_IN), (0.9, _WAVE_OUT), (0.9, _WAVE_IN),
               (0.9, _WAVE_OUT),
               (1.5, READY)]),
    "celebrate": Gesture(
        "celebrate", "Both arms extending up and out, open hands, palms facing forward", True,
        cycle=[(1.5, _pose(ShoulderPitch=-1.30, ShoulderRoll=0.45, ElbowYaw=-1.2,
                           ElbowRoll=-0.15, WristYaw=PALM_FORWARD, Hand=1.0)),
               (1.5, READY)]),
    "hug": Gesture(
        "hug", "Both arms extending out and forward, palms facing in", True,
        prep=_HUG_OPEN,
        cycle=[(1.5, dict(_HUG_OPEN, ShoulderPitch=0.20, ShoulderRoll=0.05, ElbowRoll=-0.90)),
               (1.5, _HUG_OPEN)]),
    "what": Gesture(
        "what", "Both arms extending out and upward, open hands, palms facing up", True,
        cycle=[(1.5, _pose(ShoulderPitch=0.70, ShoulderRoll=0.60, ElbowYaw=-0.30,
                           ElbowRoll=-1.20, WristYaw=PALM_UP, Hand=1.0)),
               (1.5, READY)]),
}

GESTURE_ORDER = ("bump", "give", "wave", "celebrate", "hug", "what")

_MIRRORED = ("ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw")


def side_pose(pose: Pose, side: str) -> Dict[str, float]:
    """Left-arm pose -> full joint names for `side` ("L"/"R"), mirrored and clamped."""
    out = {}
    for j, v in pose.items():
        if side == "R" and j in _MIRRORED:
            v = -v
        name = f"{side}{j}"
        out[name] = clamp(name, v)
    return out


@dataclass
class Trajectory:
    """Absolute-time keyframes, ready for ALMotion.angleInterpolation(isAbsolute=True)."""
    names: List[str]
    angles: List[List[float]]
    times: List[List[float]]
    rep_starts: List[float] = field(default_factory=list)    # seconds, for segmentation

    @property
    def duration(self) -> float:
        return self.times[0][-1] if self.times and self.times[0] else 0.0


def build_trajectory(gesture: Gesture, arms: str, reps: int = 5, speed: float = 1.0,
                     rest_between_reps: float = 1.0, transition: float = 1.5) -> Trajectory:
    """
    arms: "L", "R" or "both" (robot arms that move).
    speed < 1 slows everything down. A short hold between repetitions gives the
    child a clear end-of-repetition (the paper's segmentation limitation).
    """
    sides = ["L", "R"] if arms == "both" else [arms]
    frames: List[Tuple[float, Pose]] = []
    t = 0.0
    rep_starts = []

    def add(dt, pose):
        nonlocal t
        t += dt / speed
        frames.append((t, pose))

    if gesture.prep is not None:
        add(transition, gesture.prep)
    for r in range(reps):
        rep_starts.append(t)
        for dt, pose in gesture.cycle:
            add(dt, pose)
        if r < reps - 1 and rest_between_reps > 0:
            add(rest_between_reps, frames[-1][1])          # hold
    if frames[-1][1] is not READY:
        add(transition, READY)

    names, angles, times = [], [], []
    for side in sides:
        per_frame = [side_pose(p, side) for _, p in frames]
        for name in per_frame[0]:
            names.append(name)
            angles.append([pf[name] for pf in per_frame])
            times.append([ft for ft, _ in frames])
    return Trajectory(names, angles, times, rep_starts)


def sample_trajectory(traj: Trajectory, t: float, start: Dict[str, float]) -> List[float]:
    """
    Target angle of every joint in traj.names at time t (seconds from start),
    easing between keyframes with a minimum-jerk profile (smooth start/stop,
    like NAO's own angleInterpolation). `start` holds each joint's angle at t=0.
    """
    out = []
    for name, ang, tm in zip(traj.names, traj.angles, traj.times):
        kt = [0.0] + list(tm)
        kv = [start[name]] + list(ang)
        if t >= kt[-1]:
            out.append(kv[-1])
            continue
        i = max(k for k in range(len(kt) - 1) if kt[k] <= t)
        s = (t - kt[i]) / (kt[i + 1] - kt[i])
        s = s * s * s * (10 - 15 * s + 6 * s * s)
        out.append(kv[i] + (kv[i + 1] - kv[i]) * s)
    return out
