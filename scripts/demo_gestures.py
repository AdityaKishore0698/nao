"""
demo_gestures.py — perform gestures and print the robot's own joint-angle ranges
================================================================================
Use this to tune imitation_eval/gestures.py on the real robot: each gesture
should move the joints the paper says it does (e.g. bump/give = shoulder
flexion + elbow extension, wave = 40 deg elbow oscillation).

    python scripts/demo_gestures.py --sim                  # kinematics only
    python scripts/demo_gestures.py --gestures wave --reps 1   # on NAO
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from imitation_eval.gestures import GESTURE_ORDER, GESTURES, build_trajectory  # noqa: E402
from imitation_eval.joint_angles import ANGLE_NAMES  # noqa: E402
from imitation_eval.scoring import robot_angle_series  # noqa: E402


def summarize(rec, arms):
    _, ang = robot_angle_series(rec)
    for arm in (["L", "R"] if arms == "both" else [arms]):
        for name in ANGLE_NAMES:
            x = ang[arm][name]
            ok = np.isfinite(x)
            if ok.any():
                print(f"    {arm} {name:<18} min {np.nanmin(x):7.1f}  max {np.nanmax(x):7.1f}"
                      f"  range {np.nanmax(x) - np.nanmin(x):6.1f} deg   defined {ok.mean():4.0%}")
            else:
                print(f"    {arm} {name:<18} undefined throughout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gestures", nargs="+", default=list(GESTURE_ORDER), choices=GESTURE_ORDER)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--arm", choices=["L", "R"], default="L", help="arm for unimanual gestures")
    p.add_argument("--sim", action="store_true")
    args = p.parse_args()

    cfg = config.load()
    if args.sim:
        from imitation_eval.robot import SimulatedRobot
        robot = SimulatedRobot(verbose=False)
    else:
        from imitation_eval.robot import NaoRobot
        robot = NaoRobot(cfg["nao_ip"], cfg["nao_port"])
        robot.prepare(cfg["nao_posture"])
    try:
        for g in args.gestures:
            arms = "both" if GESTURES[g].bimanual else args.arm
            traj = build_trajectory(GESTURES[g], arms, reps=args.reps, speed=args.speed)
            print(f"\n{g} ({arms}), {traj.duration:.1f} s — {GESTURES[g].description}")
            rec = robot.perform(traj)
            summarize(rec, arms)
            robot.go_ready()
    finally:
        robot.finish(cfg["nao_posture"])


if __name__ == "__main__":
    main()
