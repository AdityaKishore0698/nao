"""
run_session.py — run one imitation assessment session
=====================================================
Experiment (simulated NAO on screen + RealSense):
    python scripts/run_session.py --participant P001 --group autistic --age 9

Robots (--robot):
    qibullet   physics-simulated NAO shown in a window (default, used for the experiments)
    kinematic  keyframe interpolation only, no window (fast dry runs, tests)
    nao        the physical NAO over qi

Dry runs:
    python scripts/run_session.py --participant TEST --robot kinematic --sim-child
    python scripts/run_session.py --participant TEST --sim-child          # qiBullet, no camera
    python scripts/run_session.py --participant TEST --gestures wave --reps 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from imitation_eval.gestures import GESTURE_ORDER  # noqa: E402
from imitation_eval.protocol import SessionConfig, run_session  # noqa: E402
from imitation_eval.scoring import ScoringConfig  # noqa: E402


def print_rows(gesture, arms, rows):
    print(f"\n[{gesture} / robot arm {arms}]")
    for r in rows:
        cost = "  n/a " if not r["scored"] else f"{r['dtw_cost']:8.1f}"
        print(f"  robot {r['robot_arm']} -> child {r['child_arm']}  {r['angle']:<18} "
              f"DTW {cost}   valid robot {r['robot_valid_frac']:.2f} child {r['child_valid_frac']:.2f}")


def main():
    cfg_file = config.load()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True, help="pseudonymous ID, e.g. P001")
    p.add_argument("--group", default="unknown", choices=["autistic", "neurotypical", "unknown"])
    p.add_argument("--age", type=float)
    p.add_argument("--gestures", nargs="+", default=list(GESTURE_ORDER), choices=GESTURE_ORDER)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0, help="<1 slows NAO down")
    p.add_argument("--no-mirror", action="store_true", help="compare same-side arms instead")
    p.add_argument("--robot", choices=["qibullet", "kinematic", "nao"], default=cfg_file["robot"])
    p.add_argument("--sim-child", action="store_true", help="no camera: synthetic child")
    p.add_argument("--nao-ip", default=cfg_file["nao_ip"])
    p.add_argument("--data-dir", default=cfg_file["data_dir"])
    p.add_argument("--save-video", action="store_true", help="also save annotated .mp4 (identifiable!)")
    args = p.parse_args()

    if args.robot == "kinematic":
        from imitation_eval.robot import SimulatedRobot
        robot = SimulatedRobot(realtime=not args.sim_child)
        posture = None
    elif args.robot == "qibullet":
        from imitation_eval.sim_robot import QiBulletNao
        robot = QiBulletNao(speech=cfg_file["sim_speech"])
        posture = cfg_file["sim_posture"]
    else:
        from imitation_eval.robot import NaoRobot
        robot = NaoRobot(args.nao_ip, cfg_file["nao_port"])
        posture = cfg_file["nao_posture"]
    if posture:
        robot.prepare(posture)

    camera = recorder = None
    if args.sim_child:
        from imitation_eval.simulate import simulate_child

        def start_child():
            pass

        def stop_child(robot_rec):
            return simulate_child(robot_rec)
    else:
        from imitation_eval.camera import RealSenseCamera
        from imitation_eval.pose3d import ArmPoseEstimator, ChildRecorder
        camera = RealSenseCamera(cfg_file["camera_width"], cfg_file["camera_height"],
                                 cfg_file["camera_fps"], cfg_file["camera_serial"]).start()
        writer = None
        if args.save_video:
            import cv2
            os.makedirs(args.data_dir, exist_ok=True)
            writer = cv2.VideoWriter(os.path.join(args.data_dir, f"{args.participant}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), cfg_file["camera_fps"],
                                     (cfg_file["camera_width"], cfg_file["camera_height"]))
        recorder = ChildRecorder(camera, ArmPoseEstimator(), video_writer=writer)

        def start_child():
            recorder.start()

        def stop_child(robot_rec):
            return recorder.stop()

    session = SessionConfig(participant=args.participant, group=args.group, age=args.age,
                            gestures=args.gestures, reps=args.reps, speed=args.speed,
                            data_dir=args.data_dir, mongo_uri=cfg_file["mongo_uri"])
    if args.sim_child and args.robot == "kinematic":
        session.lead_in_s = session.tail_s = session.pause_between_trials_s = 0.0
    try:
        out_dir, _ = run_session(session, robot, start_child, stop_child,
                                 ScoringConfig(mirror=not args.no_mirror),
                                 source="simulated" if args.sim_child else "live",
                                 robot_type=args.robot,
                                 on_trial=print_rows)
        print(f"\nSaved to {out_dir}")
    except KeyboardInterrupt:
        print("\nInterrupted — trials completed so far are saved.")
    finally:
        if recorder is not None:
            recorder.stop()
        if camera is not None:
            camera.stop()
        robot.finish(posture or "Sit")


if __name__ == "__main__":
    main()
