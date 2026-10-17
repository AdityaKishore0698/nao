"""
check_camera.py — live RealSense + MediaPipe preview with the four arm angles
============================================================================
Check camera placement before a session: both shoulders, elbows, wrists and
hips should be tracked, and the angles should read ~0 with arms hanging,
~90 for flexion with arms straight forward, ~90 for abduction straight out.

    python scripts/check_camera.py            # press q to quit
    python scripts/check_camera.py --list     # list connected RealSense devices
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from imitation_eval.camera import CAMERA_UP, RealSenseCamera, list_devices  # noqa: E402
from imitation_eval.joint_angles import ANGLE_NAMES, all_arm_angles  # noqa: E402
from imitation_eval.pose3d import POINT_NAMES, ArmPoseEstimator  # noqa: E402

SHORT = {"shoulder_flex_ext": "flex", "shoulder_abd_add": "abd",
         "shoulder_rotation": "rot", "elbow_flex_ext": "elbow"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        devs = list_devices()
        print("\n".join(f"{n}  serial {s}" for n, s in devs) or "No RealSense device found.")
        return

    cfg = config.load()
    est = ArmPoseEstimator()
    with RealSenseCamera(cfg["camera_width"], cfg["camera_height"], cfg["camera_fps"],
                         cfg["camera_serial"]) as cam:
        while True:
            color, depth = cam.get_frames()
            if color is None:
                continue
            pts, res = est.process(color, depth, cam)
            frame = est.draw(color, res)
            d = {n: pts[i][None] for i, n in enumerate(POINT_NAMES)}
            d["mid_hip"] = (d["l_hip"] + d["r_hip"]) / 2
            ang = all_arm_angles(d, fallback_up=CAMERA_UP)
            y = 20
            tracked = int(np.isfinite(pts).all(axis=1).sum())
            cv2.putText(frame, f"tracked {tracked}/{len(POINT_NAMES)}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for side in "LR":
                y += 20
                txt = "  ".join(f"{SHORT[a]} {ang[side][a][0]:5.0f}" if np.isfinite(ang[side][a][0])
                                else f"{SHORT[a]}   --" for a in ANGLE_NAMES)
                cv2.putText(frame, f"{side}: {txt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
            cv2.imshow("check_camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    est.close()


if __name__ == "__main__":
    main()
