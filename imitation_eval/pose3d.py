"""
pose3d.py — child's 3D arm keypoints from MediaPipe Pose + RealSense depth
=========================================================================
MediaPipe finds the 2D landmarks in the colour image; the aligned depth image
gives their distance; deprojection gives 3D points in camera coordinates.
Only the landmarks needed for the four arm angles are kept.

ChildRecorder runs capture + pose in a background thread so it can record
while NAO is demonstrating. Every frame is timestamped with time.monotonic(),
the same clock RobotRecording uses.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

# MediaPipe BlazePose indices
LANDMARKS = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
}
POINT_NAMES = tuple(LANDMARKS)


def depth_at(depth, px, py, half=3, min_m=0.3, max_m=6.0, scale=0.001):
    """Median of valid depth values (metres) in a (2*half+1)^2 window; NaN if none."""
    h, w = depth.shape
    x0, x1 = max(0, px - half), min(w, px + half + 1)
    y0, y1 = max(0, py - half), min(h, py + half + 1)
    patch = depth[y0:y1, x0:x1].astype(float) * scale
    patch = patch[(patch > min_m) & (patch < max_m)]
    return float(np.median(patch)) if patch.size else float("nan")


class ArmPoseEstimator:
    def __init__(self, model_complexity=1, min_visibility=0.5):
        import mediapipe as mp
        self._mp = mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           model_complexity=model_complexity,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.min_visibility = min_visibility

    def process(self, color_bgr, depth, camera):
        """
        Returns (points (len(POINT_NAMES), 3) with NaN rows for missing joints,
                 mediapipe results for drawing).
        """
        import cv2
        results = self.pose.process(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        pts = np.full((len(POINT_NAMES), 3), np.nan)
        if not results.pose_landmarks:
            return pts, results
        lms = results.pose_landmarks.landmark
        h, w = depth.shape
        for i, name in enumerate(POINT_NAMES):
            lm = lms[LANDMARKS[name]]
            if lm.visibility < self.min_visibility:
                continue
            px = int(round(lm.x * w))
            py = int(round(lm.y * h))
            if not (0 <= px < w and 0 <= py < h):
                continue
            z = depth_at(depth, px, py, scale=camera.depth_scale)
            if np.isfinite(z):
                pts[i] = camera.deproject(px, py, z)
        return pts, results

    def draw(self, image, results):
        if results is not None and results.pose_landmarks:
            self._mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, self._mp.solutions.pose.POSE_CONNECTIONS)
        return image

    def close(self):
        self.pose.close()


@dataclass
class ChildRecording:
    t: np.ndarray            # (T,) host time.monotonic()
    points: np.ndarray       # (T, len(POINT_NAMES), 3) camera coordinates, metres (NaN = missing)
    point_names: tuple = POINT_NAMES

    def as_points(self):
        """Dict of (T,3) arrays in the layout joint_angles.all_arm_angles expects."""
        p = {n: self.points[:, i] for i, n in enumerate(self.point_names)}
        p["mid_hip"] = (p["l_hip"] + p["r_hip"]) / 2.0
        return p


class ChildRecorder:
    """Background capture thread: start() ... stop() -> ChildRecording."""

    def __init__(self, camera, estimator, preview=False, video_writer=None):
        self.camera, self.estimator = camera, estimator
        self.preview, self.video_writer = preview, video_writer
        self._thread, self._stop = None, threading.Event()
        self._t, self._pts = [], []
        self.last_frame = None

    def start(self):
        self._t, self._pts = [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                color, depth = self.camera.get_frames()
            except RuntimeError as exc:          # frame timeout: keep recording
                print(f"[Camera] {exc}")
                continue
            t = time.monotonic()
            if color is None:
                continue
            pts, results = self.estimator.process(color, depth, self.camera)
            self._t.append(t)
            self._pts.append(pts)
            if self.preview or self.video_writer is not None:
                frame = self.estimator.draw(color, results)
                self.last_frame = frame
                if self.video_writer is not None:
                    self.video_writer.write(frame)

    def stop(self) -> ChildRecording:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if not self._t:
            return ChildRecording(np.zeros(0), np.zeros((0, len(POINT_NAMES), 3)))
        return ChildRecording(np.asarray(self._t), np.stack(self._pts))
