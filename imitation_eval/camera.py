"""
camera.py — Intel RealSense D4xx RGB-D capture (replaces the paper's markers)
============================================================================
Adapted from NAO/nao_imitation/by kimi/realsense_camera.py: colour + depth
streams with depth aligned to colour, and pixel -> 3D deprojection.

Coordinates returned by deproject() are RealSense camera coordinates in metres:
x right, y down, z forward (away from the camera). Right-handed.
"""

import numpy as np

CAMERA_UP = np.array([0.0, -1.0, 0.0])   # used when the hips are not visible


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial=None):
        import pyrealsense2 as rs
        self._rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.intrinsics = None

    def start(self):
        profile = self.pipeline.start(self.config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics = (profile.get_stream(self._rs.stream.color)
                           .as_video_stream_profile().get_intrinsics())
        print(f"[RealSense] {self.width}x{self.height} @ {self.fps} fps, "
              f"depth scale {self.depth_scale:.6f} m/unit")
        return self

    def get_frames(self, timeout_ms=1000):
        """Returns (color_bgr (H,W,3) uint8, depth (H,W) uint16) or (None, None)."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self.align.process(frames)
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            return None, None
        return np.asanyarray(color.get_data()).copy(), np.asanyarray(depth.get_data()).copy()

    def deproject(self, px, py, depth_m):
        return np.array(self._rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [float(px), float(py)], float(depth_m)))

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:                       # noqa: BLE001
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def list_devices():
    import pyrealsense2 as rs
    return [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            for d in rs.context().query_devices()]
