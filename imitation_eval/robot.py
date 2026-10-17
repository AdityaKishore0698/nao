"""
robot.py — NAO demonstrator: performs a gesture and records its own arm angles
=============================================================================
While NAO demonstrates, its arm joint SENSOR angles are sampled (ALMotion
getAngles(useSensors=True)). These replace the paper's markers on Zeno: they
are passed through nao_kinematics -> joint_angles, so the robot's reference
curve is computed with the same code as the child's.

Connection helpers are adapted from NAO_Project/scripts/dynamic_imitation.py.

SimulatedRobot has the same interface and interpolates the keyframes, so the
whole pipeline can run without a robot (tests, dry runs).
"""

import time
from dataclasses import dataclass

import numpy as np

from .gestures import READY, Trajectory, sample_trajectory, side_pose
from .nao_kinematics import SENSED_JOINTS

LED_BLUE, LED_GREEN, LED_WHITE = 0x0000FF, 0x00FF00, 0xFFFFFF


@dataclass
class RobotRecording:
    t: np.ndarray            # (T,) host time.monotonic() seconds
    angles: np.ndarray       # (T, 8) sensed angles, columns = SENSED_JOINTS
    joint_names: tuple = SENSED_JOINTS

    def as_dict(self):
        return {n: self.angles[:, i] for i, n in enumerate(self.joint_names)}


class NaoRobot:
    def __init__(self, ip, port=9559, retries=3, delay=3.0):
        import qi
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                self.session = qi.Session()
                self.session.connect(f"tcp://{ip}:{port}")
                break
            except Exception as exc:          # noqa: BLE001 — qi raises RuntimeError
                last_exc = exc
                print(f"[NAO] Connection attempt {attempt}/{retries} failed: {exc}")
                time.sleep(delay)
        else:
            raise RuntimeError(f"Could not connect to NAO at {ip}:{port} — {last_exc}")
        print(f"[NAO] Connected to {ip}:{port}")
        self.motion = self.session.service("ALMotion")
        self.posture = self.session.service("ALRobotPosture")
        self.tts = self.session.service("ALTextToSpeech")
        self.leds = self.session.service("ALLeds")

    # ── setup / teardown ──────────────────────────────────────────────────
    def prepare(self, posture="Sit"):
        self.motion.wakeUp()
        self.posture.goToPosture(posture, 0.5)
        try:
            self.motion.setCollisionProtectionEnabled("Arms", True)
        except Exception as exc:              # noqa: BLE001
            print(f"[NAO] Could not enable arm collision protection: {exc}")
        self.go_ready()

    def go_ready(self, duration=1.5):
        names, angles = [], []
        for side in "LR":
            for n, v in side_pose(READY, side).items():
                names.append(n)
                angles.append(v)
        self.motion.angleInterpolation(names, angles, [duration] * len(names), True)

    def finish(self, posture="Sit", rest=False):
        try:
            self.posture.goToPosture(posture, 0.5)
            if rest:
                self.motion.rest()
        except Exception as exc:              # noqa: BLE001
            print(f"[NAO] finish: {exc}")

    # ── interaction ───────────────────────────────────────────────────────
    def say(self, text):
        try:
            self.tts.say(f"\\rspd=80\\ {text}")
        except Exception as exc:              # noqa: BLE001
            print(f"[TTS] {exc}")

    def leds_to(self, colour, duration=0.3):
        try:
            self.leds.fadeRGB("FaceLeds", colour, duration)
        except Exception:                     # noqa: BLE001
            pass

    def read_angles(self):
        return self.motion.getAngles(list(SENSED_JOINTS), True)

    def perform(self, traj: Trajectory, sample_hz=50.0) -> RobotRecording:
        """Run the trajectory (non-blocking) while sampling sensor angles."""
        ts, rows = [], []
        period = 1.0 / sample_hz
        fut = self.motion.angleInterpolation(traj.names, traj.angles, traj.times, True,
                                             _async=True)
        while True:
            t0 = time.monotonic()
            rows.append(self.read_angles())
            ts.append(t0)
            if fut.isFinished():
                break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        if fut.hasError():
            raise RuntimeError(f"angleInterpolation failed: {fut.error()}")
        return RobotRecording(np.asarray(ts), np.asarray(rows, dtype=float))


class SimulatedRobot:
    """Pure keyframe interpolation (no physics, no display) — for tests and fast
    dry runs. realtime=False returns instantly. For experiments use
    sim_robot.QiBulletNao, the physics-simulated NAO."""

    def __init__(self, realtime=False, verbose=True):
        self.realtime = realtime
        self.verbose = verbose
        self._current = {**side_pose(READY, "L"), **side_pose(READY, "R")}

    def prepare(self, posture="Sit"):
        pass

    def go_ready(self, duration=1.5):
        pass

    def finish(self, posture="Sit", rest=False):
        pass

    def say(self, text):
        if self.verbose:
            print(f"[SimNAO says] {text}")

    def leds_to(self, colour, duration=0.3):
        pass

    def perform(self, traj: Trajectory, sample_hz=50.0) -> RobotRecording:
        start = time.monotonic()
        t_rel = np.arange(0.0, traj.duration + 1e-9, 1.0 / sample_hz)
        idx = [traj.names.index(n) if n in traj.names else None for n in SENSED_JOINTS]
        begin = {n: self._current.get(n, 0.0) for n in traj.names}
        rows = []
        for t in t_rel:
            targets = sample_trajectory(traj, t, begin)
            rows.append([targets[i] if i is not None else self._current[n]
                         for n, i in zip(SENSED_JOINTS, idx)])
        angles = np.asarray(rows)
        self._current.update(zip(traj.names, sample_trajectory(traj, traj.duration, begin)))
        if self.realtime:
            time.sleep(traj.duration)
        return RobotRecording(start + t_rel, angles)
