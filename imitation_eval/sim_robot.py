"""
sim_robot.py — physics-simulated NAO demonstrator (qiBullet / PyBullet)
======================================================================
Replaces the physical NAO for the experiments. The simulated NAO is shown on
a screen facing the child; it performs the same gesture keyframes as the real
robot would, and its simulated joint encoders are recorded exactly like the
real robot's sensors (RobotRecording), so scoring is unchanged.

qiBullet: https://github.com/softbankrobotics-research/qibullet
(first run downloads the NAO meshes after accepting SoftBank's licence —
see scripts/install_qibullet_meshes.py).

Display: the PyBullet window shows NAO from the front, filling the view, with
the debug panels hidden. Put it full-screen on the monitor the child faces.
Facing the child, NAO's left arm appears on the child's right — the same
mirror relation as with the physical robot.
"""

import shutil
import subprocess
import time

import numpy as np

from .gestures import READY, Trajectory, build_trajectory, sample_trajectory, side_pose
from .nao_kinematics import SENSED_JOINTS
from .robot import RobotRecording


class QiBulletNao:
    def __init__(self, gui=True, camera_distance=0.75, camera_height=0.33, speech=True):
        from qibullet import NaoVirtual, SimulationManager
        import pybullet as p
        self._p = p
        self.manager = SimulationManager()
        self.client = self.manager.launchSimulation(gui=gui, auto_step=True)
        self.nao = self.manager.spawnNao(self.client, spawn_ground_plane=True)
        assert isinstance(self.nao, NaoVirtual)
        self.gui = gui
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            # NAO faces +x; look at it from the front
            p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=90,
                                         cameraPitch=-5,
                                         cameraTargetPosition=[0, 0, camera_height],
                                         physicsClientId=self.client)
        self.speech = (shutil.which("espeak-ng") or shutil.which("espeak")) if speech else None
        self._current = {}

    # ── setup / teardown ──────────────────────────────────────────────────
    def prepare(self, posture="Stand"):
        if posture not in ("Stand", "StandInit", "StandZero", "Crouch"):
            posture = "Stand"           # qiBullet has no "Sit" posture
        self.nao.goToPosture(posture, 0.6)
        time.sleep(1.0)
        self.go_ready()

    def go_ready(self, duration=1.5):
        names, targets = [], []
        for side in "LR":
            for n, v in side_pose(READY, side).items():
                names.append(n)
                targets.append(v)
        traj = Trajectory(names, [[v] for v in targets], [[duration]] * len(names))
        self._run(traj, record=False)

    def finish(self, posture="Stand", rest=False):
        try:
            self.manager.stopSimulation(self.client)
        except Exception:                       # noqa: BLE001
            pass

    # ── interaction ───────────────────────────────────────────────────────
    def say(self, text):
        print(f"[SimNAO says] {text}")
        if self.speech:
            subprocess.run([self.speech, "-s", "130", text], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def leds_to(self, colour, duration=0.3):
        pass

    def read_angles(self):
        return self.nao.getAnglesPosition(list(SENSED_JOINTS))

    # ── motion ────────────────────────────────────────────────────────────
    def _run(self, traj: Trajectory, record=True, sample_hz=50.0, settle_s=0.3):
        """Stream eased keyframe targets to the position controllers in real time
        and (optionally) record the simulated joint encoders."""
        begin = dict(zip(traj.names, self.nao.getAnglesPosition(list(traj.names))))
        ts, rows = [], []
        period = 1.0 / sample_hz
        t0 = time.monotonic()
        while True:
            now = time.monotonic()
            t = now - t0
            self.nao.setAngles(list(traj.names), sample_trajectory(traj, t, begin), 1.0)
            if record:
                rows.append(self.read_angles())
                ts.append(now)
            if t >= traj.duration + settle_s:
                break
            time.sleep(max(0.0, period - (time.monotonic() - now)))
        if record:
            return RobotRecording(np.asarray(ts), np.asarray(rows, dtype=float))
        return None

    def perform(self, traj: Trajectory, sample_hz=50.0) -> RobotRecording:
        return self._run(traj, record=True, sample_hz=sample_hz)


def demo(gesture="wave", arms="L"):
    """Quick visual check: python -m imitation_eval.sim_robot"""
    from .gestures import GESTURES
    robot = QiBulletNao()
    robot.prepare()
    robot.perform(build_trajectory(GESTURES[gesture], arms, reps=2))
    robot.finish()


if __name__ == "__main__":
    demo()
