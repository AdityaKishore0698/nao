import numpy as np
import pytest

from imitation_eval.gestures import GESTURES, GESTURE_ORDER, build_trajectory
from imitation_eval.joint_angles import all_arm_angles
from imitation_eval.nao_kinematics import JOINT_LIMITS, arm_fk, robot_points
from imitation_eval.robot import SimulatedRobot
from imitation_eval.scoring import robot_angle_series


def test_fk_arm_forward_and_down():
    sh, el, wr = arm_fk("L", 0.0, 0.0, 0.0, 0.0)          # straight forward
    d = (wr - sh)[0]
    assert d[0] > 0.15 and abs(d[2]) < 1e-9
    sh, el, wr = arm_fk("L", np.pi / 2, 0.0, 0.0, 0.0)    # hanging
    assert (wr - sh)[0][2] < -0.15


def test_fk_shoulder_roll_moves_arm_outward_on_both_sides():
    _, el_l, _ = arm_fk("L", 0.0, 1.0, 0.0, 0.0)
    _, el_r, _ = arm_fk("R", 0.0, -1.0, 0.0, 0.0)
    assert el_l[0][1] > 0.098 and el_r[0][1] < -0.098


def test_robot_angles_for_simple_postures():
    base = {f"{s}{j}": 0.0 for s in "LR" for j in ("ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll")}
    # arms straight forward -> flexion 90, abduction ~0
    a = all_arm_angles(robot_points({k: np.array([v]) for k, v in base.items()}))
    for s in "LR":
        assert a[s]["shoulder_flex_ext"][0] == pytest.approx(90, abs=1)
        assert a[s]["elbow_flex_ext"][0] == pytest.approx(0, abs=1)
    # arms out to the side
    side = dict(base, LShoulderRoll=1.3, RShoulderRoll=-1.3, LShoulderPitch=np.pi / 2,
                RShoulderPitch=np.pi / 2)
    a = all_arm_angles(robot_points({k: np.array([v]) for k, v in side.items()}))
    for s in "LR":
        assert a[s]["shoulder_abd_add"][0] == pytest.approx(np.degrees(1.3), abs=5)


@pytest.mark.parametrize("name", GESTURE_ORDER)
def test_trajectories_within_limits_and_increasing(name):
    g = GESTURES[name]
    for arms in (["both"] if g.bimanual else ["L", "R"]):
        tr = build_trajectory(g, arms, reps=2)
        for n, ang, tm in zip(tr.names, tr.angles, tr.times):
            lo, hi = JOINT_LIMITS[n]
            assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in ang), n
            assert all(b > a for a, b in zip(tm, tm[1:])), n


def _ranges(name, arm="L"):
    g = GESTURES[name]
    arms = "both" if g.bimanual else arm
    rec = SimulatedRobot(verbose=False).perform(build_trajectory(g, arms, reps=1))
    _, ang = robot_angle_series(rec)
    return {k: np.nanmax(v) - np.nanmin(v) for k, v in ang[arm].items()}, ang[arm]


def test_gestures_move_the_joints_the_paper_describes():
    r, a = _ranges("bump")
    assert r["shoulder_flex_ext"] > 30 and r["elbow_flex_ext"] > 60
    r, a = _ranges("give")
    assert r["shoulder_flex_ext"] > 30 and r["elbow_flex_ext"] > 60
    r, a = _ranges("wave")
    assert np.nanmax(a["shoulder_abd_add"]) > 30           # arm raised to the side
    r, a = _ranges("celebrate")
    assert np.nanmax(a["shoulder_flex_ext"]) > 140 or np.nanmax(a["shoulder_abd_add"]) > 140
    r, a = _ranges("hug")
    assert r["shoulder_abd_add"] > 40
    r, a = _ranges("what")
    assert r["elbow_flex_ext"] > 30


def test_wave_elbow_oscillation_is_about_40_degrees():
    rec = SimulatedRobot(verbose=False).perform(build_trajectory(GESTURES["wave"], "L", reps=1))
    _, ang = robot_angle_series(rec)
    elbow = ang["L"]["elbow_flex_ext"]
    t = rec.t - rec.t[0]
    osc = elbow[(t > 1.6) & (t < 5.9)]
    assert np.nanmax(osc) - np.nanmin(osc) == pytest.approx(40, abs=3)
