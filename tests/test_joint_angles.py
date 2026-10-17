import numpy as np
import pytest

from imitation_eval.joint_angles import all_arm_angles

# Body frame for these tests: x forward, y left, z up
L_SH, R_SH, HIP = np.array([0, 0.2, 1.4]), np.array([0, -0.2, 1.4]), np.array([0, 0, 0.9])
UPPER, FORE = 0.3, 0.25


def pose(l_upper, l_fore, r_upper=(0, 0, -1), r_fore=(0, 0, -1)):
    def arm(sh, up, fo):
        up, fo = np.array(up, float), np.array(fo, float)
        el = sh + UPPER * up / np.linalg.norm(up)
        return el, el + FORE * fo / np.linalg.norm(fo)
    le, lw = arm(L_SH, l_upper, l_fore)
    re, rw = arm(R_SH, r_upper, r_fore)
    pts = dict(l_shoulder=L_SH, r_shoulder=R_SH, l_elbow=le, r_elbow=re,
               l_wrist=lw, r_wrist=rw, mid_hip=HIP)
    return {k: np.asarray(v, float)[None] for k, v in pts.items()}


def angles(*a, **k):
    out = all_arm_angles(pose(*a, **k))
    return {s: {n: float(v[0]) for n, v in d.items()} for s, d in out.items()}


def test_hanging_arm_is_zero():
    a = angles((0, 0, -1), (1, 0, 0))            # elbow at 90 deg, forearm forward
    assert a["L"]["shoulder_flex_ext"] == pytest.approx(0, abs=1e-6)
    assert a["L"]["shoulder_abd_add"] == pytest.approx(0, abs=1e-6)
    assert a["L"]["shoulder_rotation"] == pytest.approx(0, abs=1e-6)
    assert a["L"]["elbow_flex_ext"] == pytest.approx(90, abs=1e-6)


def test_forward_flexion_and_overhead():
    assert angles((1, 0, 0), (1, 0, 0))["L"]["shoulder_flex_ext"] == pytest.approx(90)
    assert angles((0.01, 0, 1), (0, 0, 1))["L"]["shoulder_flex_ext"] == pytest.approx(180, abs=1)
    assert angles((-0.5, 0, -1), (0, 0, -1))["L"]["shoulder_flex_ext"] < 0     # extension


def test_abduction_is_symmetric_between_arms():
    a = angles((0, 1, 0), (0, 1, 0), r_upper=(0, -1, 0), r_fore=(0, -1, 0))
    assert a["L"]["shoulder_abd_add"] == pytest.approx(90)
    assert a["R"]["shoulder_abd_add"] == pytest.approx(90)
    assert a["L"]["elbow_flex_ext"] == pytest.approx(0, abs=1e-6)


def test_external_rotation_positive_on_both_arms():
    # arm hanging, elbow 90, forearm turned outward = external rotation 90
    a = angles((0, 0, -1), (0, 1, 0), r_fore=(0, -1, 0))
    assert a["L"]["shoulder_rotation"] == pytest.approx(90)
    assert a["R"]["shoulder_rotation"] == pytest.approx(90)
    # forearm across the belly = internal rotation
    assert angles((0, 0, -1), (0, -1, 0))["L"]["shoulder_rotation"] == pytest.approx(-90)


def test_rotation_undefined_with_straight_elbow():
    assert np.isnan(angles((1, 0, 0), (1, 0, 0))["L"]["shoulder_rotation"])


def test_frame_invariance_under_rigid_motion():
    """Angles must not depend on where the camera is (rotation + translation)."""
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    p = pose((0.3, 0.6, -0.5), (0.5, 0.2, 0.4), r_upper=(0.7, -0.2, 0.1), r_fore=(0, -0.3, 1))
    moved = {k: v @ q.T + np.array([1.0, -2.0, 3.0]) for k, v in p.items()}
    a, b = all_arm_angles(p), all_arm_angles(moved)
    for s in "LR":
        for n in a[s]:
            np.testing.assert_allclose(a[s][n], b[s][n], atol=1e-6)
