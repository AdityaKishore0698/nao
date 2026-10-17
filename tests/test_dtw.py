import numpy as np
import pytest

from imitation_eval.dtw import dtw, dtw_znorm, znorm


def test_identical_is_zero():
    x = np.sin(np.linspace(0, 6, 80))
    assert dtw(x, x)[0] == pytest.approx(0)


def test_known_small_case():
    # optimal alignment of [0,1,2] with [0,0,1,2] costs 0
    assert dtw([0, 1, 2], [0, 0, 1, 2])[0] == pytest.approx(0)
    assert dtw([0, 1, 2], [0, 2])[0] == pytest.approx(1)


def test_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=40), rng.normal(size=55)
    assert dtw(a, b)[0] == pytest.approx(dtw(b, a)[0])


def test_time_warp_costs_less_than_different_shape():
    t = np.linspace(0, 1, 120)
    robot = np.sin(2 * np.pi * 2 * t)
    slow = np.sin(2 * np.pi * 2 * np.clip((t - 0.1) / 1.1, 0, 1))      # delayed + slower
    other = np.cos(2 * np.pi * 5 * t)
    assert dtw_znorm(robot, slow)["dtw_cost"] < 0.3 * dtw_znorm(robot, other)["dtw_cost"]


def test_znorm_scale_invariance():
    x = np.sin(np.linspace(0, 6, 50))
    assert dtw(znorm(x), znorm(10 * x + 3))[0] == pytest.approx(0, abs=1e-9)
    assert np.all(znorm(np.ones(5)) == 0)


def test_window_never_blocks_path():
    cost, _ = dtw(np.zeros(10), np.zeros(30), window=2)
    assert np.isfinite(cost)
