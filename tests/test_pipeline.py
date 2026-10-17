"""End-to-end on synthetic data: simulated NAO -> simulated mirror-imitating child -> DTW."""
import numpy as np
import pytest

from imitation_eval import storage
from imitation_eval.gestures import GESTURES, build_trajectory
from imitation_eval.protocol import SessionConfig, run_session
from imitation_eval.robot import SimulatedRobot
from imitation_eval.scoring import ScoringConfig, score_trial
from imitation_eval.simulate import simulate_child


def _rec(name, arms):
    return SimulatedRobot(verbose=False).perform(build_trajectory(GESTURES[name], arms, reps=2))


def _total(rows):
    return np.nansum([r["dtw_cost"] for r in rows])


@pytest.mark.parametrize("name,arms", [("bump", "L"), ("wave", "R"), ("celebrate", "both")])
def test_good_imitation_beats_poor_imitation(name, arms):
    rec = _rec(name, arms)
    robot_arms = ["L", "R"] if arms == "both" else [arms]
    good = score_trial(rec, simulate_child(rec, seed=1), robot_arms)
    poor = score_trial(rec, simulate_child(rec, seed=1, poor=True), robot_arms)
    assert _total(good) < 0.6 * _total(poor)


def test_mirror_mapping_matters():
    rec = _rec("bump", "L")
    child = simulate_child(rec, seed=2)
    mirrored = score_trial(rec, child, ["L"], ScoringConfig(mirror=True))
    same_side = score_trial(rec, child, ["L"], ScoringConfig(mirror=False))
    assert mirrored[0]["child_arm"] == "R"
    assert _total(mirrored) < _total(same_side)


def test_missing_frames_are_tolerated():
    rec = _rec("give", "R")
    rows = score_trial(rec, simulate_child(rec, dropout=0.2, seed=3), ["R"])
    assert any(r["scored"] for r in rows)


def test_session_saves_and_rescoring_matches(tmp_path):
    cfg = SessionConfig(participant="TEST", gestures=["bump", "hug"], reps=1,
                        data_dir=str(tmp_path), lead_in_s=0, tail_s=2.0,
                        pause_between_trials_s=0)
    d, rows = run_session(cfg, SimulatedRobot(verbose=False), lambda: None, simulate_child,
                          source="simulated")
    assert len(rows) == (1 + 1 + 2) * 4                # bump L, bump R, hug L+R; 4 angles each
    trials = sorted(d.glob("*.npz"))
    assert [t.name for t in trials] == ["bump_L.npz", "bump_R.npz", "hug_both.npz"]
    robot, child, meta = storage.load_trial(trials[0])
    again = score_trial(robot, child, [meta["arms"]], tail_s=2.0)
    first = [r for r in rows if r["gesture"] == "bump" and r["trial_arms"] == "L"]
    np.testing.assert_allclose([r["dtw_cost"] for r in again], [r["dtw_cost"] for r in first])
    assert (tmp_path / "results.csv").exists()
