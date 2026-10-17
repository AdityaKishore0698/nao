"""
storage.py — raw trial data (.npz) + long-format results table (.csv)
=====================================================================
Layout under data_dir (git-ignored — contains participant data):

    sessions/<participant>/<session_id>/<gesture>_<arms>.npz   raw robot + child data
    sessions/<participant>/<session_id>/session.json           participant metadata
    results.csv                                                 one row per arm x angle

Use pseudonymous participant IDs (e.g. P001), never names.
Raw .npz files keep everything needed to re-score later (scripts/rescore.py).
"""

import csv
import json
from pathlib import Path

import numpy as np

from .pose3d import ChildRecording
from .robot import RobotRecording

RESULT_FIELDS = [
    "participant", "group", "age", "session_id", "gesture", "trial_arms",
    "robot_arm", "child_arm", "angle", "dtw_cost", "dtw_cost_per_step", "path_length",
    "scored", "robot_valid_frac", "child_valid_frac", "n_robot", "n_child",
    "robot_range_deg", "child_range_deg", "source",
]


def session_dir(data_dir, participant, session_id):
    d = Path(data_dir) / "sessions" / participant / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session_meta(d, meta):
    (Path(d) / "session.json").write_text(json.dumps(meta, indent=2))


def save_trial(d, gesture, arms, robot_rec, child_rec, extra=None):
    path = Path(d) / f"{gesture}_{arms}.npz"
    np.savez_compressed(
        path,
        robot_t=robot_rec.t, robot_angles=robot_rec.angles,
        robot_joint_names=np.array(robot_rec.joint_names),
        child_t=child_rec.t, child_points=child_rec.points,
        child_point_names=np.array(child_rec.point_names),
        meta=json.dumps({"gesture": gesture, "arms": arms, **(extra or {})}),
    )
    return path


def load_trial(path):
    z = np.load(path, allow_pickle=False)
    robot = RobotRecording(z["robot_t"], z["robot_angles"], tuple(z["robot_joint_names"]))
    child = ChildRecording(z["child_t"], z["child_points"], tuple(z["child_point_names"]))
    return robot, child, json.loads(str(z["meta"]))


def append_results(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def save_to_mongo(rows, uri, db="nao_imitation_eval", collection="dtw_scores"):
    """Optional mirror of the results rows into MongoDB (as NAO_Project did)."""
    if not uri or not rows:
        return
    try:
        import pymongo
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        client[db][collection].insert_many([dict(r) for r in rows])
        client.close()
    except Exception as exc:                  # noqa: BLE001
        print(f"[MongoDB] WARNING: could not save results — {exc}")
