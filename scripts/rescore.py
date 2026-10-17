"""
rescore.py — recompute DTW results from saved raw trials (.npz)
===============================================================
Use after changing the scoring settings; raw data are never modified.

    python scripts/rescore.py --data-dir data --out data/results_rescored.csv
    python scripts/rescore.py --fs 60 --cutoff 8 --no-mirror
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitation_eval import storage  # noqa: E402
from imitation_eval.scoring import ScoringConfig, score_trial  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--min-valid", type=float, default=0.6)
    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    cfg = ScoringConfig(fs=args.fs, cutoff_hz=args.cutoff, min_valid_fraction=args.min_valid,
                        mirror=not args.no_mirror, dtw_window_s=args.window_s)
    out = Path(args.out or Path(args.data_dir) / "results_rescored.csv")
    if out.exists():
        out.unlink()
    n = 0
    for meta_path in sorted(Path(args.data_dir).glob("sessions/*/*/session.json")):
        meta = json.loads(meta_path.read_text())
        tail = meta.get("tail_s", 2.0)
        for trial in sorted(meta_path.parent.glob("*.npz")):
            robot, child, tmeta = storage.load_trial(trial)
            arms = tmeta["arms"]
            rows = score_trial(robot, child, ["L", "R"] if arms == "both" else [arms], cfg,
                               tail_s=tail)
            for r in rows:
                r.update(participant=meta["participant"], group=meta["group"], age=meta["age"],
                         session_id=meta["session_id"], gesture=tmeta["gesture"],
                         trial_arms=arms, source=meta.get("source", ""),
                         robot=meta.get("robot", ""))
            storage.append_results(out, rows)
            n += 1
    print(f"Rescored {n} trials -> {out}")


if __name__ == "__main__":
    main()
