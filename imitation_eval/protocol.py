"""
protocol.py — one assessment session, following Fears et al. (2023) 2.3
=======================================================================
For each gesture: NAO gives the mirror instruction, then performs 5 continuous
repetitions while the RealSense records the child imitating.
    unimanual (bump, give, wave): 5 reps with NAO's left arm, then 5 with its right
    bimanual (celebrate, hug, what): 5 reps with both arms

No performance feedback is given during the assessment (it would change the
behaviour being measured); NAO only gives neutral encouragement.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from . import storage
from .gestures import GESTURE_ORDER, GESTURES, build_trajectory
from .robot import LED_BLUE, LED_GREEN, LED_WHITE
from .scoring import ScoringConfig, score_trial

INSTRUCTION = ("Now, I am going to move. Do exactly what I do, like you are looking in a "
               "mirror. If I move this side, you move your other side.")


@dataclass
class SessionConfig:
    participant: str
    group: str = "unknown"            # autistic / neurotypical / unknown
    age: Optional[float] = None
    gestures: List[str] = field(default_factory=lambda: list(GESTURE_ORDER))
    reps: int = 5
    speed: float = 1.0
    rest_between_reps: float = 1.0
    lead_in_s: float = 1.0            # camera starts this long before NAO moves
    tail_s: float = 2.0               # and keeps recording this long after
    pause_between_trials_s: float = 3.0
    data_dir: str = "data"
    mongo_uri: Optional[str] = None
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
                            + "_" + uuid.uuid4().hex[:6])


def trial_plan(gestures, unimanual_arms=("L", "R")):
    for g in gestures:
        if GESTURES[g].bimanual:
            yield g, "both"
        else:
            for arm in unimanual_arms:
                yield g, arm


def run_session(cfg: SessionConfig, robot, start_child, stop_child,
                scoring: ScoringConfig = None, source="live", robot_type="unknown",
                on_trial: Callable = None):
    """
    robot: NaoRobot or SimulatedRobot.
    start_child() starts recording the child; stop_child(robot_rec) returns a
    ChildRecording (the robot recording is passed in so a simulator can use it).
    """
    scoring = scoring or ScoringConfig()
    d = storage.session_dir(cfg.data_dir, cfg.participant, cfg.session_id)
    storage.save_session_meta(d, {**asdict(cfg), "scoring": asdict(scoring),
                                  "source": source, "robot": robot_type, "started": datetime.now().isoformat()})
    all_rows = []
    robot.say(INSTRUCTION)
    for gesture, arms in trial_plan(cfg.gestures):
        traj = build_trajectory(GESTURES[gesture], arms, reps=cfg.reps, speed=cfg.speed,
                                rest_between_reps=cfg.rest_between_reps)
        robot.leds_to(LED_BLUE)
        robot.say("Watch me, and copy me!")
        start_child()
        time.sleep(cfg.lead_in_s if source == "live" else 0.0)
        robot_rec = robot.perform(traj)
        time.sleep(cfg.tail_s if source == "live" else 0.0)
        child_rec = stop_child(robot_rec)
        robot.leds_to(LED_GREEN)

        storage.save_trial(d, gesture, arms, robot_rec, child_rec,
                           {"rep_starts": traj.rep_starts, "reps": cfg.reps, "speed": cfg.speed})
        robot_arms = ["L", "R"] if arms == "both" else [arms]
        rows = score_trial(robot_rec, child_rec, robot_arms, scoring, tail_s=cfg.tail_s)
        for r in rows:
            r.update(participant=cfg.participant, group=cfg.group, age=cfg.age,
                     session_id=cfg.session_id, gesture=gesture, trial_arms=arms,
                     source=source, robot=robot_type)
        storage.append_results(d / "results.csv", rows)
        storage.append_results(f"{cfg.data_dir}/results.csv", rows)
        storage.save_to_mongo(rows, cfg.mongo_uri)
        all_rows.extend(rows)
        if on_trial:
            on_trial(gesture, arms, rows)

        robot.say("Great job!")
        robot.leds_to(LED_WHITE)
        robot.go_ready()
        time.sleep(cfg.pause_between_trials_s if source == "live" else 0.0)
    robot.say("We are all done. Thank you for playing with me!")
    return d, all_rows
