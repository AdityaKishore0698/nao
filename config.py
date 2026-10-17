# config.py — adapted from NAO_Project/config.py
# Machine-specific settings live in config.json (git-ignored); copy
# config.example.json to config.json and edit it.
import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

_DEFAULTS = {
    "robot": "qibullet",          # qibullet (simulated NAO) | kinematic | nao (physical)
    "sim_posture": "Stand",
    "sim_speech": True,           # speak instructions with espeak if installed
    "nao_ip": "172.29.1.216",
    "nao_port": 9559,
    "nao_posture": "Sit",
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "camera_serial": None,
    "data_dir": "data",
    "mongo_uri": None,
}


def load() -> dict:
    cfg = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    # environment overrides (NAO_Project's app.py passes NAO_IP this way)
    if os.environ.get("NAO_IP"):
        cfg["nao_ip"] = os.environ["NAO_IP"]
    return cfg


def save(data: dict):
    current = load()
    current.update(data)
    CONFIG_PATH.write_text(json.dumps(current, indent=2))


def get(key: str):
    return load().get(key)
