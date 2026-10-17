# NAO Imitation Evaluation

An implementation of the imitation assessment in

> Fears, N. E., Sherrod, G. M., Blankenship, D., Patterson, R. M., Hynan, L. S., Wijayasinghe, I., Popa, D. O., Bugnariu, N. L., & Miller, H. L. (2023).
> *Motor differences in autism during a human-robot imitative gesturing task.* Clinical Biomechanics, 106, 105987.
> https://doi.org/10.1016/j.clinbiomech.2023.105987

The original study used the Zeno R30 robot and a 16-camera marker-based motion capture system. This project uses a **simulated NAO** (qiBullet physics simulation, shown on a screen facing the child) as the demonstrator and an **Intel RealSense depth camera** (markerless, MediaPipe Pose and aligned depth) to capture the child. The same code can also drive a physical NAO V4 (`--robot nao`).

## What it does

1. NAO demonstrates the paper's six gestures (Appendix B), each with 5 continuous repetitions:
   - **Unimanual** (bump, give, wave): left arm, then right arm.
   - **Bimanual** (celebrate, hug, what): both arms.
2. The child is told to imitate "like a mirror". While NAO moves, its (simulated) joint **encoder** angles are recorded, and the RealSense records the child.
3. The same code computes four anatomical angles per arm for both NAO and the child:
   - shoulder flexion/extension
   - shoulder abduction/adduction
   - shoulder internal/external rotation
   - elbow flexion/extension

   NAO's points come from forward kinematics and the child's from 3D landmarks.
4. Each angle curve is z-normalised and compared with exact **dynamic time warping**: the robot's arm against the child's mirrored arm. Lower cost means more accurate imitation.
5. `scripts/analyze.py` fits the paper's linear mixed-effects model on log DTW cost, with a random intercept per participant.

### Differences from the paper

| Paper | Here | Why |
|---|---|---|
| Zeno R30 robot, physically present | Simulated NAO (qiBullet) on a screen | Chosen for the experiments. Gestures are NAO keyframes in `imitation_eval/gestures.py`. The child sees a 2D screen rather than a 3D robot, which may affect imitation |
| Reflective markers on robot | Simulated joint encoders + forward kinematics | Exact, and needs no markers on the robot |
| Joint angles from Visual3D (Simon 2004) | Cardan flexion, abduction, rotation of the humerus relative to the trunk | Same four angles, computed identically for robot and child (`joint_angles.py`) |
| 16-camera marker mocap, 120 Hz | RealSense RGB-D + MediaPipe, ~30 Hz | Markerless; DTW is insensitive to the sample-rate difference after resampling |
| Work analysis (force plates) | Not implemented | Needs force plates. The paper's work findings were also weak |
| No pause between reps | 1 s hold between reps (configurable) | Addresses the paper's segmentation limitation (section 4.1) |

## Layout

```
imitation_eval/
  gestures.py        six gestures as NAO keyframes (left arm; right is mirrored)
  nao_kinematics.py  NAO V4 arm forward kinematics and joint limits
  joint_angles.py    the paper's four angles from 3D shoulder/elbow/wrist/hip points
  dtw.py             z-normalisation and exact DTW
  signal_utils.py    resampling, gap filling, 6 Hz low-pass filter
  sim_robot.py       QiBulletNao: physics-simulated NAO demonstrator (default robot)
  robot.py           NaoRobot (physical, qi) and SimulatedRobot (kinematic, no window)
  camera.py          RealSense capture (adapted from NAO/nao_imitation/by kimi)
  pose3d.py          MediaPipe and depth to 3D arm keypoints; background ChildRecorder
  scoring.py         per trial, per arm, per angle DTW, with mirroring
  protocol.py        session flow following the paper's procedure
  storage.py         raw .npz per trial and long-format results.csv
  simulate.py        synthetic mirror-imitating child for dry runs and tests
scripts/
  run_session.py     run an assessment (live or simulated)
  demo_gestures.py   perform gestures and print the robot's angle ranges (for tuning)
  check_camera.py    live preview with the four angles (for camera placement)
  rescore.py         recompute results from raw .npz with different settings
  analyze.py         mixed-effects group comparison
tests/               pytest suite (no robot or camera needed)
```

## Setup (on `cir`)

```bash
conda activate ~/anaconda3/envs/nao_eval      # clone of NAO_Project's "NAO" env + pandas/statsmodels/pytest/pybullet/qibullet
cp config.example.json config.json            # robot = qibullet (default); camera_serial if more than one camera
python -m pytest -q                           # runs without the robot or camera
```

The env lives on the main drive (`/`, with `/home/anishma`). Give conda an explicit `-p` path for new envs, because `~/.condarc` points the default envs folder at a different, full drive. pip on `cir` needs the campus proxy: `export https_proxy=http://172.31.2.3:8080`.

qiBullet needs the NAO meshes installed once. This asks you to accept SoftBank Robotics' meshes licence:

```bash
python -c "from qibullet import tools; tools._install_resources()"
```

## Running

```bash
# 1. Dry run, no hardware
python scripts/run_session.py --participant TEST --sim-robot --sim-child

# 2. Watch the simulated NAO perform the gestures (compare with the paper's Appendix C)
python scripts/demo_gestures.py --gestures wave --reps 1

# 3. Place the camera: the child's shoulders, elbows, wrists and hips must be tracked
python scripts/check_camera.py

# 4. Real session (use pseudonymous IDs)
python scripts/run_session.py --participant P001 --group autistic --age 9

# 5. Analysis once both groups have data
python scripts/analyze.py --results data/results.csv
```

Set-up: the monitor showing the simulated NAO goes in front of the child, with the PyBullet window full-screen. The RealSense goes about 2–3 m in front of the child at chest height (for example just above or beside the monitor), with the whole upper body and hips in view.

Note on angles the robot barely moves (for example rotation during bump): z-normalising a nearly flat robot curve turns it into noise, so those DTW costs mean little. `analyze.py --min-robot-range 15` excludes them. The paper reported all angles; the default does the same.

## Data and privacy

`data/` is git-ignored. It holds per-trial raw recordings (`.npz`: robot angles and the child's 3D keypoints, no images), `results.csv`, and, only with `--save-video`, annotated video. Use participant codes, not names, and keep the code key separate.

## Status and to-do

- [ ] Check the gestures visually on the simulated NAO (`demo_gestures.py`). Palm orientation (`PALM_*` in `gestures.py`) needs checking.
- [ ] Validate the camera angles against a goniometer or known poses (`check_camera.py`).
- [ ] Optional: integrate with NAO_Project's `app.py` dashboard.
