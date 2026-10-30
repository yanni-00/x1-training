"""Recompute modified gait FK with existing MuJoCo; native Lab FK still required."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--csv', type=Path, required=True)
p.add_argument('--gait-manifest', type=Path, required=True)
p.add_argument('--source-manifest', type=Path, required=True)
p.add_argument('--runtime-checkpoint', type=Path, required=True)
p.add_argument('--mjcf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
gait = json.loads(a.gait_manifest.read_text())
assert hashlib.sha256(a.csv.read_bytes()).hexdigest() == gait['output_sha256']
sdk_names = json.loads(a.source_manifest.read_text())['joint_names']
runtime = torch.load(a.runtime_checkpoint, map_location='cpu', weights_only=True)['infos']['oma_runtime']
model = mujoco.MjModel.from_xml_path(str(a.mjcf))
data = mujoco.MjData(model)
qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in sdk_names])
body_ids = [model.body(n).id for n in runtime['body_names']]
feet = [runtime['body_names'].index(n) for n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
rows = np.loadtxt(a.csv, delimiter=',')
positions, quaternions = [], []
for row in rows:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = row[:3]
    data.qpos[3:7] = row[[6, 3, 4, 5]]
    data.qpos[qids] = row[7:]
    mujoco.mj_forward(model, data)
    positions.append(data.xpos[body_ids].copy())
    quaternions.append(data.xquat[body_ids].copy())
positions, quaternions = np.array(positions), np.array(quaternions)
count = gait['cycle_intervals']
assert len(rows) == count+1
stride = np.array([gait['root_forward_stride_m'], 0.0, 0.0])
closure = positions[-1]-positions[0]-stride
orientation_error = 1-np.abs(np.sum(quaternions[-1]*quaternions[0], axis=-1))
unique = positions[:-1]
previous, following = np.roll(unique, 1, axis=0), np.roll(unique, -1, axis=0)
previous[0] -= stride
following[-1] += stride
velocity = (following-previous)/(2/gait['output_fps'])
height = unique[:, feet, 2].max(axis=1)
foot_speed = np.linalg.norm(velocity[:, feet], axis=-1).max(axis=1)
height_limit = float(np.quantile(height, 0.25))
speed_limit = float(np.quantile(foot_speed, 0.5))
mask = (height <= height_limit) & (foot_speed <= speed_limit)
supports = []
if mask.any() and not mask.all():
    start = (np.flatnonzero(~mask)[0]+1) % count
    order = (np.arange(count)+start) % count
    changes = np.diff(np.r_[False, mask[order], False].astype(int))
    for lo, hi in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        indices = order[lo:hi]
        supports.append(int(indices[np.argmin(foot_speed[indices])]))
supports.sort()
report = {'scope': 'Modified-CSV G1 MuJoCo kinematic FK only; native Lab FK and dynamics not accepted',
          'csv_sha256': gait['output_sha256'], 'mjcf_sha256': hashlib.sha256(a.mjcf.read_bytes()).hexdigest(),
          'runtime_order_source_checkpoint_sha256': hashlib.sha256(a.runtime_checkpoint.read_bytes()).hexdigest(),
          'maximum_body_translation_closure_error_m': float(np.abs(closure).max()),
          'maximum_body_quaternion_dot_closure_error': float(np.abs(orientation_error).max()),
          'support_frames': supports, 'support_candidate_count': len(supports),
          'support_height_threshold_m': height_limit, 'support_speed_threshold_mps': speed_limit,
          'support_source': 'Existing quantile0.25/0.5 rule, circular intervals merged across the gait seam',
          'all_joint_limits_pass': gait['all_joint_limits_pass'],
          'dynamic_feasibility_verified': False, 'native_fk_recomputed': False, 'behavior_acceptance': False}
report['kinematic_readiness'] = (report['maximum_body_translation_closure_error_m'] <= 3e-5 and
                                  report['maximum_body_quaternion_dot_closure_error'] <= 3e-5 and
                                  len(supports) >= 2 and gait['all_joint_limits_pass'])
report['closure_tolerance_source'] = 'Existing offline observation absolute tolerance3e-5; numerical interface check only'
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
if not report['kinematic_readiness']:
    raise SystemExit('Kinematic gait readiness failed; do not launch native training')
