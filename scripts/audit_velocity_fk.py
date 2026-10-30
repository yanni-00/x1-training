"""Pack actual straight windows, then audit foot FK in the existing MuJoCo env.

Support and full-cycle labels are kinematic proxies. No physics is stepped and
no reference asset is accepted automatically from these rankings.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rotation_xyzw(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=1).reshape(-1, 3, 3)


def pack(args):
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {f['filename']: f for f in manifest['files']}
    arrays, cache = [], {}
    for window in audit['candidate_windows']:
        name = window['filename']
        if name not in cache:
            path = args.manifest.parent/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[name]['sha256']
            cache[name] = np.loadtxt(path, delimiter=',')
        arrays.append(cache[name][window['start_frame']:window['end_frame']+1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rows=np.stack(arrays),
                        joint_names=np.array(manifest['joint_names']), fps=np.array([manifest['input_fps']]),
                        windows_json=np.array(json.dumps(audit['candidate_windows'])))
    print(json.dumps({'packed_windows': len(arrays), 'output': str(args.output),
                       'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


def fk(args):
    import mujoco
    with np.load(args.input, allow_pickle=False) as source:
        rows = source['rows']
        names = source['joint_names'].tolist()
        fps = float(source['fps'][0])
        windows = json.loads(str(source['windows_json']))
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    foot_ids = [model.body(n).id for n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
    candidates, reports = [], []
    for window_index, (raw, window) in enumerate(zip(rows, windows)):
        feet = []
        for row in raw:
            mujoco.mj_resetData(model, data)
            data.qpos[:3] = row[:3]
            data.qpos[3:7] = row[[6, 3, 4, 5]]
            data.qpos[qids] = row[7:]
            mujoco.mj_forward(model, data)
            feet.append(data.xpos[foot_ids].copy())
        feet = np.array(feet)
        rotations = rotation_xyzw(raw[:, 3:7])
        local = np.einsum('nji,nkj->nki', rotations, feet-raw[:, None, :3])
        foot_speed = np.linalg.norm(np.gradient(feet, 1/fps, axis=0), axis=-1).max(axis=1)
        heights = feet[:, :, 2].max(axis=1)
        height_limit = float(np.quantile(heights, 0.25))
        speed_limit = float(np.quantile(foot_speed, 0.5))
        mask = (heights <= height_limit) & (foot_speed <= speed_limit)
        change = np.diff(np.r_[False, mask, False].astype(int))
        supports = [int(lo+np.argmin(foot_speed[lo:hi])) for lo, hi in
                    zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))]
        # Both feet must have lifted within a candidate full stride. Quantiles
        # come from this actual window; they are not hardware clearance limits.
        lift = feet[:, :, 2]-feet[:, :, 2].min(axis=1, keepdims=True)
        lift_level = np.quantile(lift, 0.75, axis=0)
        yaw = np.unwrap(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
        joints = raw[:, 7:]
        dq = np.gradient(joints, 1/fps, axis=0)
        accepted = 0
        for i, first in enumerate(supports):
            for last in supports[i+1:]:
                first_lead = np.sign(local[first, 0, 0]-local[first, 1, 0])
                last_lead = np.sign(local[last, 0, 0]-local[last, 1, 0])
                if first_lead != last_lead:
                    continue
                swing = lift[first:last+1].max(axis=0) > lift_level
                if not swing.all():
                    continue
                duration = (last-first)/fps
                delta = rotations[first].T @ (raw[last, :3]-raw[first, :3])
                if delta[0] <= 0:
                    continue
                q_error = joints[last]-joints[first]
                foot_error = local[last]-local[first]
                path = float(np.linalg.norm(np.diff(raw[first:last+1, :2], axis=0), axis=1).sum())
                displacement = float(np.linalg.norm(raw[last, :2]-raw[first, :2]))
                if path == 0 or displacement == 0:
                    continue
                entry = {'filename': window['filename'], 'source_sha256': window['source_sha256'],
                         'window_index': window_index,
                         'start_frame': window['start_frame']+first,
                         'end_frame': window['start_frame']+last, 'input_fps': fps,
                         'duration_s': duration, 'forward_stride_m': float(delta[0]),
                         'lateral_stride_m': float(delta[1]), 'natural_forward_speed_mps': float(delta[0]/duration),
                         'net_yaw_rad': float(yaw[last]-yaw[first]),
                         'straightness_ratio': displacement/path,
                         'joint_seam_max_rad': float(np.max(np.abs(q_error))),
                         'joint_seam_rms_rad': float(np.sqrt(np.mean(q_error*q_error))),
                         'joint_velocity_seam_rms_radps': float(np.sqrt(np.mean((dq[last]-dq[first])**2))),
                         'foot_local_seam_max_m': float(np.linalg.norm(foot_error, axis=1).max()),
                         'left_and_right_swing_observed': True, 'same_fore_aft_foot_order': True,
                         'kinematic_support_only': True, 'dynamic_feasibility_verified': False}
                entry['selection_score'] = (entry['joint_seam_rms_rad'] + abs(entry['net_yaw_rad']) +
                                             1-entry['straightness_ratio'] +
                                             abs(entry['lateral_stride_m'])/displacement +
                                             entry['foot_local_seam_max_m']/displacement)
                candidates.append(entry)
                accepted += 1
        reports.append({'filename': window['filename'], 'start_frame': window['start_frame'],
                         'end_frame': window['end_frame'], 'support_frames_in_window': supports,
                         'height_quantile': 0.25, 'speed_quantile': 0.5,
                         'height_threshold_m': height_limit, 'speed_threshold_mps': speed_limit,
                         'lift_thresholds_from_window_m': lift_level.tolist(), 'candidate_cycles': accepted})
    candidates.sort(key=lambda c: c['selection_score'])
    report = {'scope': 'Exact existing G1 MuJoCo FK, kinematic cycle audit only; no dynamics stepped',
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'mjcf_sha256': hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
              'mujoco_version': mujoco.__version__, 'windows': reports, 'candidate_cycles': candidates,
              'support_rule_source': 'Existing prepare_walkstop_motion.py quantiles0.25/0.5',
              'cycle_rule': 'Same fore-aft foot order at low-height/low-speed support candidates, '
                            'both left/right lift observed above window75th-percentile proxy; forward progress.',
              'behavior_acceptance': False, 'asset_accepted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'candidate_cycles': len(candidates),
                      'top_candidates': candidates[:3]}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pack', 'fk'], required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--mjcf', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.mode == 'pack':
        pack(arguments)
    else:
        fk(arguments)
