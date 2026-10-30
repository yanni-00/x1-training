"""Audit saved G1 motions for velocity-conditioned straight-gait references.

All reported velocities and seam residuals come from the actual CSV files.
Candidate selection is kinematic only; it does not establish contact/dynamic
feasibility, trained behavior, or hardware velocity limits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rotation_xyzw(quat):
    q = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ], axis=1).reshape(-1, 3, 3)


def percentile(values):
    return {str(p): float(np.percentile(values, p)) for p in [0, 5, 25, 50, 75, 95, 100]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-s', type=float, default=4.0)
    parser.add_argument('--hop-s', type=float, default=1.0)
    parser.add_argument('--keep', type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    gates = json.loads(args.gates.read_text(encoding='utf-8'))
    fps = manifest['input_fps']
    window = round(args.window_s * fps) + 1
    hop = round(args.hop_s * fps)
    if min(window, hop, args.keep) <= 0:
        raise ValueError('Invalid audit settings')
    knee = manifest['joint_names'].index('left_knee_joint')
    report = {
        'scope': 'Data audit only, not dynamics or joystick behavior acceptance',
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'input_fps_source': 'manifest.input_fps', 'input_fps': fps,
        'settings': {'window_s': args.window_s, 'hop_s': args.hop_s, 'keep_per_file': args.keep,
                     'selection_speed_mps': gates['stop_speed_max_mps'],
                     'selection_path_m': gates['go_path_min_m']},
        'settings_source': 'Window4s matches old WalkStopCommandCfg maximum GO setting; hop/keep '
                           'are audit sampling choices. Speed/path filters reuse frozen behavior_gates.json '
                           'as kinematic selection settings, not hardware limits.',
        'velocity_method': 'np.gradient(root XYZ,1/input_fps), rotate into root body coordinates; '
                           'yaw_rate=np.gradient(unwrapped quaternion yaw,1/input_fps)',
        'files': [], 'candidate_windows': [],
        'remaining': ['Exact G1 FK and support/contact checks', 'Periodic seam/velocity validation',
                      'Command-conditioned reference and native PPO implementation',
                      'Actual command-versus-state CSV/video and deployment interface checks'],
    }
    for entry in manifest['files']:
        path = args.manifest.parent / entry['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        data = np.loadtxt(path, delimiter=',')
        assert data.shape == (entry['frames'], len(manifest['joint_names'])+7)
        assert np.isfinite(data).all()
        rotation = rotation_xyzw(data[:, 3:7])
        yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
        velocity = np.gradient(data[:, :3], 1/fps, axis=0)
        velocity_b = np.einsum('nji,nj->ni', rotation, velocity)
        yaw_rate = np.gradient(yaw, 1/fps)
        joints = data[:, 7:]
        dq = np.gradient(joints, 1/fps, axis=0)
        candidates = []
        for start in range(0, len(data)-window+1, hop):
            end = start + window - 1
            pos = data[start:end+1, :2]
            distance = float(np.linalg.norm(pos[-1]-pos[0]))
            path_length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
            vx = float(velocity_b[start:end+1, 0].mean())
            if vx <= gates['stop_speed_max_mps'] or distance < gates['go_path_min_m']:
                continue
            tangent = (pos[-1]-pos[0]) / distance
            offset = pos-pos[0]
            lateral = offset[:, 0]*tangent[1]-offset[:, 1]*tangent[0]
            yaw_span = float(np.ptp(yaw[start:end+1]))
            candidate = {
                'filename': entry['filename'], 'source_sha256': entry['sha256'],
                'start_frame': start, 'end_frame': end, 'start_s': start/fps, 'end_s': end/fps,
                'duration_s': (end-start)/fps, 'forward_mean_mps': vx,
                'lateral_mean_mps': float(velocity_b[start:end+1, 1].mean()),
                'yaw_rate_mean_radps': float(yaw_rate[start:end+1].mean()),
                'yaw_span_rad': yaw_span, 'net_yaw_rad': float(yaw[end]-yaw[start]),
                'displacement_m': distance, 'path_m': path_length,
                'straightness_ratio': distance/path_length,
                'line_residual_rms_m': float(np.sqrt(np.mean(lateral*lateral))),
            }
            # This ranks reference geometry only, never policy rollout outcomes.
            # Pelvis yaw naturally oscillates within a human gait. Rank net
            # turning and path geometry, rather than suppressing that motion.
            candidate['selection_score'] = (abs(candidate['net_yaw_rad']) +
                                            1-candidate['straightness_ratio'] +
                                            candidate['line_residual_rms_m']/distance +
                                            abs(candidate['lateral_mean_mps'])/vx)
            candidates.append(candidate)
        selected = sorted(candidates, key=lambda c: c['selection_score'])[:args.keep]
        for candidate in selected:
            start, end = candidate['start_frame'], candidate['end_frame']
            knee_values = joints[start:end+1, knee]
            level = np.median(knee_values)
            crossings = np.flatnonzero((knee_values[:-1] <= level) & (knee_values[1:] > level)) + start + 1
            cycles = []
            for first, last in zip(crossings[:-1], crossings[1:]):
                q_error, dq_error = joints[last]-joints[first], dq[last]-dq[first]
                cycles.append({'start_frame': int(first), 'end_frame': int(last),
                               'duration_s': float((last-first)/fps),
                               'displacement_m': float(np.linalg.norm(data[last, :2]-data[first, :2])),
                               'net_yaw_rad': float(yaw[last]-yaw[first]),
                               'joint_seam_max_rad': float(np.max(np.abs(q_error))),
                               'joint_seam_rms_rad': float(np.sqrt(np.mean(q_error*q_error))),
                               'joint_velocity_seam_rms_radps': float(np.sqrt(np.mean(dq_error*dq_error))),
                               'support_validated': False})
            candidate['cycle_candidates'] = sorted(cycles, key=lambda c: c['joint_seam_rms_rad'])[:3]
        report['candidate_windows'].extend(selected)
        report['files'].append({'filename': entry['filename'], 'frames': len(data),
                                'sha256': entry['sha256'], 'forward_velocity_mps': percentile(velocity_b[:, 0]),
                                'lateral_velocity_mps': percentile(velocity_b[:, 1]),
                                'yaw_rate_radps': percentile(yaw_rate),
                                'eligible_forward_windows': len(candidates)})
    report['candidate_windows'].sort(key=lambda c: c['selection_score'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'files_audited': len(report['files']),
                      'candidate_windows': len(report['candidate_windows']),
                      'best_geometry_candidates': report['candidate_windows'][:3]}, allow_nan=False))


if __name__ == '__main__':
    main()
