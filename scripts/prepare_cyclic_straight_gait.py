"""Prepare a source-audited periodic straight-gait CSV, pending native FK/play.

Endpoint averaging and periodic splines are explicit data transformations.
Their residuals are recorded; this is not a dynamics feasibility certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-fps', type=int, default=50)
    args = p.parse_args()
    selected = json.loads(args.selection.read_text(encoding='utf-8'))['selected']
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    path = args.manifest.parent/selected['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == selected['source_sha256']
    original = np.loadtxt(path, delimiter=',')[selected['start_frame']:selected['end_frame']+1]
    source_fps = manifest['input_fps']
    source_period = (len(original)-1)/source_fps
    count = round(source_period*args.output_fps)
    if count < 2:
        raise ValueError('Insufficient cycle samples')
    period = count/args.output_fps
    phase = np.linspace(0.0, 1.0, len(original))
    orientation = Rotation.from_quat(original[:, 3:7])
    rpy = orientation.as_euler('xyz')
    yaw = np.unwrap(rpy[:, 2])
    # Preserve the measured within-stride yaw; remove the recorded route turn.
    rpy[:, 2] = yaw-(yaw[0]+phase*(yaw[-1]-yaw[0]))
    values = np.column_stack([original[:, 7:], original[:, 2], rpy])
    transformed = values.copy()
    transformed[0] = transformed[-1] = (values[0]+values[-1])*0.5
    spline = CubicSpline(phase, transformed, axis=0, bc_type='periodic')
    out_phase = np.linspace(0.0, 1.0, count+1)
    fitted = spline(out_phase)
    output = np.zeros((count+1, original.shape[1]))
    output[:, 0] = selected['forward_stride_m']*out_phase
    output[:, 2] = fitted[:, 29]
    output[:, 3:7] = Rotation.from_euler('xyz', fitted[:, 30:33]).as_quat()
    output[:, 7:] = fitted[:, :29]
    assert np.isfinite(output).all() and (output[:, 2] > 0).all()
    joints = {j.attrib['name']: j for j in ET.parse(args.urdf).getroot().findall('joint')}
    limit_checks = {}
    for index, name in enumerate(manifest['joint_names']):
        limit = joints[name].find('limit')
        low, high = float(limit.attrib['lower']), float(limit.attrib['upper'])
        q = output[:, 7+index]
        ok = bool(q.min() >= low-1e-6 and q.max() <= high+1e-6)
        limit_checks[name] = {'passed': ok, 'min_rad': float(q.min()), 'max_rad': float(q.max()),
                              'urdf_lower_rad': low, 'urdf_upper_rad': high}
    source_fit = spline(phase)
    q_error = source_fit[:, :29]-values[:, :29]
    q_seam = spline(1.0)[:29]-spline(0.0)[:29]
    dq_seam = (spline(1.0, 1)[:29]-spline(0.0, 1)[:29])/period
    ddq_seam = (spline(1.0, 2)[:29]-spline(0.0, 2)[:29])/period**2
    report = {
        'scope': 'Prepared kinematic CSV only; exact native FK and dynamic training/play still required',
        'source': selected, 'source_period_s': source_period, 'output_period_s': period,
        'period_quantization_rule': 'Nearest integer number of existing50Hz policy steps',
        'output_fps': args.output_fps, 'cycle_intervals': count, 'rows_including_duplicate_endpoint': count+1,
        'root_forward_stride_m': selected['forward_stride_m'],
        'nominal_reference_speed_mps': selected['forward_stride_m']/period,
        'source_joint_fit_max_rad': float(np.abs(q_error).max()),
        'source_joint_fit_rms_rad': float(np.sqrt(np.mean(q_error*q_error))),
        'analytic_joint_position_seam_max_rad': float(np.abs(q_seam).max()),
        'analytic_joint_velocity_seam_max_radps': float(np.abs(dq_seam).max()),
        'analytic_joint_acceleration_seam_max_radps2': float(np.abs(ddq_seam).max()),
        'root_yaw_residual_max_rad': float(np.abs(fitted[:, 32]).max()),
        'straight_root_y_exact': bool(np.all(output[:, 1] == 0)),
        'joint_limit_checks': limit_checks, 'all_joint_limits_pass': all(v['passed'] for v in limit_checks.values()),
        'urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        'dynamic_feasibility_verified': False, 'native_fk_recomputed': False, 'behavior_acceptance': False,
        'remaining': ['Recompute exact native G1 FK; validate body/foot seams and support',
                      'Command-conditioned cyclic phase and support-aware braking',
                      'Native PPO new-task run and actual command-tracking CSV/video'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not report['all_joint_limits_pass']:
        (args.output.with_suffix('.rejected.json')).write_text(json.dumps(report, indent=2)+'\n')
        raise RuntimeError('Prepared gait violates URDF joint limits; no CSV accepted')
    np.savetxt(args.output, output, delimiter=',', fmt='%.10g')
    report['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['joint_limit_checks', 'source']}))


if __name__ == '__main__':
    main()
