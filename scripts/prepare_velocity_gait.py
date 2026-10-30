"""[OMA] Periodic local gait from fresh native-articulation FK, never stale tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-(av*bv).sum(-1, keepdims=True), aw*bv+bw*av+np.cross(av,bv)), -1)


def periodic_angular_velocity(q, fps):
    previous, following = np.roll(q, 1, axis=0).copy(), np.roll(q, -1, axis=0)
    previous[..., 1:] *= -1
    delta = multiply(following, previous)
    delta = np.where(delta[..., :1] < 0, -delta, delta)
    vector = delta[..., 1:]
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2*np.arctan2(length, delta[..., :1])
    return vector*np.divide(angle, length, out=np.full_like(length, 2), where=length > 1e-12)*(fps/2)


def support_candidates(position, velocity, names, height_quantile, speed_quantile):
    feet = [names.index(name) for name in ['left_ankle_roll_link', 'right_ankle_roll_link']]
    heights = position[:, feet, 2].max(axis=1)
    speeds = np.linalg.norm(velocity[:, feet], axis=-1).max(axis=1)
    height_limit = float(np.quantile(heights, height_quantile))
    speed_limit = float(np.quantile(speeds, speed_quantile))
    mask = (heights <= height_limit) & (speeds <= speed_limit)
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    if mask.all():
        starts = np.array([0])
    candidates = []
    for start in starts:
        interval = [int(start)]
        while len(interval) < len(mask) and mask[(interval[-1]+1) % len(mask)]:
            interval.append((interval[-1]+1) % len(mask))
        candidates.append(interval[int(np.argmin(speeds[interval]))])
    return np.array(sorted(candidates), dtype=np.int64), height_limit, speed_limit


def prepare(input_file, selected_file, output_file, height_quantile=.25, speed_quantile=.5, seam_tolerance=3e-5):
    selected_path = Path(selected_file)
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    csv = selected_path.parent/selected['csv']
    if hashlib.sha256(csv.read_bytes()).hexdigest() != selected['csv_sha256']:
        raise ValueError('Prepared CSV checksum mismatch')
    if not (0 < height_quantile < 1 and 0 < speed_quantile < 1 and seam_tolerance > 0):
        raise ValueError('Invalid kinematic audit settings')
    with np.load(input_file, allow_pickle=False) as data:
        a = {key:data[key].copy() for key in data.files}
    n, fps = int(selected['cycle_intervals']), float(np.asarray(a['fps']).reshape(-1)[0])
    if fps != selected['input_fps'] or len(a['joint_pos']) != n+1:
        raise ValueError('Native FK must include every same-FPS sample and duplicated endpoint')
    names = a['body_names'].tolist()
    pelvis = names.index('pelvis')
    if pelvis != 0 or a['joint_pos'].shape != (n+1, 29):
        raise ValueError('Unexpected native root or joint dimensions')
    for key in ['joint_pos','body_pos_w','body_quat_w']:
        if not np.isfinite(a[key]).all():
            raise ValueError('Non-finite native FK '+key)
    quaternion = a['body_quat_w'].astype(np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.max(np.abs(norm-1)) > seam_tolerance:
        raise ValueError('Native body quaternions are not normalized')
    quaternion /= norm
    local = a['body_pos_w'].astype(np.float64)
    local[..., :2] -= local[:, pelvis:pelvis+1, :2].copy()
    seam = {
        'joint_position_rad':float(np.max(np.abs(a['joint_pos'][-1]-a['joint_pos'][0]))),
        'body_local_position_m':float(np.max(np.abs(local[-1]-local[0]))),
        'body_quaternion_dot_error':float(np.max(1-np.abs((quaternion[-1]*quaternion[0]).sum(-1))))}
    if any(value > seam_tolerance for value in seam.values()):
        raise ValueError('Native FK cyclic closure failed: '+json.dumps(seam))
    nominal = float(selected['nominal_reference_speed_mps'])
    if nominal <= 0 or not np.isfinite(nominal):
        raise ValueError('Invalid reference nominal speed')
    # Drop duplicated endpoint only after native closure verification.
    q = a['joint_pos'][:n].astype(np.float64)
    local, quaternion = local[:n], quaternion[:n]
    centered = lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))*(fps/2)
    local_v = centered(local)
    nominal_world_v = local_v.copy()
    nominal_world_v[..., 0] += nominal
    support, height_limit, speed_limit = support_candidates(local, nominal_world_v, names, height_quantile, speed_quantile)
    report = {'format':'beyondmimic_velocity_gait_native_audit_v1',
              'native_fk_file_sha256':hashlib.sha256(Path(input_file).read_bytes()).hexdigest(),
              'source_csv_sha256':selected['csv_sha256'],'fps':fps,'cycle_frames':n,
              'nominal_reference_speed_mps':nominal,'native_fk_recomputed':True,
              'closure':seam,'seam_tolerance':seam_tolerance,
              'support_frames':support.tolist(),'support_count':len(support),
              'height_quantile':height_quantile,'speed_quantile':speed_quantile,
              'height_threshold_m':height_limit,'speed_threshold_mps':speed_limit,
              'kinematic_candidates_only':True,'dynamic_feasibility_verified':False,
              'behavior_acceptance':False}
    report_path = Path(output_file).with_suffix('.native-audit.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if len(support) < 2:
        raise ValueError('Insufficient native circular support candidates; inspect native-audit.json')
    output = {'fps':np.asarray([fps]),'joint_names':a['joint_names'],'body_names':a['body_names'],
              'joint_pos':q.astype(np.float32),'joint_vel':centered(q).astype(np.float32),
              'body_pos_w':local.astype(np.float32),'body_quat_w':quaternion.astype(np.float32),
              'body_lin_vel_w':local_v.astype(np.float32),
              'body_ang_vel_w':periodic_angular_velocity(quaternion,fps).astype(np.float32),
              'support_frames':support,'nominal_reference_speed_mps':np.array([nominal]),
              'cycle_frames':np.array([n],dtype=np.int64),
              'reference_space':np.array('cyclic_pelvis_xy_local_v1'),
              'source_csv_sha256':np.array(selected['csv_sha256'])}
    for key in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        if not np.isfinite(output[key]).all():
            raise ValueError('Non-finite periodic array '+key)
    np.savez(output_file, **output)
    print(json.dumps(report),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--selected',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    prepare(a.input,a.selected,a.output)
