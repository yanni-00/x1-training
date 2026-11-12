"""Owner-selected URDF baseline, explicitly separate from parameter identification."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / 'configs/owner_baseline_model.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_owner_baseline():
    data = json.loads(MODEL_FILE.read_text(encoding='utf-8'))
    original_contract = ROOT / 'configs/robot_contract.json'
    contract = json.loads(original_contract.read_text(encoding='utf-8'))
    if data.get('kind') != 'owner_baseline' or data.get('robot') != 'F1-12DOF':
        raise ValueError('Invalid owner-baseline identity')
    if data['robot_contract_sha256'] != digest(original_contract):
        raise ValueError('Original robot contract changed')
    if data['joint_names'] != contract['joint_names']:
        raise ValueError('Owner baseline changed the joint mapping')
    for path_key, hash_key in [('source_urdf', 'source_urdf_sha256'),
                               ('urdf_relative_path', 'urdf_sha256')]:
        path = (ROOT / data[path_key]).resolve()
        if ROOT not in path.parents or digest(path) != data[hash_key]:
            raise ValueError('Owner-baseline asset hash/path mismatch: ' + path_key)
    if data['source_urdf_sha256'] != contract['urdf_sha256']:
        raise ValueError('Owner baseline must derive from the original F1 asset')
    source = ET.parse(ROOT / data['source_urdf']).getroot()
    expected = copy.deepcopy(source)
    names = contract['joint_names']
    changes = {item['joint']: item for item in data['changes']}
    if set(changes) != set(names) or len(data['changes']) != len(names):
        raise ValueError('Owner dynamics must cover each active joint exactly once')
    for joint in expected.findall('joint'):
        if joint.get('name') in names:
            node = joint.find('dynamics')
            item = changes[joint.get('name')]
            if dict(node.attrib) != item['before'] or item['after'] != {'damping': 0.05, 'friction': 0.0}:
                raise ValueError('Owner dynamics differ from the recorded instruction')
            node.attrib = {'damping': '0.05', 'friction': '0'}
    actual = ET.parse(ROOT / data['urdf_relative_path']).getroot()
    if ET.tostring(actual) != ET.tostring(expected):
        raise ValueError('Derived URDF changed fields beyond approved joint dynamics')
    if any(data.get(key) is not False for key in (
        'identification_injected', 'extra_armature_enabled', 'explicit_friction_enabled',
        'delay_enabled', 'posture_rotor_model_enabled',
    )) or data['delivery']['enabled'] is not False:
        raise ValueError('Unexpected additional dynamics in owner baseline')
    return data


def apply_owner_baseline(cfg):
    model = validate_owner_baseline()
    actual = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT))).resolve()
    if actual != (ROOT / model['urdf_relative_path']).resolve():
        raise ValueError('Effective environment is not using the approved derived URDF')
    for key in ('randomize_joint_friction', 'randomize_joint_damping',
                'randomize_joint_armature', 'randomize_coulomb_friction',
                'enable_delivery', 'add_lag', 'add_dof_lag',
                'add_dof_pos_vel_lag', 'add_imu_lag'):
        if getattr(cfg.domain_rand, key, False):
            raise ValueError('Owner baseline forbids additional physical layer: ' + key)
    if cfg.asset.armature != 0:
        raise ValueError('Owner baseline forbids added asset armature')
    cfg.owner_physical_model = model
    cfg.identification_sha256 = None
    return cfg
