"""F1 asset, identification, admission and checkpoint contracts (no simulator imports)."""
import hashlib
import json
import math
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
LOOP_ROOT = TRAINING_ROOT.parent
CONFIG_ROOT = TRAINING_ROOT / 'configs'
IDENTIFICATION_FILE = CONFIG_ROOT / 'parameter_identification.json'
ROBOT_FILE = CONFIG_ROOT / 'robot_contract.json'
TASKS = ('f1_walk_stage1', 'f1_walk_stage2')


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


ROBOT_CONTRACT = read_json(ROBOT_FILE)


def nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label}: requires a finite, nonnegative identified value; no fallback')
    return float(value)


def identified_range(nominal, interval, label):
    if interval is None:
        return [nominal, nominal]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f'{label}: expected [lower, upper] or null for fixed nominal')
    lower, upper = [nonnegative(value, label) for value in interval]
    if not lower <= nominal <= upper:
        raise ValueError(f'{label}: interval must contain the identified nominal')
    return [lower, upper]


def validate_identification(payload=None):
    data = read_json(IDENTIFICATION_FILE) if payload is None else payload
    if data.get('status') != 'ready':
        raise ValueError('F1 parameter identification is pending; no training is permitted')
    if data.get('schema_version') != 1 or data.get('robot') != 'F1-12DOF':
        raise ValueError('F1 identification schema/robot mismatch')
    urdf = TRAINING_ROOT / ROBOT_CONTRACT['urdf_relative_path']
    if sha256(urdf) != ROBOT_CONTRACT['urdf_sha256'] or data.get('urdf_sha256') != ROBOT_CONTRACT['urdf_sha256']:
        raise ValueError('Identification must bind to this exact F1 URDF SHA256')
    sources = data.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Actual identification data source file/fields are required')
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get('file'), str) or not source['file'].strip() or not isinstance(source.get('fields'), list) or not source['fields']:
            raise ValueError('Each identification source requires file and nonempty fields')
        if not all(isinstance(field, str) and field.strip() for field in source['fields']):
            raise ValueError('Identification source fields must be nonempty strings')
    passive = data.get('passive_dynamics', {})
    if passive.get('mode') != 'preserve_urdf':
        raise ValueError('Current approved scope preserves URDF passive damping/friction')
    joints = data.get('joints', {})
    if set(joints) != set(ROBOT_CONTRACT['joint_names']):
        raise ValueError('Identification must cover exactly the F1 actuated joint names')
    active_friction = False
    for name in ROBOT_CONTRACT['joint_names']:
        if not isinstance(joints[name], dict):
            raise ValueError(f'Invalid joint identification: {name}')
        for key in ('armature', 'coulomb', 'viscous'):
            value = nonnegative(joints[name].get(key), f'{name}.{key}')
            interval = identified_range(value, joints[name].get(key + '_range'), f'{name}.{key}_range')
            if key in ('coulomb', 'viscous') and interval[1] > 0:
                active_friction = True
    if active_friction and passive.get('explicit_friction_semantics') != 'additional_to_urdf':
        raise ValueError('Explicit friction must be identified as additional_to_urdf; resolve double counting first')
    delivery = data.get('delivery', {})
    if not isinstance(delivery.get('enabled'), bool):
        raise ValueError('Identification must explicitly enable/disable delivery LPF')
    names = delivery.get('joint_names')
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names) or len(names) != len(set(names)) or not set(names) <= set(joints):
        raise ValueError('Invalid delivery joint-name mapping')
    if delivery['enabled'] and (not names or nonnegative(delivery.get('tau_s'), 'delivery.tau_s') <= 0):
        raise ValueError('Enabled delivery requires identified positive tau_s and joint names')
    return data


def assert_run_admission(activity='training'):
    if activity not in ('training', 'evaluation'):
        raise ValueError(f'Unknown activity: {activity}')
    launch = read_json(CONFIG_ROOT / 'launch.json')
    if launch.get(activity + '_enabled') is not True or not launch.get('authorization_source'):
        raise ValueError(f'F1 {activity} disabled: scaffold-only, awaiting identification and owner instruction')
    oma = LOOP_ROOT / '.oma'
    if oma.exists():
        config = read_json(oma / 'config.json')
        field = 'trainingEnabled' if activity == 'training' else 'evaluationEnabled'
        if config.get(field) is not True:
            raise ValueError(f'OMA instance has not enabled {activity}')
        state = read_json(oma / 'loop.json')
        scope = 'loop' if activity == 'training' else 'evaluation'
        scopes = state.get('pausedScopes')
        if not isinstance(scopes, list) or not set(scopes) <= {'loop', 'monitor', 'evaluation', 'retry'}:
            raise ValueError('Missing/invalid OMA canonical pause scopes')
        if scope in scopes:
            raise ValueError(f'OMA {scope} is paused; do not resume implicitly')
    if launch.get('physical_model') == 'owner_baseline':
        from humanoid.f1_owner_baseline import validate_owner_baseline
        return validate_owner_baseline()
    return validate_identification()


def assert_training_admission():
    return assert_run_admission('training')


def apply_identification(cfg, data=None):
    if getattr(cfg.env, 'physical_model', None) == 'owner_baseline':
        if read_json(CONFIG_ROOT / 'launch.json').get('physical_model') != 'owner_baseline':
            raise ValueError('Owner baseline is not selected in the launch contract')
        if data is not None:
            raise ValueError('Cannot combine an owner baseline with identification injection')
        from humanoid.f1_owner_baseline import apply_owner_baseline
        return apply_owner_baseline(cfg)
    identified = validate_identification(data)
    dr = cfg.domain_rand
    # Preserve the asset's passive dynamics; explicit friction is a separate identified layer.
    dr.randomize_joint_friction = False
    dr.randomize_joint_damping = False
    dr.randomize_joint_armature = True
    dr.randomize_joint_armature_each_joint = True
    dr.randomize_coulomb_friction = True
    dr.identified_friction_ranges = {}
    nominal_only = getattr(cfg.env, 'nominal_identification', False)
    for index, name in enumerate(ROBOT_CONTRACT['joint_names']):
        values = identified['joints'][name]
        armature = identified_range(values['armature'], None if nominal_only else values.get('armature_range'), name + '.armature')
        setattr(dr, f'joint_{index + 1}_armature_range', armature)
        dr.identified_friction_ranges[name] = {
            key: identified_range(values[key], None if nominal_only else values.get(key + '_range'), name + '.' + key)
            for key in ('coulomb', 'viscous')
        }
    delivery = identified['delivery']
    dr.enable_delivery = delivery['enabled']
    dr.delivery_joint_ids = [ROBOT_CONTRACT['joint_names'].index(name) for name in delivery['joint_names']]
    dr.delivery_tau_d = delivery['tau_s']
    if delivery['enabled'] and delivery['tau_s'] < cfg.sim.dt:
        raise ValueError('Delivery tau_s is below sim.dt; current discrete LPF requires review')
    cfg.identification_sha256 = sha256(IDENTIFICATION_FILE)
    return cfg


def checkpoint_metadata_path(path):
    return Path(str(path) + '.f1.json')


def verify_checkpoint(path, required_task=None, required_iteration=None):
    path = Path(path).expanduser().resolve(strict=True)
    metadata = read_json(checkpoint_metadata_path(path))
    expected = {'schema_version': 1, 'robot': 'F1-12DOF',
                'urdf_sha256': ROBOT_CONTRACT['urdf_sha256'],
                'robot_contract_sha256': sha256(ROBOT_FILE),
                'identification_sha256': sha256(IDENTIFICATION_FILE),
                'checkpoint_sha256': sha256(path)}
    if metadata.get('schema_version') == 2:
        from humanoid.f1_owner_baseline import MODEL_FILE, validate_owner_baseline
        model = validate_owner_baseline()
        expected.update(schema_version=2, urdf_sha256=model['urdf_sha256'],
                        identification_sha256=None,
                        physical_model_sha256=sha256(MODEL_FILE),
                        training_profile_sha256=sha256(TRAINING_ROOT / model['profile_file']))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Checkpoint F1 provenance mismatch: {key}')
    if metadata.get('task') not in TASKS or (required_task and metadata['task'] != required_task):
        raise ValueError('Checkpoint is not from the required F1 stage')
    iteration = metadata.get('iteration')
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        raise ValueError('Checkpoint must record completed update count')
    if required_iteration is not None and iteration != required_iteration:
        raise ValueError(f'Expected completed stage-one iteration {required_iteration}, got {iteration}')
    return path, metadata


def write_checkpoint_metadata(path, task, iteration, parent=None, physical_model=None,
                              goal_revision='F1-WALK-v1'):
    if task not in TASKS:
        raise ValueError('Unknown F1 checkpoint task')
    metadata = {'schema_version': 1, 'robot': 'F1-12DOF', 'task': task,
                'goal_revision': goal_revision, 'iteration': int(iteration),
                'urdf_sha256': ROBOT_CONTRACT['urdf_sha256'],
                'robot_contract_sha256': sha256(ROBOT_FILE),
                'identification_sha256': sha256(IDENTIFICATION_FILE),
                'checkpoint_sha256': sha256(path), 'parent': parent}
    if physical_model is not None:
        from humanoid.f1_owner_baseline import MODEL_FILE, validate_owner_baseline
        model = validate_owner_baseline()
        if model != physical_model:
            raise ValueError('Runtime physical model differs from the bound baseline')
        metadata.update(schema_version=2, urdf_sha256=model['urdf_sha256'],
                        identification_sha256=None, physical_model_kind='owner_baseline',
                        physical_model_sha256=sha256(MODEL_FILE),
                        training_profile_sha256=sha256(TRAINING_ROOT / model['profile_file']))
    checkpoint_metadata_path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata
