"""[OMA] Exact motion/control state needed by the existing MuJoCo play path."""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import numpy as np
import torch


def audit_reward_parameters(base, command):
    """Inspect the live reward manager; a CLI request alone is not execution proof."""
    rewards=base.cfg.rewards
    parameters={'motion_body_lin_vel':{
        'weight':float(rewards.motion_body_lin_vel.weight),
        'std':float(rewards.motion_body_lin_vel.params['std'])}}
    foot_term=getattr(rewards,'feet_contact_planar_speed',None)
    if foot_term is not None and foot_term.weight != 0:
        parameters['feet_contact_planar_speed']={
            'weight':float(foot_term.weight),'force_threshold':float(foot_term.params['force_threshold']),
            'statistic':'sum_current_contact_body_planar_speed_norm',
            'body_names':['left_ankle_roll_link','right_ankle_roll_link']}
    audits={}
    requested=json.loads(os.environ.get('OMA_VELOCITY_REWARD_DELTA','{}')).get('feet_contact_planar_speed',{})
    if requested.get('kind')=='strengthen_existing_contact_cost_v1':
        if foot_term is None or parameters.get('feet_contact_planar_speed')!=requested.get('after'):
            raise RuntimeError('Native reweighted contact cost differs from requested after state')
        manager=base.reward_manager
        if 'feet_contact_planar_speed' not in manager.active_terms:
            raise RuntimeError('Reweighted contact cost absent from native reward manager')
        live=manager.get_term_cfg('feet_contact_planar_speed')
        source=Path(__file__).resolve().parents[1]/'tasks/tracking/mdp/rewards.py'
        if (live.weight!=foot_term.weight or live.params!=foot_term.params or live.func is not foot_term.func or
                foot_term.func.__name__!='feet_contact_planar_speed' or
                Path(inspect.getsourcefile(foot_term.func) or '').resolve()!=source):
            raise RuntimeError('Reweighted contact cost native function/config mismatch')
        names=['left_ankle_roll_link','right_ankle_roll_link']
        for selector,entity in [('asset_cfg','robot'),('sensor_cfg','contact_forces')]:
            cfg=live.params[selector]
            if cfg.name!=entity or list(cfg.body_names)!=names or not cfg.preserve_order:
                raise RuntimeError('Reweighted contact cost native bilateral selector mismatch')
        audits['feet_contact_planar_speed']={
            'enabled_in_native_manager':True,'resolved_weight':float(live.weight),
            'selected_body_names':names,'function':foot_term.func.__name__,
            'source_sha256':hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()}
    term=getattr(rewards,'motion_pelvis_lin_vel',None)
    if term is not None and term.weight != 0:
        weight,std=float(term.weight),float(term.params['std'])
        if not (math.isfinite(weight) and weight>0 and math.isfinite(std) and std>0):
            raise RuntimeError('Invalid pelvis velocity reward coefficient')
        if term.params.get('body_names')!=['pelvis'] or term.params.get('command_name')!='motion':
            raise RuntimeError('Pelvis velocity reward must select only motion/pelvis')
        source=Path(__file__).resolve().parents[1]/'tasks/tracking/mdp/rewards.py'
        if (term.func.__name__!='motion_global_body_linear_velocity_error_exp' or
                Path(inspect.getsourcefile(term.func) or '').resolve()!=source):
            raise RuntimeError('Unexpected pelvis velocity reward implementation')
        manager=base.reward_manager
        if 'motion_pelvis_lin_vel' not in manager.active_terms:
            raise RuntimeError('Requested pelvis reward absent from native manager')
        live=manager.get_term_cfg('motion_pelvis_lin_vel')
        if live.weight!=term.weight or live.params!=term.params or live.func is not term.func:
            raise RuntimeError('Native pelvis reward differs from resolved configuration')
        names=list(command.cfg.body_names)
        selected=[i for i,name in enumerate(names) if name=='pelvis']
        if len(selected)!=1:
            raise RuntimeError('Pelvis reward requires exactly one native tracked pelvis')
        reference,actual=command.body_lin_vel_w,command.robot_body_lin_vel_w
        if (reference.shape!=actual.shape or reference.ndim!=3 or
                reference.shape[-2:]!=(len(names),3) or
                not torch.isfinite(reference).all() or not torch.isfinite(actual).all()):
            raise RuntimeError('Invalid native body velocity tensors for pelvis reward')
        parameters['motion_pelvis_lin_vel']={
            'weight':weight,'std':std,'command_name':'motion','body_names':['pelvis'],
            'statistic':'motion_global_body_linear_velocity_error_exp'}
        audits['motion_pelvis_lin_vel']={
            'enabled_in_native_manager':True,'selected_body_names':['pelvis'],
            'selected_body_indices':selected,'tracked_body_count':len(names),
            'native_velocity_shape':list(actual.shape),'function':term.func.__name__,
            'source_sha256':hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()}
    return parameters,audits


def collect_runtime(env, policy):
    base = env.unwrapped
    command = base.command_manager.get_term("motion")
    robot = command.robot
    data = robot.data
    velocity_mode = hasattr(command, 'reference')
    velocity_metadata = {}
    with np.load(command.cfg.motion_file, allow_pickle=False) as source:
        joint_names = source["joint_names"].tolist()
        body_names = source["body_names"].tolist()
        motion = {key: torch.from_numpy(source[key].copy()) for key in
                  ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                   "body_lin_vel_w", "body_ang_vel_w", "support_frames"]}
        if velocity_mode:
            for key in ['cycle_frames', 'nominal_reference_speed_mps']:
                motion[key] = torch.from_numpy(source[key].copy())
            velocity_metadata = {
                'reference_space':str(source['reference_space']),
                'source_csv_sha256':str(source['source_csv_sha256']),
                'command_limits':[list(value) for value in command.command_limits],
                'supported_command_scope':('forward_speed_walking_turns_and_stable_stop_training'
                    if any(command.cfg.yaw_rate_range) else 'forward_speed_and_stable_stop_integration_only'),
                'command_constraints':{'lateral':False,'backward':False,'in_place_turn':False},
                'warm_start':json.loads(os.environ.get('OMA_VELOCITY_RESUME_PROVENANCE','null')),
                'main_joystick_goal_accepted':False,
                'training_command_curriculum':{
                    'speed_scale_range':list(command.cfg.speed_scale_range),
                    'yaw_rate_range_radps':list(command.cfg.yaw_rate_range),
                    'straight_go_probability':float(command.cfg.straight_probability),
                    'go_duration_range_s':list(command.cfg.go_duration_range),
                    'stop_duration_range_s':list(command.cfg.stop_duration_range),
                    'initial_go_probability':float(command.cfg.initial_go_probability),
                    'episode_length_s':float(base.cfg.episode_length_s),
                },
                'native_gait_audit':json.loads(Path(command.cfg.motion_file).with_suffix('.native-audit.json').read_text()),
            }
    if joint_names != list(robot.joint_names):
        raise RuntimeError("Checkpoint reference/articulation joint order mismatch")
    if motion["joint_pos"].shape[1] != 29 or len(joint_names) != 29:
        raise RuntimeError("Checkpoint reference joint dimension mismatch")
    if any(not torch.isfinite(value).all() for value in motion.values()):
        raise RuntimeError("Non-finite runtime reference")
    if len(body_names) != motion["body_pos_w"].shape[1]:
        raise RuntimeError("Checkpoint reference body order mismatch")
    reward_parameters,reward_audits=audit_reward_parameters(base,command)
    velocity_metadata['reward_audits']=reward_audits
    velocity_metadata['authorized_reward_delta']=json.loads(os.environ.get('OMA_VELOCITY_REWARD_DELTA','{}'))
    action = base.action_manager.get_term("joint_pos")
    scale = action._scale
    scale = scale[0].detach().cpu().clone() if isinstance(scale, torch.Tensor) else torch.full((29,), float(scale))
    nominal = getattr(data, "default_joint_pos_nominal", None)
    if nominal is None:
        raise RuntimeError("Nominal pre-randomization joint positions were not recorded")
    control = {
        "default_joint_pos": nominal.detach().cpu().clone(),
        "kp": data.default_joint_stiffness[0].detach().cpu().clone(),
        "kd": data.default_joint_damping[0].detach().cpu().clone(),
        "armature": data.default_joint_armature[0].detach().cpu().clone(),
        "effort_limit": data.joint_effort_limits[0].detach().cpu().clone(),
        "action_scale": scale,
    }
    if any(value.shape != (29,) or not torch.isfinite(value).all() for value in control.values()):
        raise RuntimeError("Invalid runtime control parameters")
    from .actor_observations import audit_native_observations
    actor_audit = audit_native_observations(env, policy)
    from .deployment_limits import audit_deployment_action
    target_contract=audit_deployment_action(env)
    proprio = bool(getattr(base.cfg, "oma_proprioceptive_actor", False))
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    eps = getattr(normalizer, "eps", None)
    return {
        "format": ("beyondmimic_g1_velocity_proprio_runtime_v1" if proprio else
                   "beyondmimic_g1_velocity_runtime_v1" if velocity_mode else "beyondmimic_g1_runtime_v1"),
        "actor_observation_contract": actor_audit,
        "deployment_target_contract":target_contract,
        **velocity_metadata,
        "joint_names": joint_names, "body_names": body_names,
        "tracked_body_names": list(command.cfg.body_names),
        "anchor_body_name": command.cfg.anchor_body_name,
        "physics_dt": float(base.cfg.sim.dt), "decimation": int(base.cfg.decimation),
        "policy_dt": float(base.cfg.sim.dt * base.cfg.decimation),
        "phase_acceleration": float(command.cfg.phase_acceleration),
        "normalizer_epsilon": float(eps) if eps is not None else None,
        "actor_obs_normalization": bool(getattr(policy, "actor_obs_normalization", False)),
        "contact_force_threshold": float(base.scene["contact_forces"].cfg.force_threshold),
        # Record resolved reward configuration, not just requested CLI values.
        # This is provenance only; reward computation remains in native MDP.
        "reward_parameters":reward_parameters,
        "reference_sha256": hashlib.sha256(Path(command.cfg.motion_file).read_bytes()).hexdigest(),
        "motion": motion, "control": control,
    }


def validate_runtime_continuation(parent, current, allowed_reward_delta=None, allowed_target_delta=None):
    """Require the actual new environment interface/control to match its parent."""
    for key in ['format','reference_sha256','joint_names','body_names','tracked_body_names',
                'anchor_body_name','physics_dt','decimation','policy_dt','phase_acceleration',
                'normalizer_epsilon','actor_obs_normalization']:
        if parent[key] != current[key]:raise RuntimeError('Continuation runtime differs: '+key)
    previous=dict(parent['reward_parameters']);resolved=dict(current['reward_parameters'])
    for key,expected in (allowed_reward_delta or {}).items():
        if expected.get('kind')=='strengthen_existing_contact_cost_v1':
            if (key!='feet_contact_planar_speed' or len(allowed_reward_delta)!=1 or
                    set(expected)!={'kind','before','after'} or
                    previous.get(key)!=expected['before'] or resolved.get(key)!=expected['after']):
                raise RuntimeError('Unauthorized or mismatched existing contact reweight')
            before,after=dict(expected['before']),dict(expected['after'])
            old_weight,new_weight=before.pop('weight'),after.pop('weight')
            canonical={'force_threshold':10.0,'statistic':'sum_current_contact_body_planar_speed_norm',
                       'body_names':['left_ankle_roll_link','right_ankle_roll_link']}
            if (before!=canonical or after!=canonical or
                    not (math.isfinite(old_weight) and math.isfinite(new_weight) and new_weight<old_weight<0)):
                raise RuntimeError('Contact reweight must only strengthen same existing bilateral cost')
            if not (parent['format']=='beyondmimic_g1_velocity_proprio_runtime_v1' and
                    parent.get('deployment_target_contract',{}).get('applied_to_native_action') is True):
                raise RuntimeError('Contact reweight requires existing native-limited154 parent')
            previous.pop(key);resolved.pop(key)
        elif key not in {'feet_contact_planar_speed','motion_pelvis_lin_vel'} or key in previous or resolved.pop(key,None)!=expected:
            raise RuntimeError('Unauthorized or mismatched continuation reward delta: '+key)
    if previous!=resolved:raise RuntimeError('Continuation runtime differs: reward_parameters')
    for key,old in parent['control'].items():
        if not torch.equal(old.cpu(),current['control'][key].cpu()):
            raise RuntimeError('Continuation native control differs: '+key)
    before=parent.get('deployment_target_contract');after=current.get('deployment_target_contract')
    if before!=after:
        if before is not None or allowed_target_delta!='shared_cpp_target_limits_v1' or after is None or after.get('version')!=allowed_target_delta or after.get('applied_to_native_action') is not True:
            raise RuntimeError('Unauthorized continuation target-processing delta')
    return {'same_reference_interface_control':True,'authorized_target_processing_delta':allowed_target_delta}
