"""[OMA] Exact motion/control state needed by the existing MuJoCo play path."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


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
    reward_parameters={'motion_body_lin_vel':{
        'weight':float(base.cfg.rewards.motion_body_lin_vel.weight),
        'std':float(base.cfg.rewards.motion_body_lin_vel.params['std'])}}
    foot_term=getattr(base.cfg.rewards,'feet_contact_planar_speed',None)
    if foot_term is not None and foot_term.weight != 0:
        reward_parameters['feet_contact_planar_speed']={
            'weight':float(foot_term.weight),'force_threshold':float(foot_term.params['force_threshold']),
            'statistic':'sum_current_contact_body_planar_speed_norm',
            'body_names':['left_ankle_roll_link','right_ankle_roll_link']}
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
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    eps = getattr(normalizer, "eps", None)
    return {
        "format": "beyondmimic_g1_velocity_runtime_v1" if velocity_mode else "beyondmimic_g1_runtime_v1",
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


def validate_runtime_continuation(parent, current, allowed_reward_delta=None):
    """Require the actual new environment interface/control to match its parent."""
    for key in ['format','reference_sha256','joint_names','body_names','tracked_body_names',
                'anchor_body_name','physics_dt','decimation','policy_dt','phase_acceleration',
                'normalizer_epsilon','actor_obs_normalization']:
        if parent[key] != current[key]:raise RuntimeError('Continuation runtime differs: '+key)
    previous=dict(parent['reward_parameters']);resolved=dict(current['reward_parameters'])
    for key,expected in (allowed_reward_delta or {}).items():
        if key!='feet_contact_planar_speed' or key in previous or resolved.pop(key,None)!=expected:
            raise RuntimeError('Unauthorized or mismatched continuation reward delta: '+key)
    if previous!=resolved:raise RuntimeError('Continuation runtime differs: reward_parameters')
    for key,old in parent['control'].items():
        if not torch.equal(old.cpu(),current['control'][key].cpu()):
            raise RuntimeError('Continuation native control differs: '+key)
    return {'same_reference_interface_control':True}
