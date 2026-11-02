"""Synthetic CPU checks: real reward selection and unauthorized continuation rejection."""
import ast,copy,importlib.util,subprocess,sys,types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'source/whole_body_tracking/whole_body_tracking';reward_file=PKG/'tasks/tracking/mdp/rewards.py'
spec=importlib.util.spec_from_file_location('checkpoint_audit',PKG/'utils/oma_checkpoint.py');checkpoint=importlib.util.module_from_spec(spec);spec.loader.exec_module(checkpoint)
# Execute the two actual pure reward functions, with postponed Isaac type names.
tree=ast.parse(reward_file.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['_get_body_indexes','motion_global_body_linear_velocity_error_exp']]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[]);ast.fix_missing_locations(module)
ns={'torch':torch,'__name__':'fixture_rewards'};exec(compile(module,str(reward_file),'exec'),ns);reward=ns['motion_global_body_linear_velocity_error_exp']
S=types.SimpleNamespace
names=['pelvis','left_ankle_roll_link','right_ankle_roll_link','torso_link']
command=S(cfg=S(body_names=names),body_lin_vel_w=torch.zeros(2,4,3),robot_body_lin_vel_w=torch.zeros(2,4,3))
command.robot_body_lin_vel_w[:,0,0]=torch.tensor([.15,.3]);command.robot_body_lin_vel_w[:,1:]=10
native_env=S(command_manager=S(get_term=lambda name:command))
before=reward(native_env,'motion',.3,['pelvis']);command.robot_body_lin_vel_w[:,1:]=100
assert torch.equal(before,reward(native_env,'motion',.3,['pelvis'])),'Non-pelvis velocities leaked into pelvis reward'
command.robot_body_lin_vel_w[:,0,0]*=2
assert torch.all(reward(native_env,'motion',.3,['pelvis'])<before),'Pelvis error does not reduce the actual reward'
term=S(weight=.5,params={'command_name':'motion','std':.3,'body_names':['pelvis']},func=reward)
foot=S(weight=-.2,params={'force_threshold':10})
rewards=S(motion_body_lin_vel=S(weight=1.,params={'std':1.}),feet_contact_planar_speed=foot,motion_pelvis_lin_vel=term)
manager=S(active_terms=['motion_pelvis_lin_vel'],get_term_cfg=lambda name:term)
base=S(cfg=S(rewards=rewards),reward_manager=manager)
params,audits=checkpoint.audit_reward_parameters(base,command)
assert audits['motion_pelvis_lin_vel']['selected_body_indices']==[0]
assert audits['motion_pelvis_lin_vel']['enabled_in_native_manager']
assert params['feet_contact_planar_speed']['weight']==-.2

def rejects(call):
 try:call()
 except RuntimeError:return
 raise AssertionError('Unsafe native reward/continuation was accepted')
term.params['body_names']=names;rejects(lambda:checkpoint.audit_reward_parameters(base,command));term.params['body_names']=['pelvis']
term.params['std']=float('nan');rejects(lambda:checkpoint.audit_reward_parameters(base,command));term.params['std']=.3
manager.active_terms=[];rejects(lambda:checkpoint.audit_reward_parameters(base,command));manager.active_terms=['motion_pelvis_lin_vel']
wrong_live=copy.deepcopy(term);wrong_live.weight=1.;manager.get_term_cfg=lambda name:wrong_live
rejects(lambda:checkpoint.audit_reward_parameters(base,command));manager.get_term_cfg=lambda name:term
command.cfg.body_names=['missing',*names[1:]];rejects(lambda:checkpoint.audit_reward_parameters(base,command));command.cfg.body_names=names
term.weight=0;old,old_audits=checkpoint.audit_reward_parameters(base,command)
assert 'motion_pelvis_lin_vel' not in old and old_audits=={},'Default disabled path changed legacy metadata'
term.weight=.5
parent={k:None for k in ['format','reference_sha256','joint_names','body_names','tracked_body_names','anchor_body_name','physics_dt','decimation','policy_dt','phase_acceleration','normalizer_epsilon','actor_obs_normalization']}
parent.update(reward_parameters=old,control={'kp':torch.ones(29)},deployment_target_contract={'version':'shared_cpp_target_limits_v1','applied_to_native_action':True})
current=copy.deepcopy(parent);current['reward_parameters']=params;allowed={'motion_pelvis_lin_vel':params['motion_pelvis_lin_vel']}
checkpoint.validate_runtime_continuation(parent,current,allowed)
rejects(lambda:checkpoint.validate_runtime_continuation(parent,current))
wrong=copy.deepcopy(current);wrong['reward_parameters']['motion_body_lin_vel']['std']=.5;rejects(lambda:checkpoint.validate_runtime_continuation(parent,wrong,allowed))
wrong=copy.deepcopy(current);wrong['reward_parameters']['feet_contact_planar_speed']['weight']=-.4;rejects(lambda:checkpoint.validate_runtime_continuation(parent,wrong,allowed))
wrong=copy.deepcopy(current);wrong['control']['kp']*=2;rejects(lambda:checkpoint.validate_runtime_continuation(parent,wrong,allowed))
# An incomplete explicit flag pair must fail before any simulator/asset/network work.
r=subprocess.run([sys.executable,str(ROOT/'scripts/oma_bootstrap.py'),'--velocity_pelvis_weight','.5'],capture_output=True,text=True)
assert r.returncode!=0 and 'requires both explicit weight and std' in r.stderr
print('PASS actual pelvis-only reward selection; native active/shape/config checks; default disabled legacy metadata; authorized-only reward addition; unchanged foot/control; paired CLI')
