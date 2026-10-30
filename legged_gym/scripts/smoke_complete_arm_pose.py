"""Verify the entire reference functional unit and execute ten real PPO updates."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r36')
before,after=class_to_dict(base),class_to_dict(cfg)
fields={'arm_rest_shoulder_pitch_rad':.3,'arm_rest_elbow_rad':.97,'arm_rest_shoulder_roll_rad':.27}
changes=[]
for key,value in fields.items():
    assert after['rewards'][key]==value
    changes.append({'field':'rewards.'+key,'before':before['rewards'].get(key),'after':value})
    if key in before['rewards']:after['rewards'][key]=before['rewards'][key]
    else:del after['rewards'][key]
assert before==after,'Unexpected configuration outside the complete rest reference'
ta,tb=class_to_dict(tc),class_to_dict(bt)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert ta==tb,'Unexpected input/optimizer/PPO/seed/network change'
cls=task_registry.get_task_class(args.task)
parent=task_registry.get_task_class('g1r36')
for name in ('step','reset_idx','check_termination','_compute_torques','compute_observations',
             '_reward_shoulder_deviation','_reward_standing_joint_deviation'):
    assert getattr(cls,name) is getattr(parent,name),name+' was overridden'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert not env.use_disturb
assert env.num_actions==23
assert env.curriculum_scale==base.rewards.curriculum_init
assert set(env.reward_scales)=={k for k,v in class_to_dict(base.rewards.scales).items() if v!=0}
# The gait clock buffers are initialized by the first simulation step.
env.step(torch.zeros(env.num_envs,env.num_actions,device=env.device))
expected=env.dof_pos-env.default_dof_pos
expected[:,env.arm_rest_elbow_inds]-=.97
expected[:,env.arm_rest_shoulder_roll_inds]-=torch.tensor([.27,-.27],device=env.device)
phase=env.foot_indices[:,[1,0]]
swing=-cfg.rewards.arm_swing_pitch_amplitude*torch.cos(2*torch.pi*phase-cfg.rewards.arm_swing_phase_offset)
swing*=((env.commands[:,0]/cfg.rewards.arm_swing_speed_reference).clamp(-1.,1.)*(~env.standing_envs_mask)).unsqueeze(1)
expected[:,env.swing_shoulder_pitch_inds]-=.3+swing
assert torch.allclose(env._arm_rest_error(),expected,atol=1e-6),'Whole-arm mapping mismatch'
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_complete_arm_pose').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_COMPLETE_ARM_POSE_OK '+json.dumps({'task':args.task,'changes':changes,
    'resume_iter':12000,'final_iter':12010,'finite':True,
    'learned_all23_joint_actions':True,'no_reference_action_override':True,
    'PD_reward_scales_commands_initialization_collision_termination_PPO_unchanged':True}))
