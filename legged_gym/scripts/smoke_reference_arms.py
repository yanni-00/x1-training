"""Audit hybrid targets, history, masks, unchanged body and real PPO updates."""
import hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.g1.g1_config import G1Cfg

args=get_args();cfg,tc=task_registry.get_cfgs(args.task);base,bt=task_registry.get_cfgs('g1r41')
before,after=class_to_dict(base),class_to_dict(cfg)
arms=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')
for name in ('stiffness','damping'):
    for j in arms:
        assert after['control'][name][j]==getattr(G1Cfg.control,name)[j]
        after['control'][name][j]=before['control'][name][j]
assert after['rewards']['curriculum_init']==0.9980095958541829
after['rewards']['curriculum_init']=before['rewards']['curriculum_init']
assert before==after,'Unexpected configuration change outside declared controller/curriculum'
ta,tb=class_to_dict(tc),class_to_dict(bt)
for k in ('experiment_name','resume_path'):ta['runner'][k]=tb['runner'][k]
assert ta==tb,'Unexpected PPO/network/horizon change'
cls=task_registry.get_task_class(args.task);parent=task_registry.get_task_class('g1r41')
for name in ('step','reset_idx','check_termination','_compute_torques','_preprocess_obs',
             'compute_observations','_arm_rest_error','_reward_orientation_control'):
    assert getattr(cls,name) is getattr(parent,name),name+' changed'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='2e468ff18b5d34797530182c1e87f0e09441ee11aadd24661fff10c1042f6b51'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12600 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16;env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
env.reset()
assert not env.use_disturb and not env.interrupt_mask.any() and not env.disturb_masks.any()
assert torch.all(env.commands[:,env.command_interrupt_flag_dim]==0)
ai=env.reference_arm_indices;bi=[i for i in range(env.num_actions) if i not in ai]
assert len(ai)==8 and len(bi)==15
assert env.curriculum_scale==cfg.rewards.curriculum_init
assert env.cfg.rewards.scales.__dict__==base.rewards.scales.__dict__
raw=torch.randn((env.num_envs,env.num_actions),device=env.device)
target=env.arm_reference_target()
# Independent reconstruction of the accepted diagnostic's exact target formula.
phase=env.foot_indices[:,[1,0]]
pitch=-0.09178921571001411*torch.cos(2*math.pi*phase-0.30418359554562746)
pitch *= (env.commands[:,0]/.5).clamp(-1,1).unsqueeze(1)*(~env.standing_envs_mask).unsqueeze(1)
expected=env.default_dof_pos[:,ai].expand(env.num_envs,-1).clone()
expected[:,0]+=.3+pitch[:,0];expected[:,4]+=.3+pitch[:,1]
expected[:,1]+=.27;expected[:,5]-=.27;expected[:,3]+=.97;expected[:,7]+=.97
assert torch.equal(target,expected),'Target differs from accepted execution'
submitted=raw.clone();submitted[:,ai]=(expected-env.default_dof_pos[:,ai])/cfg.control.action_scale
executed=env.calculate_action(raw)
assert torch.equal(env.actions,submitted),'History must contain hybrid submitted action'
assert torch.equal(executed,submitted.clamp(-cfg.normalization.clip_actions,cfg.normalization.clip_actions))
assert torch.equal(env.actions[:,bi],raw[:,bi]),'Body actor outputs changed'
# Reference target is also identical to the pre-existing reward reference.
reward_target=(env.dof_pos-env._arm_rest_error())[:,ai]
assert torch.allclose(target,reward_target,atol=2e-7)
# A terminating-body contact must still cause termination with reference active.
forces=env.contact_forces.clone();rpy=env.rpy.clone();length=env.episode_length_buf.clone()
env.contact_forces[:]=0;env.rpy[:]=0;env.episode_length_buf[:]=0
env.contact_forces[0,env.termination_contact_indices[0],0]=2
env.check_termination();assert env.reset_buf[0] and not env.time_out_buf[0]
env.contact_forces[:]=forces;env.rpy[:]=rpy;env.episode_length_buf[:]=length
env.reset()
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_reference_arms').resolve()))
assert runner.current_learning_iteration==12600
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12610
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
assert not env.interrupt_mask.any() and not env.disturb_masks.any()
print('SMOKE_REFERENCE_ARMS_OK '+json.dumps({'task':args.task,'resume_iter':12600,'final_iter':12610,
    'finite':True,'policy_physical_outputs':15,'reference_arm_outputs':8,'target_matches_accepted':True,
    'hybrid_history_verified':True,'contact_termination_verified':True,'legacy_masks_false':True,
    'body_PD_reward_scales_initialization_collision_termination_PPO_unchanged':True}))
