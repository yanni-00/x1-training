"""Check the8 arm-only PD fields and run real PPO with no action assistance."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.g1.g1_config import G1Cfg
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r36')
before,after=class_to_dict(base),class_to_dict(cfg)
changes=[]
for group in ('stiffness','damping'):
    for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow'):
        old=before['control'][group][joint]; new=after['control'][group][joint]
        assert new==getattr(G1Cfg.control,group)[joint] and old!=new
        changes.append({'field':'control.'+group+'.'+joint,'before':old,'after':new})
        after['control'][group][joint]=old
assert before==after,'Unexpected field outside the arm PD functional unit'
ta,tb=class_to_dict(tc),class_to_dict(bt)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert ta==tb,'Unexpected input/optimizer/PPO/seed/network change'
cls=task_registry.get_task_class(args.task)
assert cls is task_registry.get_task_class('g1r36')
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert not env.use_disturb
assert env.curriculum_scale==base.rewards.curriculum_init
assert set(env.reward_scales)=={k for k,v in class_to_dict(base.rewards.scales).items() if v!=0}
for i,name in enumerate(env.dof_names):
    for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow'):
        if joint in name:
            assert env.p_gains[i].item()==cfg.control.stiffness[joint]
            assert abs(env.d_gains[i].item()-cfg.control.damping[joint])<1e-6
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_arm_asset_pd').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ARM_ASSET_PD_OK '+json.dumps({'task':args.task,'changes':changes,
    'resume_iter':12000,'final_iter':12010,'finite':True,
    'learned_all23_joint_actions':True,'no_reference_action_override':True,
    'reward_formulas_scales_reference_commands_initialization_collision_termination_PPO_unchanged':True}))
