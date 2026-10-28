"""Verify actual terminal/optimizer loading and unchanged objective for R33."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args();cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r32')
before,after=class_to_dict(base),class_to_dict(cfg)
expected=base.rewards.curriculum_init
for _ in range(bt.runner.max_iterations//100):
    expected=pow(expected,base.rewards.penalize_curriculum_sigma)
assert cfg.rewards.curriculum_init==expected
after['rewards']['curriculum_init']=before['rewards']['curriculum_init']
assert before==after,'Unexpected objective/command/PD/observation change'
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
for key in ('experiment_name','resume_path'):
    t_after['runner'][key]=t_before['runner'][key]
assert t_before==t_after,'Unexpected seed/PPO/network/update-count change'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert env.curriculum_scale==expected
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_torso_arm_continuation').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_TORSO_ARM_CONTINUATION_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['rewards.curriculum_init'],'restored_penalty_scalar':expected,
    'same_R32_objective_commands_PD_PPO_network_observations':True,'input_optimizer_present':True,
    'resume_iter':12000,'final_iter':12010,'finite':True,'new_reward_terms':False}))
