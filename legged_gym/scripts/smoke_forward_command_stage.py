"""Verify command-stage isolation, live reward-time commands and actual PPO."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg, tc=task_registry.get_cfgs(args.task)
base, bt=task_registry.get_cfgs('g1r33')
before, after=class_to_dict(base), class_to_dict(cfg)
assert cfg.commands.curriculum is False
assert cfg.commands.ranges.lin_vel_x == [.5,.5]
assert cfg.commands.ranges.lin_vel_y == [0.,0.]
assert cfg.commands.ranges.ang_vel_yaw == [0.,0.]
after['commands']['curriculum']=before['commands']['curriculum']
for key in ('lin_vel_x','lin_vel_y','ang_vel_yaw'):
    after['commands']['ranges'][key]=before['commands']['ranges'][key]
assert before==after, 'Unexpected reward/reference/PD/terrain/observation change'
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after, 'Unexpected checkpoint/seed/PPO/network/update change'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert env.curriculum_scale==base.rewards.curriculum_init
counts={'reward_calls':0,'walking_samples':0,'standing_samples':0}
def verify_commands():
    walking=~env.standing_envs_mask
    expected=torch.tensor([.5,0.,0.],device=env.device)
    assert torch.allclose(env.commands[walking,:3],expected.expand(int(walking.sum()),3),atol=1e-7,rtol=0)
    assert torch.all(env.commands[~walking,:3]==0)
    counts['walking_samples']+=int(walking.sum())
    counts['standing_samples']+=int((~walking).sum())
# Repeatedly exercise sampler/curriculum path before real learning.
for _ in range(3):
    env._resample_commands(torch.arange(env.num_envs,device=env.device))
    verify_commands()
original_compute=env.compute_reward
def checked_reward():
    verify_commands()
    counts['reward_calls']+=1
    return original_compute()
env.compute_reward=checked_reward
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_forward_command_stage').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
assert counts['walking_samples']>0 and counts['standing_samples']>0
print('SMOKE_FORWARD_COMMAND_STAGE_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['commands.curriculum','commands.ranges.lin_vel_x','commands.ranges.lin_vel_y','commands.ranges.ang_vel_yaw'],
    'same_R33_input_optimizer_PPO_rewards_references_PD_network_terrain_randomization':True,
    'resume_iter':12000,'final_iter':12010,'finite':True,'new_reward_terms':False,
    'runtime_command_checks':counts}))
