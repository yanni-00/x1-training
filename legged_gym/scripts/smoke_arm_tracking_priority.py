"""Verify the sole weight difference and finite PPO from the matched input."""
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

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r29')
before,after=class_to_dict(base),class_to_dict(cfg)
assert {k for k in before['rewards'] if before['rewards'][k]!=after['rewards'][k]}=={'arm_swing_pitch_weight'}
assert cfg.rewards.arm_swing_pitch_weight==10*base.rewards.arm_swing_pitch_weight
after['rewards']['arm_swing_pitch_weight']=before['rewards']['arm_swing_pitch_weight']
assert before==after
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after, 'Unexpected input/seed/PPO/network/update-count change'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='17b4a0a3dbbf9844841f00a7e8d4c8030911c753d68043b8aa8ea89a2cd75985'
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_arm_tracking_priority').resolve()))
assert runner.current_learning_iteration==11400
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==11410
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ARM_TRACKING_PRIORITY_OK '+json.dumps({'task':args.task,'pitch_weight':cfg.rewards.arm_swing_pitch_weight,
    'changed_environment_fields':['rewards.arm_swing_pitch_weight'],'matched_R29_input_updates_seed_PPO_curriculum':True,
    'resume_iter':11400,'final_iter':11410,'finite':True,'new_reward_terms':False}))
