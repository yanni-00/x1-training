"""Verify only the existing height scale changes in a matched-input run."""
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
base,bt=task_registry.get_cfgs('g1r26')
before,after=class_to_dict(base),class_to_dict(cfg)
scales_before=before['rewards']['scales'];scales_after=after['rewards']['scales']
assert {k for k in scales_before if scales_before[k]!=scales_after[k]}=={'base_height'}
assert scales_after['base_height']==2*scales_before['base_height']
scales_after['base_height']=scales_before['base_height']
assert before==after
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='58b8a0d0b96342388321291052b23efb89b226f22387bc47e1fad1215f56a381'
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert 'base_height' in env.reward_names and 'base_height' in env.curriculum_reward_list
assert env.reward_scales['base_height']==cfg.rewards.scales.base_height*env.dt
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_height_weight').resolve()))
assert runner.current_learning_iteration==10800
initial_scale=env.curriculum_scale
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==10810
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_HEIGHT_WEIGHT_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['rewards.scales.base_height'],
    'base_height_scale':cfg.rewards.scales.base_height,'matched_R26_input_iterations_seed_PPO':True,
    'curriculum_init':initial_scale,'feet_reference':cfg.rewards.feet_clearance_reference_height,
    'resume_iter':10800,'final_iter':10810,'finite':True}))
