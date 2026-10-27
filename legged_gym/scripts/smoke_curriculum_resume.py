"""Validate a matched-input single-field curriculum-restoration probe."""
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
cfg,train_cfg=task_registry.get_cfgs(args.task)
base,base_train=task_registry.get_cfgs('g1r25')
before,after=class_to_dict(base),class_to_dict(cfg)
changed={key for key in before['rewards'] if before['rewards'][key]!=after['rewards'][key]}
assert changed=={'curriculum_init'},changed
after['rewards']['curriculum_init']=before['rewards']['curriculum_init']
assert before==after
bt,nt=class_to_dict(base_train),class_to_dict(train_cfg)
nt['runner']['experiment_name']=bt['runner']['experiment_name']
assert bt==nt, 'Unexpected optimizer/network/input/iteration/seed change'
checkpoint=Path(train_cfg.runner.resume_path)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='58b8a0d0b96342388321291052b23efb89b226f22387bc47e1fad1215f56a381'
saved=torch.load(str(checkpoint),map_location='cpu')
assert saved['iter']==10800 and saved['infos'] is None
assert cfg.rewards.curriculum_init==0.8952970367055774 and cfg.rewards.penalize_curriculum
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
initial_scale=env.curriculum_scale
assert initial_scale==cfg.rewards.curriculum_init and env.learning_iter==0
for _ in range(100):
    env.training_curriculum()
assert env.curriculum_scale==pow(initial_scale,cfg.rewards.penalize_curriculum_sigma)
env.learning_iter=0
env.curriculum_scale=initial_scale
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=train_cfg,
    log_root=str(Path('resources/smoke_curriculum_resume').resolve()))
assert runner.current_learning_iteration==10800 and env.curriculum_scale==initial_scale
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==10810
assert env.curriculum_scale==initial_scale
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
result={'task':args.task,'changed_environment_fields':['rewards.curriculum_init'],
    'initial_curriculum_scale':initial_scale,'legacy_initial_scale':base.rewards.curriculum_init,
    'deterministic_period_check_passed':True,'matched_R25_input_iterations_seed_and_PPO':True,
    'checkpoint_state_keys':list(saved.keys()),'checkpoint_infos_is_none':True,
    'resume_iter':10800,'final_iter':10810,'finite':True,
    'push_robots':getattr(cfg.domain_rand,'push_robots',None),
    'max_push_vel_xy':getattr(cfg.domain_rand,'max_push_vel_xy',None),
    'max_push_ang_vel_xy':getattr(cfg.domain_rand,'max_push_ang_vel_xy',None)}
print('SMOKE_CURRICULUM_RESUME_OK '+json.dumps(result))
