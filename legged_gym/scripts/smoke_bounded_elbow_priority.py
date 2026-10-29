"""Verify matched input/objective isolation and real PPO for the elbow probe."""
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
base,bt=task_registry.get_cfgs('g1r34')
before,after=class_to_dict(base),class_to_dict(cfg)
assert getattr(base.rewards,'arm_rest_elbow_weight',.25)==.25
assert cfg.rewards.arm_rest_elbow_weight==2.
if 'arm_rest_elbow_weight' in before['rewards']:
    after['rewards']['arm_rest_elbow_weight']=before['rewards']['arm_rest_elbow_weight']
else:
    del after['rewards']['arm_rest_elbow_weight']
assert before==after,'Unexpected reward/command/reference/PD/observation change'
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after,'Unexpected checkpoint/seed/PPO/network/update change'
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert env.curriculum_scale==base.rewards.curriculum_init
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_bounded_elbow_priority').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
walking=~env.standing_envs_mask
expected=torch.tensor([.5,0.,0.],device=env.device)
assert torch.allclose(env.commands[walking,:3],expected.expand(int(walking.sum()),3),atol=1e-7,rtol=0)
print('SMOKE_BOUNDED_ELBOW_PRIORITY_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['rewards.arm_rest_elbow_weight'],
    'elbow_weight_before':.25,'elbow_weight_after':cfg.rewards.arm_rest_elbow_weight,
    'same_R34_input_optimizer_commands_PPO_other_rewards_references_PD_network_terrain_randomization':True,
    'resume_iter':12000,'final_iter':12010,'finite':True,'new_reward_terms':False}))
