"""Check exactly one live external reward scale changes and run actual PPO."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args();cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r36')
before,after=class_to_dict(base),class_to_dict(cfg)
assert base.rewards.scales.shoulder_deviation==-1.
assert cfg.rewards.scales.shoulder_deviation==-2.
after['rewards']['scales']['shoulder_deviation']=before['rewards']['scales']['shoulder_deviation']
assert before==after,'Unexpected reference/internal-weight/command/PD/observation/safety change'
tb,ta=class_to_dict(bt),class_to_dict(tc)
ta['runner']['experiment_name']=tb['runner']['experiment_name']
assert tb==ta,'Unexpected input/seed/PPO/network/update change'
assert task_registry.get_task_class(args.task) is task_registry.get_task_class('g1r36')
path=Path(tc.runner.resume_path)
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5b753bde8247ce500fb848c0562975a5698ff6d0f4605b78aa8b4cd5de0dd730'
saved=torch.load(str(path),map_location='cpu')
assert saved['iter']==12000 and 'optimizer_state_dict' in saved
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
assert env.curriculum_scale==base.rewards.curriculum_init
assert abs(env.reward_scales['shoulder_deviation']-(-2*env.dt))<1e-12
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_arm_posture_priority').resolve()))
assert runner.current_learning_iteration==12000
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==12010
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ARM_POSTURE_PRIORITY_OK '+json.dumps({'task':args.task,
    'changed_environment_fields':['rewards.scales.shoulder_deviation'],
    'external_scale_before':-1.,'external_scale_after':-2.,'runtime_dt_scaled_component':env.reward_scales['shoulder_deviation'],
    'same_R36_input_optimizer_reference_internal_weights_commands_PPO_PD_observations_initialization_collision_termination':True,
    'resume_iter':12000,'final_iter':12010,'finite':True,'new_reward_terms':False}))
