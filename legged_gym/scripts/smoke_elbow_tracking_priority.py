"""Real PPO smoke: sole elbow weight difference, matched input, finite updates."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

args=get_args()
cfg,tc=task_registry.get_cfgs(args.task)
base,bt=task_registry.get_cfgs('g1r30')
before,after=class_to_dict(base),class_to_dict(cfg)
assert 'arm_rest_elbow_weight' not in before['rewards']
assert after['rewards'].pop('arm_rest_elbow_weight') == base.rewards.arm_swing_pitch_weight == 20.
assert before==after, 'Unexpected environment change'
t_before,t_after=class_to_dict(bt),class_to_dict(tc)
t_after['runner']['experiment_name']=t_before['runner']['experiment_name']
assert t_before==t_after, 'Unexpected seed/PPO/network/input/update-count change'
assert hashlib.sha256(Path(tc.runner.resume_path).read_bytes()).hexdigest()=='17b4a0a3dbbf9844841f00a7e8d4c8030911c753d68043b8aa8ea89a2cd75985'
cfg.env.num_envs=16
env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
env.reset()  # Initialize the gait clocks through the normal environment path.
# Same registered reward set and scale; generic-class fallback preserves R30.
saved=cfg.rewards.arm_rest_elbow_weight
registered_rewards=cfg.rewards
fallback_fields=class_to_dict(registered_rewards)
fallback_fields.pop('arm_rest_elbow_weight')
cfg.rewards=SimpleNamespace(**fallback_fields)
fallback=env._reward_shoulder_deviation().clone()
cfg.rewards.arm_rest_elbow_weight=.25
assert torch.equal(fallback,env._reward_shoulder_deviation())
cfg.rewards=registered_rewards
runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=tc,
    log_root=str(Path('resources/smoke_elbow_tracking_priority').resolve()))
assert runner.current_learning_iteration==11400
runner.learn(num_learning_iterations=10,init_at_random_ep_len=True)
assert runner.current_learning_iteration==11410
assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ELBOW_TRACKING_PRIORITY_OK '+json.dumps({'task':args.task,'elbow_weight':saved,
    'changed_environment_fields':['rewards.arm_rest_elbow_weight'],
    'matched_R30_input_updates_seed_PPO_curriculum':True,'R30_fallback_exact':True,
    'resume_iter':11400,'final_iter':11410,'finite':True,'new_reward_terms':False}))
