"""Validate a height-only stage and its continuation checkpoint before launch."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


args = get_args()
cfg, train_cfg = task_registry.get_cfgs(args.task)
base, base_train = task_registry.get_cfgs('g1r21')
assert class_to_dict(cfg.rewards.scales) == class_to_dict(base.rewards.scales)
assert class_to_dict(cfg.commands) == class_to_dict(base.commands)
assert class_to_dict(cfg.control) == class_to_dict(base.control)
assert class_to_dict(train_cfg.algorithm) == class_to_dict(base_train.algorithm)
before = class_to_dict(base.rewards)
after = class_to_dict(cfg.rewards)
changed = {name for name in before if before[name] != after[name]}
assert changed == {'base_height_target'}, changed
cfg.env.num_envs = 16
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg,
    log_root=str(Path('resources/smoke_height_stage').resolve()))
initial = runner.current_learning_iteration
assert initial == 8400
runner.learn(num_learning_iterations=10, init_at_random_ep_len=True)
assert runner.current_learning_iteration == initial + 10
assert all(torch.isfinite(value).all() for value in runner.alg.actor_critic.parameters())
print('SMOKE_HEIGHT_STAGE_OK ' + json.dumps({'task': args.task,
    'changed_reward_config_fields': sorted(changed),
    'base_height_target': cfg.rewards.base_height_target,
    'resume_iter': initial, 'final_iter': runner.current_learning_iteration,
    'reward_scales_equal_r21': True, 'commands_equal_r21': True,
    'PPO_equal_r21': True, 'PD_equal_r21': True}))
