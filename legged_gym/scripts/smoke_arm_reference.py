"""Check arm-reference semantics and real PPO continuation before GM launch."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401, before torch
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


args = get_args()
cfg, train_cfg = task_registry.get_cfgs(args.task)
base, base_train = task_registry.get_cfgs('g1r22')
before, after = class_to_dict(base), class_to_dict(cfg)
assert set(after['rewards']) - set(before['rewards']) == {'arm_rest_elbow_rad'}
del after['rewards']['arm_rest_elbow_rad']
assert before == after, 'Unexpected environment configuration change'
for group in ('algorithm', 'policy'):
    assert class_to_dict(getattr(train_cfg, group)) == class_to_dict(getattr(base_train, group))
checkpoint = Path(train_cfg.runner.resume_path)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '8a8cf0d76040bddb92e5235e1f4707beefdafcae2d10e968b8530bf8a2a6a30f'
cfg.env.num_envs = 16
env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
saved_q = env.dof_pos.clone()
saved_standing = env.standing_envs_mask.clone()
saved_interrupt = env.interrupt_mask.clone()
target = cfg.rewards.arm_rest_elbow_rad
env.interrupt_mask[:] = False
env.standing_envs_mask[:] = True
env.dof_pos[:] = env.default_dof_pos
assert torch.allclose(env._reward_shoulder_deviation(), torch.full_like(env.rew_buf, 0.5 * target ** 2))
assert torch.allclose(env._reward_standing_joint_deviation(), torch.full_like(env.rew_buf, 2 * target ** 2))
env.dof_pos[:, env.arm_rest_elbow_inds] += target
assert torch.allclose(env._reward_shoulder_deviation(), torch.zeros_like(env.rew_buf), atol=1e-7)
assert torch.allclose(env._reward_standing_joint_deviation(), torch.zeros_like(env.rew_buf), atol=1e-7)
env.dof_pos[:] = env.default_dof_pos
env.interrupt_mask[:] = True
assert not env._reward_shoulder_deviation().any()
assert not env._reward_standing_joint_deviation().any()
env.dof_pos[:] = saved_q
env.standing_envs_mask[:] = saved_standing
env.interrupt_mask[:] = saved_interrupt
runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg,
    log_root=str(Path('resources/smoke_arm_reference').resolve()))
initial = runner.current_learning_iteration
assert initial == 9000
runner.learn(num_learning_iterations=10, init_at_random_ep_len=True)
assert runner.current_learning_iteration == initial + 10
assert all(torch.isfinite(value).all() for value in runner.alg.actor_critic.parameters())
assert torch.isfinite(env.rew_buf).all()
print('SMOKE_ARM_REFERENCE_OK ' + json.dumps({'task': args.task,
    'arm_rest_elbow_rad': target, 'resume_iter': initial,
    'final_iter': runner.current_learning_iteration,
    'reward_names_and_scales_equal_r22': True,
    'other_env_config_equal_r22': True, 'PPO_and_policy_equal_r22': True,
    'reference_and_intervention_mask_checks': True}))
