"""Check G1 translation invariants, reward parity and a short PPO update."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


def main():
    args = get_args()
    cfg, train_cfg = task_registry.get_cfgs('g1r21')
    baseline, _ = task_registry.get_cfgs('g1r20')
    assert class_to_dict(cfg.rewards.scales) == class_to_dict(baseline.rewards.scales)
    cfg.env.num_envs = 16  # existing r20 smoke size
    env, _ = task_registry.make_env('g1r21', args=args, env_cfg=cfg)
    obs, critic_obs = env.reset()
    names = env.dof_names
    assert set(env.reward_arm_inds).isdisjoint(env.reward_lower_inds)
    assert set(env.reward_arm_inds + env.reward_lower_inds) == set(range(env.num_dof))
    assert [names[i] for i in env.reward_knee_inds] == ['left_knee_joint', 'right_knee_joint']
    assert [names[i] for i in env.waist_yaw_inds] == ['waist_yaw_joint']
    assert all('shoulder_' in names[i] or 'elbow' in names[i] for i in env.reward_arm_inds)
    assert torch.all(env.commands[:, 7:10] == 0)
    assert torch.all(env.commands[:, 4] == 0.5)
    assert not env.use_disturb

    # A symmetric pair of ankle impulses must have equal lower-body costs and
    # zero upper-body costs. The old H1 [:11] split violates this invariant.
    saved = {name: getattr(env, name).clone() for name in (
        'actions', 'last_actions', 'last_last_actions', 'dof_pos', 'interrupt_mask')}
    env.actions.zero_(); env.last_actions.zero_(); env.last_last_actions.zero_()
    env.interrupt_mask.zero_()
    costs = []
    for name in ('left_ankle_roll_joint', 'right_ankle_roll_joint'):
        env.actions.zero_()
        env.actions[:, names.index(name)] = 1
        lower = env._reward_action_rate_lower()
        upper = env._reward_action_rate_upper()
        assert torch.all(upper == 0)
        costs.append(lower.clone())
    assert torch.equal(costs[0], costs[1]) and torch.all(costs[0] > 0)
    # The waist command tracks the URDF yaw axis, independent of roll.
    env.dof_pos.zero_()
    env.dof_pos[:, env.waist_yaw_inds] = env.commands[:, 9:10]
    env.dof_pos[:, names.index('waist_roll_joint')] = 1
    assert torch.all(env._reward_waist_control() == 0)
    for name, value in saved.items():
        getattr(env, name).copy_(value)

    runner, _ = task_registry.make_alg_runner(
        env, args=args, train_cfg=train_cfg,
        log_root=str(Path('resources/smoke_r21').resolve()))
    assert runner.current_learning_iteration == 7800
    initial_iteration = runner.current_learning_iteration
    runner.learn(num_learning_iterations=10, init_at_random_ep_len=True)
    assert runner.current_learning_iteration == initial_iteration + 10
    assert all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters())
    print('SMOKE_R21_OK ' + json.dumps({
        'resume_iter': initial_iteration, 'final_iter': runner.current_learning_iteration,
        'reward_names': env.reward_names,
        'reward_scales_equal_r20': True,
        'arm_names': [names[i] for i in env.reward_arm_inds],
        'lower_names': [names[i] for i in env.reward_lower_inds],
        'knees': [names[i] for i in env.reward_knee_inds],
        'waist_command_joint': [names[i] for i in env.waist_yaw_inds],
        'commands': env.commands[0].tolist()}))


if __name__ == '__main__':
    main()
