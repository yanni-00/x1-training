"""Train the TASK054-derived 100 Hz no-filter F1 stage-one comparison."""

from math import isclose
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    physical = assert_training_admission()
    if physical.get('profile_file') != 'configs/stage1_100hz_no_filter.json' or physical['delivery']['enabled']:
        raise ValueError('Requires the dedicated 100 Hz no-filter owner baseline')

    import isaacgym  # noqa: F401
    from humanoid.algo.ppo.actor_critic_f1_equivariant import register_policy_class
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_target_speed_ramp_100hz import (
        POLICY_DT, F1TargetSpeedRamp100HzCfg, F1TargetSpeedRamp100HzCfgPPO,
        F1TargetSpeedRamp100HzEnv,
    )
    from humanoid.utils import get_args

    args = get_args()
    if args.task != 'f1_walk_stage1':
        raise ValueError('This experiment uses the F1 stage-one task')
    if args.max_iterations is None or args.max_iterations <= 0:
        raise ValueError('Pass the registered budget with --max_iterations')

    register_policy_class()
    task_registry.register(args.task, F1TargetSpeedRamp100HzEnv,
                           F1TargetSpeedRamp100HzCfg(), F1TargetSpeedRamp100HzCfgPPO())
    env, cfg = task_registry.make_env(name=args.task, args=args)
    if not isclose(env.dt, POLICY_DT, rel_tol=1e-7) or cfg.domain_rand.enable_delivery:
        raise ValueError('Effective simulation differs from the 100 Hz no-filter profile')

    runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    print(f'[F1 timing] dt={env.dt:g} s, history stride={cfg.env.observation_history_stride}, '
          f'rollout steps={runner.num_steps_per_env}', flush=True)
    print('[F1 command] ' + cfg.env.command_semantics, flush=True)
    print('[F1 reward] ' + cfg.env.reward_semantics, flush=True)
    print('[F1 policy] ' + train_cfg.runner.policy_class_name, flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
