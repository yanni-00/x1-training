"""Train the identified F1 walking strategy from random initialization."""

from math import isclose
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    physical = assert_training_admission()
    if physical.get('profile') != 'stage1_100hz_identified':
        raise ValueError('Select the identified 100 Hz launch configuration')

    import isaacgym  # noqa: F401
    from humanoid.envs.f1.f1_identified_100hz import POLICY_DT, register_identified_task
    from humanoid.utils import get_args

    args = get_args()
    train = physical['training']
    if args.task != train['task'] or args.resume or args.checkpoint_path:
        raise ValueError('This task starts from random initialization')
    for key in ('num_envs', 'seed', 'max_iterations'):
        override = getattr(args, key)
        if override is not None and override != train[key]:
            raise ValueError(f'--{key} must match the identified profile ({train[key]})')

    registry = register_identified_task(physical)
    env, cfg = registry.make_env(name=args.task, args=args)
    dr = cfg.domain_rand
    if (not isclose(env.dt, POLICY_DT, rel_tol=1e-7)
            or not (dr.add_lag and dr.add_dof_lag and dr.enable_delivery)
            or dr.delivery_tau_d != physical['delivery']['tau_s']):
        raise ValueError('Effective timing or actuator dynamics differ from the identified profile')

    runner, train_cfg, _ = registry.make_alg_runner(env=env, name=args.task, args=args)
    print(f'[F1 timing] policy_dt={env.dt:g} s, history stride={cfg.env.observation_history_stride}, '
          f'rollout steps={runner.num_steps_per_env}', flush=True)
    print(f'[F1 dynamics] action lag={dr.lag_timesteps_range}, feedback lag={dr.dof_lag_timesteps_range} '
          f'(physics steps), ankle torque tau={dr.delivery_tau_d:g} s', flush=True)
    print(f'[F1 profile] sha256={cfg.identification_sha256}, policy={train_cfg.runner.policy_class_name}, '
          f'updates={train_cfg.runner.max_iterations}, init=from_scratch', flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
