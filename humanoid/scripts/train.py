"""Native training entry; refuses scaffold-only/pending state before importing Isaac Gym."""
from humanoid.f1_contract import assert_training_admission


def main():
    assert_training_admission()
    # Isaac Gym must be imported before torch in its supported runtime.
    import isaacgym  # noqa: F401
    from humanoid.envs import task_registry
    from humanoid.utils import get_args

    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, config, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(num_learning_iterations=config.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'F1 training blocked: {error}')
