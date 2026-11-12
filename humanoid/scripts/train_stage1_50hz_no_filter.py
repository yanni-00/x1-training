"""Single-profile native entry; all existing F1 admission gates remain in force."""
from pathlib import Path
import sys

# Select this exact source tree before an image's possibly installed X1 package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    identified = assert_training_admission()
    if identified.get('kind') != 'owner_baseline' or identified['delivery']['enabled']:
        raise ValueError('50 Hz no-filter profile requires delivery LPF disabled')

    # Preserve Isaac Gym's import-before-torch requirement.
    import isaacgym  # noqa: F401
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_50hz_config import (
        F1Stage1NoFilter50HzCfg, F1Stage1NoFilter50HzCfgPPO,
    )
    from humanoid.utils import get_args

    args = get_args()
    if args.task != 'f1_walk_stage1':
        raise ValueError('This entry is restricted to F1 stage one')
    task_registry.register(
        args.task, task_registry.get_task_class(args.task),
        F1Stage1NoFilter50HzCfg(), F1Stage1NoFilter50HzCfgPPO(),
    )
    env, effective_env_cfg = task_registry.make_env(name=args.task, args=args)
    if effective_env_cfg.domain_rand.enable_delivery:
        raise ValueError('Runtime identification enabled a filter in the no-filter profile')
    if effective_env_cfg.sim.dt * effective_env_cfg.control.decimation != 1 / 50:
        raise ValueError('Effective policy period differs from owner-selected 50 Hz')
    runner, config, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(num_learning_iterations=config.runner.max_iterations,
                 init_at_random_ep_len=False)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'F1 stage-one training blocked: {error}')
