"""Stage-one directional-posture reward probe; fixed owner PD and clean stops."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    physical = assert_training_admission()
    if physical.get('kind') != 'owner_baseline' or physical['delivery']['enabled']:
        raise ValueError('This entry requires the fixed no-filter owner baseline')
    import isaacgym  # noqa: F401
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_directional_posture import F1DirectionalPostureCfg, F1DirectionalPostureEnv
    from humanoid.envs.f1.f1_50hz_config import F1Stage1NoFilter50HzCfgPPO
    from humanoid.utils import get_args
    args = get_args()
    if args.task != 'f1_walk_stage1':
        raise ValueError('Command-coverage variant is restricted to F1 stage one')
    task_registry.register(args.task, F1DirectionalPostureEnv, F1DirectionalPostureCfg(),
                           F1Stage1NoFilter50HzCfgPPO())
    env, cfg = task_registry.make_env(name=args.task, args=args)
    if cfg.domain_rand.enable_delivery or cfg.sim.dt*cfg.control.decimation != 1/50:
        raise ValueError('Effective control differs from fixed 50Hz/no-filter')
    runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    print('[F1 command modes] '+str(cfg.commands.gait), flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
