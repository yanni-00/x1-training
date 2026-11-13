"""Bounded stage-one clean-stop variant, preserving the owner's PD and route."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    physical = assert_training_admission()
    if physical.get('kind') != 'owner_baseline' or physical['delivery']['enabled']:
        raise ValueError('This entry requires the approved no-filter owner baseline')
    import isaacgym  # noqa: F401 -- must precede torch
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_50hz_config import F1Stage1NoFilter50HzCfg, F1Stage1NoFilter50HzCfgPPO
    from humanoid.envs.f1.f1_clean_stop_env import F1CleanStopEnv
    from humanoid.utils import get_args

    args = get_args()
    if args.task != 'f1_walk_stage1':
        raise ValueError('Clean-stop variant is restricted to F1 stage one')
    cfg = F1Stage1NoFilter50HzCfg()
    cfg.env.command_semantics = 'explicit_scheduled_stand_zero_twist_v1'
    cfg.env.goal_revision = 'F1-WALK-v2'
    task_registry.register(args.task, F1CleanStopEnv, cfg, F1Stage1NoFilter50HzCfgPPO())
    env, effective = task_registry.make_env(name=args.task, args=args)
    if effective.domain_rand.enable_delivery or effective.sim.dt*effective.control.decimation != 1/50:
        raise ValueError('Effective profile differs from the fixed 50 Hz no-filter route')
    runner, training_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(num_learning_iterations=training_cfg.runner.max_iterations,
                 init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
