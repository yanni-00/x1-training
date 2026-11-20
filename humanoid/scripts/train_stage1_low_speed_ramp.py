"""Stage-one forward low-speed reward candidate with fixed owner controls."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission


def main():
    physical = assert_training_admission()
    if physical.get('kind') != 'owner_baseline' or physical['delivery']['enabled']:
        raise ValueError('Requires the fixed no-filter owner baseline')
    import isaacgym  # noqa: F401
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_low_speed_ramp import F1LowSpeedRampCfg, F1LowSpeedRampEnv
    from humanoid.envs.f1.f1_equivariant_config import F1EquivariantCfgPPO
    from humanoid.algo.ppo.actor_critic_f1_equivariant import register_policy_class
    from humanoid.utils import get_args
    args = get_args()
    if args.task != 'f1_walk_stage1':
        raise ValueError('This forward low-speed reward candidate is restricted to F1 stage one')
    register_policy_class()
    task_registry.register(args.task, F1LowSpeedRampEnv, F1LowSpeedRampCfg(), F1EquivariantCfgPPO())
    env, cfg = task_registry.make_env(name=args.task, args=args)
    if cfg.domain_rand.enable_delivery or cfg.sim.dt*cfg.control.decimation != 1/50:
        raise ValueError('Effective controls differ from fixed50Hz/no-filter')
    runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    print('[F1 command] '+cfg.env.command_semantics, flush=True)
    print('[F1 reward] '+cfg.env.reward_semantics, flush=True)
    print('[F1 policy] '+train_cfg.runner.policy_class_name, flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()
