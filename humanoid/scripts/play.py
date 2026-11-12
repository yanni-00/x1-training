"""Replay an explicit, provenance-checked F1 checkpoint; no latest/X1 fallback."""
from humanoid.f1_contract import assert_run_admission, verify_checkpoint


def main():
    assert_run_admission('evaluation')
    import isaacgym  # noqa: F401
    import torch
    from humanoid.envs import task_registry
    from humanoid.utils import get_args

    args = get_args()
    if not args.checkpoint_path:
        raise ValueError('Replay requires --checkpoint_path for a checkpoint produced by this F1 loop')
    path, _ = verify_checkpoint(args.checkpoint_path, required_task=args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.execution_mode = 'evaluation'
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    # Identified actuator parameters use their nominal values during play.
    env_cfg.env.nominal_identification = True
    args.num_envs = 1
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                train_cfg=train_cfg, log_root=None)
    runner.load(str(path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            obs, _, _, _, _ = env.step(policy(obs))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'F1 replay blocked: {error}')
