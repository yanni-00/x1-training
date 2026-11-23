"""Continue the 100 Hz stage-one comparison from the formal3000 checkpoint under the stop-settle sampler."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import (
    assert_training_admission,
    verify_checkpoint,
    write_checkpoint_metadata,
)


PARENT_TASK = 'TASK_20261123_019'
PARENT_ITERATION = 3000
PARENT_SHA256 = '6d3eaf172e366ccbe8f59484393936c99c64febcd49fe749b21cd91fb2550d13'
# The size and hash bind the platform-delivered formal3000 checkpoint.
PARENT_SIZE = 10_324_999


def download_parent(url, path, physical):
    target = urlsplit(url)
    if target.scheme != 'https' or not target.hostname or target.username or target.password:
        raise ValueError('GM parent source requires an HTTPS checkpoint delivery URL')
    try:
        subprocess.run(
            ['curl', '--silent', '--fail', '--location', '--proto', '=https',
             '--proto-redir', '=https', '--max-time', '60', '--max-filesize',
             str(PARENT_SIZE), '--output', str(path), '--config', '-'],
            input=f'url = {json.dumps(url)}\n', text=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError('Could not download the GM parent checkpoint') from None
    if path.stat().st_size != PARENT_SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError('GM parent download does not match the registered checkpoint')
    # GM delivers saved weights only. Recreate contract metadata after matching
    # the exact platform object to the verified local parent.
    write_checkpoint_metadata(path, 'f1_walk_stage1', PARENT_ITERATION,
                              physical_model=physical, goal_revision='F1-WALK-100HZ-v1')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parent_source = parser.add_mutually_exclusive_group(required=True)
    parent_source.add_argument('--parent-checkpoint', type=Path)
    parent_source.add_argument('--parent-download-url')
    parser.add_argument('--parent-iteration', required=True, type=int)
    parser.add_argument('--log-root', default='default')
    resume, native_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *native_args]

    physical = assert_training_admission()
    if physical.get('kind') != 'owner_baseline' or physical['delivery']['enabled']:
        raise ValueError('Requires the dedicated 100 Hz no-filter owner baseline')
    if physical.get('profile_file') != 'configs/stage1_100hz_no_filter.json':
        raise ValueError('Requires the dedicated 100 Hz no-filter runtime profile')
    if resume.parent_iteration != PARENT_ITERATION:
        raise ValueError('This continuation requires the registered formal3000 checkpoint')
    with ExitStack() as resources:
        parent_path = resume.parent_checkpoint
        if resume.parent_download_url:
            download_dir = Path(resources.enter_context(TemporaryDirectory(prefix='f1-parent-')))
            parent_path = download_parent(resume.parent_download_url, download_dir / 'model_3000.pt',
                                          physical)
        parent, metadata = verify_checkpoint(parent_path, required_task='f1_walk_stage1',
                                             required_iteration=PARENT_ITERATION)
        if metadata['checkpoint_sha256'] != PARENT_SHA256:
            raise ValueError('Local checkpoint differs from the registered GM parent')
        train(resume.log_root, parent, metadata)


def train(log_root, parent, metadata):
    import isaacgym  # noqa: F401
    from humanoid.algo.ppo.actor_critic_f1_equivariant import register_policy_class
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_target_speed_ramp_100hz import (
        POLICY_DT, F1TargetSpeedRamp100HzCfg, F1TargetSpeedRamp100HzCfgPPO,
        F1TargetSpeedRamp100HzEnv,
    )
    from humanoid.utils import get_args

    args = get_args()
    if args.task != 'f1_walk_stage1' or args.resume or args.checkpoint_path:
        raise ValueError('Use the explicit parent arguments for F1 stage-one warm start')
    if args.max_iterations is None or args.max_iterations <= 0:
        raise ValueError('Specify a positive number of additional updates')
    register_policy_class()
    task_registry.register(args.task, F1TargetSpeedRamp100HzEnv,
                           F1TargetSpeedRamp100HzCfg(), F1TargetSpeedRamp100HzCfgPPO())
    env, cfg = task_registry.make_env(name=args.task, args=args)
    if not abs(env.dt - POLICY_DT) <= 1e-9 or cfg.domain_rand.enable_delivery:
        raise ValueError('Effective simulation differs from the 100 Hz no-filter profile')

    runner, train_cfg, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=log_root)
    runner.load(str(parent), load_optimizer=False)
    if runner.current_learning_iteration != PARENT_ITERATION:
        raise ValueError('Parent update counter differs from the requested checkpoint')

    print('[F1 command] ' + cfg.env.command_semantics, flush=True)
    print('[F1 reward] ' + cfg.env.reward_semantics, flush=True)
    print('[F1 policy] ' + train_cfg.runner.policy_class_name, flush=True)
    print(f'[F1 parent] {PARENT_TASK} {metadata["checkpoint_sha256"]} at {PARENT_ITERATION}, fresh optimizer',
          flush=True)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    main()