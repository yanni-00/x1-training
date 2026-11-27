"""r25 posture-first: continue F1 50 Hz stage one from TASK054 model_1000 (upright, -1.92 deg) with the r21 late-swing direction recipe (w60), single variable = parent timing."""

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
from humanoid.f1_contract import assert_training_admission, verify_checkpoint, write_checkpoint_metadata


PARENT_TASK = 'TASK_20261121_054'
PARENT_ITERATION = 1000
PARENT_SHA256 = '9da6579f0f0c105894d2ba667d7684f15328ca39203b2bd0f9a0458aa952f4a6'
# The size and hash bind the TASK054 model_1000 checkpoint (upright posture-first base).
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
                              physical_model=physical, goal_revision='F1-WALK-v3')
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
        raise ValueError('Requires the owner no-filter simulation baseline')
    if resume.parent_iteration != PARENT_ITERATION:
        raise ValueError('This continuation requires the registered TASK054 model_1000 checkpoint')
    with ExitStack() as resources:
        parent_path = resume.parent_checkpoint
        if resume.parent_download_url:
            download_dir = Path(resources.enter_context(TemporaryDirectory(prefix='f1-parent-')))
            parent_path = download_parent(resume.parent_download_url, download_dir / 'model_1000.pt',
                                          physical)
        parent, metadata = verify_checkpoint(parent_path, required_task='f1_walk_stage1',
                                             required_iteration=PARENT_ITERATION)
        if metadata['checkpoint_sha256'] != PARENT_SHA256:
            raise ValueError('Local checkpoint differs from the registered TASK054 parent')
        train(resume.log_root, parent, metadata)


def train(log_root, parent, metadata):
    import isaacgym  # noqa: F401
    from humanoid.algo.ppo.actor_critic_f1_equivariant import register_policy_class
    from humanoid.envs import task_registry
    from humanoid.envs.f1.f1_equivariant_config import F1EquivariantCfgPPO
    from humanoid.envs.f1.f1_lateral_swing_direction_50hz import (
        F1LateralSwingDirection50HzCfg, F1LateralSwingDirection50HzEnv)
    from humanoid.utils import get_args

    args = get_args()
    if args.task != 'f1_walk_stage1' or args.resume or args.checkpoint_path:
        raise ValueError('Use the explicit parent arguments for F1 stage-one warm start')
    if args.max_iterations is None or args.max_iterations <= 0:
        raise ValueError('Specify a positive number of additional updates')
    register_policy_class()
    task_registry.register(args.task, F1LateralSwingDirection50HzEnv,
                           F1LateralSwingDirection50HzCfg(), F1EquivariantCfgPPO())
    env, cfg = task_registry.make_env(name=args.task, args=args)
    if cfg.domain_rand.enable_delivery or cfg.sim.dt * cfg.control.decimation != 1 / 50:
        raise ValueError('Effective controls differ from the 50 Hz no-filter baseline')
    effective = float(cfg.rewards.scales.lateral_swing_direction) * cfg.sim.dt * cfg.control.decimation
    if abs(effective - 60.0 * 0.02) > 1e-9:
        raise ValueError('Effective lateral_swing_direction coefficient differs from 60*dt')

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