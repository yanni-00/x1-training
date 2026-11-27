"""Warm-start the identified F1 walking strategy from the r11 checkpoint with a delay curriculum.

Loads the parent Actor/Critic weights only, rebuilds both optimizers, restarts the
update counter at zero, then trains four 1500-update segments that step the pure
delay ranges up to the identified envelope:
  s1 action [5, 5] ms  feedback [0, 5] ms
  s2 action [5, 15] ms feedback [0, 15] ms
  s3 action [5, 30] ms feedback [0, 30] ms
  s4 action [5, 40] ms feedback [0, 40] ms
Model and optimizer state carry across segments (learn() is called per segment).
Pass --resume-checkpoint to continue an interrupted run from an own model_N.pt:
the model and both optimizer states are restored and the schedule resumes at the
segment that covers the saved iteration.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.f1_contract import assert_training_admission

TRAINING_ROOT = Path(__file__).resolve().parents[2]
PARENT_FILE = TRAINING_ROOT / 'configs/warm_start_parent.json'
SEGMENT_ITERATIONS = 1500
SEGMENTS = (
    {'name': 's1', 'action_ms': (5, 5), 'feedback_ms': (0, 5)},
    {'name': 's2', 'action_ms': (5, 15), 'feedback_ms': (0, 15)},
    {'name': 's3', 'action_ms': (5, 30), 'feedback_ms': (0, 30)},
    {'name': 's4', 'action_ms': (5, 40), 'feedback_ms': (0, 40)},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_parent(checkpoint, record):
    path = Path(checkpoint).expanduser().resolve(strict=True)
    digest = sha256(path)
    if digest != record['sha256']:
        raise ValueError('Parent checkpoint SHA differs from the registered r11 model')
    sidecar = Path(str(path) + '.f1.json')
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    if meta.get('checkpoint_sha256') != digest:
        raise ValueError('Parent sidecar checkpoint hash mismatch')
    for key, value in record['sidecar'].items():
        if meta.get(key) != value:
            raise ValueError('Parent sidecar mismatch: ' + key)
    return path, meta


def set_delay_ranges(env, action_ms, feedback_ms):
    dr = env.cfg.domain_rand
    dr.add_lag = True
    dr.add_dof_lag = True
    dr.randomize_lag_timesteps = True
    dr.randomize_dof_lag_timesteps = True
    dr.randomize_lag_timesteps_perstep = False
    dr.randomize_dof_lag_timesteps_perstep = False
    dr.lag_timesteps_range = [int(action_ms[0]), int(action_ms[1])]
    dr.dof_lag_timesteps_range = [int(feedback_ms[0]), int(feedback_ms[1])]
    import torch
    n = env.num_envs
    env.lag_buffer = torch.zeros(n, env.num_actions, int(action_ms[1]) + 1, device=env.device)
    env.lag_timestep = torch.randint(int(action_ms[0]), int(action_ms[1]) + 1, (n,), device=env.device)
    env.dof_lag_buffer = torch.zeros(n, env.num_actions * 2, int(feedback_ms[1]) + 1, device=env.device)
    env.dof_lag_timestep = torch.randint(int(feedback_ms[0]), int(feedback_ms[1]) + 1, (n,), device=env.device)
    # A zeroed feedback buffer makes the policy read fictitious joint positions until
    # the history fills (measured to roughly halve start-up survival). Fill the new
    # buffer with the real current q/dq so the first inference of the segment sees
    # valid observations.
    with torch.no_grad():
        current = torch.cat((env.dof_pos, env.dof_vel), dim=1).unsqueeze(-1)
        env.dof_lag_buffer[:] = current.repeat(1, 1, env.dof_lag_buffer.shape[-1])
        env.compute_observations()
        gap = (env.lagged_dof_pos - env.dof_pos).abs().max().item()
    print(f'[F1 warmstart] feedback buffer filled from current state, '
          f'max|lagged_dof_pos - dof_pos| = {gap:.3e}', flush=True)
    if gap > 1e-4:
        raise ValueError('Segment feedback buffer does not match the current joint state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', default=str(TRAINING_ROOT / 'assets/warm_start/model_6000.pt'),
                        help='r11 model_6000.pt with its .f1.json sidecar '
                             '(defaults to the bundled assets/warm_start copy)')
    parser.add_argument('--smoke-iterations', type=int, default=0,
                        help='verification only: updates per segment instead of 1500')
    parser.add_argument('--smoke-envs', type=int, default=0,
                        help='verification only: smaller num_envs override')
    parser.add_argument('--resume-checkpoint', default=None,
                        help='continue from an own identified model_N.pt (+ .f1.json) with model and optimizer state')
    options, native_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + native_args

    physical = assert_training_admission()
    if physical.get('profile') != 'stage1_100hz_identified':
        raise ValueError('Select the identified 100 Hz launch configuration')

    import isaacgym  # noqa: F401
    import torch
    from humanoid.envs.f1.f1_identified_100hz import POLICY_DT, register_identified_task
    from humanoid.f1_contract import verify_checkpoint
    from humanoid.utils import get_args

    args = get_args()
    train = physical['training']
    if args.task != train['task'] or args.resume or args.checkpoint_path:
        raise ValueError('This task warm-starts from the registered r11 checkpoint only')
    for key in ('num_envs', 'seed'):
        override = getattr(args, key)
        if override is not None and override != train[key]:
            if not (options.smoke_envs and key == 'num_envs' and override == options.smoke_envs):
                raise ValueError(f'--{key} must match the identified profile ({train[key]})')

    registry = register_identified_task(physical)
    env, cfg = registry.make_env(name=args.task, args=args)
    dr = cfg.domain_rand
    from math import isclose
    if (not isclose(env.dt, POLICY_DT, rel_tol=1e-7)
            or not dr.enable_delivery
            or dr.delivery_tau_d != physical['delivery']['tau_s']):
        raise ValueError('Effective timing or actuator dynamics differ from the identified profile')

    runner, train_cfg, _ = registry.make_alg_runner(env=env, name=args.task, args=args)
    if options.resume_checkpoint:
        resume_path = Path(options.resume_checkpoint).expanduser().resolve(strict=True)
        resume_path, resume_meta = verify_checkpoint(resume_path, required_task=train['task'])
        runner.load(str(resume_path), load_optimizer=True)
        start_iteration = runner.current_learning_iteration
        completed_segments, within = divmod(start_iteration, SEGMENT_ITERATIONS)
        if completed_segments >= len(SEGMENTS):
            print(f'[F1 resume] checkpoint iteration {start_iteration} already covers all segments', flush=True)
            return
        runner.parent_checkpoint = {'path': str(resume_path), 'sha256': resume_meta['checkpoint_sha256'],
                                    'task': resume_meta['task'], 'iteration': resume_meta['iteration']}
        print(f'[F1 resume] checkpoint verified iteration={start_iteration} '
              f'segment=s{completed_segments + 1} within={within} sha256={resume_meta["checkpoint_sha256"]}',
              flush=True)
        schedule = [(s, SEGMENT_ITERATIONS - within if i == completed_segments else SEGMENT_ITERATIONS)
                    for i, s in enumerate(SEGMENTS) if i >= completed_segments]
    else:
        record = json.loads(PARENT_FILE.read_text(encoding='utf-8'))
        parent_path, parent_meta = verify_parent(options.parent_checkpoint, record)
        saved = torch.load(str(parent_path), map_location='cpu')
        if saved.get('iter') != record['iteration']:
            raise ValueError('Parent tensor iteration differs from the registered r11 model')
        policy = runner.alg.actor_critic
        policy.load_state_dict(saved['model_state_dict'], strict=True)
        learning_rate = runner.alg.optimizer.param_groups[0]['lr']
        runner.alg.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        runner.alg.state_estimator_optimizer = torch.optim.Adam(
            policy.state_estimator.parameters(), lr=learning_rate)
        runner.current_learning_iteration = 0
        runner.it = 0
        runner.parent_checkpoint = {'path': str(parent_path), 'sha256': record['sha256'],
                                    'task': record['task'], 'iteration': record['iteration']}
        print(f'[F1 warmstart] parent verified sha256={record["sha256"]} '
              f'task={record["source_task_id"]} iteration={record["iteration"]}', flush=True)
        schedule = [(s, SEGMENT_ITERATIONS) for s in SEGMENTS]

    for segment, iterations in schedule:
        iterations = options.smoke_iterations or iterations
        set_delay_ranges(env, segment['action_ms'], segment['feedback_ms'])
        print(f'[F1 warmstart] {segment["name"]} action={segment["action_ms"]} ms '
              f'feedback={segment["feedback_ms"]} ms iterations={iterations}', flush=True)
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
    print('[F1 warmstart] done', flush=True)


if __name__ == '__main__':
    main()