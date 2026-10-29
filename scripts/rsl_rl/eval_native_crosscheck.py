"""[OMA] Fixed-weight Lab/MuJoCo dynamics diagnosis; no PPO or optimizer.

Uses the existing GM Lab image and the same frozen continuous protocols.
Raw Gym observations avoid depending on RSL-RL wrapper-version conventions.
The diagnostic PT bundle contains CSV/JSON, never replacement policy weights.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint_file', required=True)
parser.add_argument('--checkpoint_sha256', required=True)
parser.add_argument('--motion_file', required=True)
parser.add_argument('--protocol_manifest', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--bundle_file', required=True)
parser.add_argument('--task', default='Tracking-WalkStop-G1-v0')
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.runtime_policy import RuntimeActor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(value):
    return value.detach().cpu().reshape(-1).tolist()


def tilt(quaternion):
    return math.acos(float((1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)).clamp(-1, 1)))


def load_scorer():
    spec = importlib.util.spec_from_file_location('score_walk_stop', ROOT / 'scripts/score_walk_stop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    checkpoint_path = Path(args.checkpoint_file).resolve()
    if sha(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError('Fixed checkpoint checksum mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    runtime = checkpoint['infos']['oma_runtime']
    with np.load(args.motion_file, allow_pickle=False) as source:
        for key, value in runtime['motion'].items():
            if not np.array_equal(source[key], value.cpu().numpy()):
                raise RuntimeError('Native reference tensor mismatch: ' + key)
    registry_path = Path(args.protocol_manifest).resolve()
    registry = json.loads(registry_path.read_text())
    cases = []
    for entry in registry['protocols']:
        path = registry_path.parent / entry['file']
        if sha(path) != entry['sha256']:
            raise RuntimeError('Frozen protocol changed: ' + entry['file'])
        protocol = json.loads(path.read_text())
        if not protocol.get('nominal'):
            raise ValueError('This dynamics crosscheck requires nominal deterministic protocols')
        cases.append((entry['case'], protocol))
    dt = env_cfg.sim.dt * env_cfg.decimation
    if dt != runtime['policy_dt']:
        raise RuntimeError('Native control timestep differs from trained runtime')
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(cases[0][1]['seed'])
    if args.device is not None:
        env_cfg.sim.device = args.device
    maximum_steps = max(sum(round(s['duration_s'] / dt) for s in p['stages']) for _, p in cases)
    env_cfg.episode_length_s = (maximum_steps + 2) * dt
    env_cfg.commands.motion.motion_file = str(Path(args.motion_file).resolve())
    env_cfg.commands.motion.evaluation_start_support = cases[0][1]['initial_support_index']
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0., 0.)
    env_cfg.observations.policy.enable_corruption = False
    for name in ['physics_material', 'add_joint_default_pos', 'base_com', 'push_robot']:
        setattr(env_cfg.events, name, None)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = gym.make(args.task, cfg=env_cfg)
    base = raw.unwrapped
    motion = base.command_manager.get_term('motion')
    robot = motion.robot
    if list(robot.joint_names) != runtime['joint_names'] or list(robot.body_names) != runtime['body_names']:
        raise RuntimeError('Native articulation order differs from exact trained reference')
    actor = RuntimeActor(checkpoint).to(base.device).eval()
    initial_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sensor = base.scene['contact_forces']
    foot_names = ['left_ankle_roll_link', 'right_ankle_roll_link']
    foot_ids = robot.find_bodies(foot_names, preserve_order=True)[0]
    sensor_ids = sensor.find_bodies(foot_names, preserve_order=True)[0]
    pelvis_index = motion.cfg.body_names.index('pelvis')
    header = ['step', 'time_s', 'stage', 'requested_go', 'reference_phase', 'reference_rate',
              'reference_exhausted', 'terminated', 'root_x', 'root_y', 'root_z', 'root_vx',
              'root_vy', 'root_vz', 'root_tilt_rad', 'reference_vx', 'reference_vy', 'reference_vz']
    for side in ['left', 'right']:
        header += [f'{side}_foot_{v}' for v in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']]
    for prefix in ['q', 'dq', 'q_ref', 'dq_ref', 'action']:
        header += [f'{prefix}__{name}' for name in robot.joint_names]
    header += ['anchor_position_error_m', 'body_velocity_rmse_mps', 'body_velocity_reward_unweighted']
    gates = json.loads((ROOT / 'configs/behavior_gates.json').read_text())
    score = load_scorer()
    results = []
    try:
        for case_name, protocol in cases:
            target = out / case_name
            target.mkdir(parents=True, exist_ok=True)
            if (target / 'joints.csv').exists():
                raise RuntimeError('Existing native evaluation; inspect before repeating ' + case_name)
            motion.cfg.evaluation_start_support = protocol['initial_support_index']
            motion.request(False)
            observations, _ = raw.reset(seed=int(protocol['seed']))
            # Native relative-body targets are filled by _update_command, but
            # Lab checks terminations before the first regular command update.
            # Initialise derived targets at static STOP without physics, clock
            # advancement, robot-state writes or an unrecorded warm-up step.
            initial_phase = motion.clock.phase.clone()
            initial_rate = motion.clock.rate.clone()
            initial_root = robot.data.root_state_w.clone()
            initial_joints = robot.data.joint_pos.clone()
            motion._update_command()
            if not (torch.equal(initial_phase, motion.clock.phase) and torch.equal(initial_rate, motion.clock.rate)
                    and torch.equal(initial_root, robot.data.root_state_w)
                    and torch.equal(initial_joints, robot.data.joint_pos)):
                raise RuntimeError('Derived-target initialization changed the physical state or reference clock')
            observations = base.observation_manager.compute(update_history=False)
            obs = observations['policy']
            if tuple(obs.shape) != (1, 160):
                raise ValueError('Invalid native policy observation shape')
            planned = sum(round(s['duration_s'] / dt) for s in protocol['stages'])
            manifest = {'checkpoint_sha256': args.checkpoint_sha256, 'saved_iteration': checkpoint['iter'],
                        'reference_sha256': runtime['reference_sha256'], 'protocol': protocol,
                        'policy_dt': dt, 'physics_dt': float(base.cfg.sim.dt), 'planned_steps': planned,
                        'observation_dim': 160, 'joint_names': list(robot.joint_names),
                        'body_names': list(robot.body_names), 'state_source': 'Isaac Lab actual state',
                        'contact_force_threshold_source': float(sensor.cfg.force_threshold),
                        'native_termination_terms': list(base.termination_manager.active_terms),
                        'derived_target_initialization': 'Static STOP update; clock and physical state equality asserted; no warm-up physics step',
                        'automatic_reset_policy': 'Reset once per independent protocol, abort at first returned done; CSV is PRE-step.',
                        'video_requested': False, 'behavior_acceptance': False}
            completed = 0
            terminated = False
            reason = None
            terminal_terms = {}
            with (target / 'joints.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                for stage in protocol['stages']:
                    count = round(stage['duration_s'] / dt)
                    if abs(count * dt - stage['duration_s']) > 1e-7:
                        raise ValueError('Protocol does not align to native control clock')
                    motion.request(bool(stage['go']))
                    for _ in range(count):
                        # Manager buffers created during stepping must remain
                        # normal tensors for later explicit protocol resets.
                        with torch.no_grad():
                            action = actor(obs)
                            if not torch.isfinite(action).all():
                                raise RuntimeError('Nonfinite native action')
                            squared_velocity = (motion.body_lin_vel_w - motion.robot_body_lin_vel_w).square().sum(-1).mean()
                            reward_std = base.cfg.rewards.motion_body_lin_vel.params['std']
                            row = [completed, completed * dt, stage['name'], int(stage['go']),
                                   float(motion.clock.phase[0]), float(motion.clock.rate[0]),
                                   int(motion.clock.exhausted[0]), 0, *flat(robot.data.root_pos_w[0]),
                                   *flat(robot.data.root_lin_vel_w[0]), tilt(robot.data.root_quat_w[0]),
                                   *flat(motion.body_lin_vel_w[0, pelvis_index])]
                            for body, contact in zip(foot_ids, sensor_ids):
                                row += [*flat(robot.data.body_pos_w[0, body]), *flat(robot.data.body_lin_vel_w[0, body]),
                                        *flat(sensor.data.net_forces_w[0, contact])]
                            row += [*flat(robot.data.joint_pos[0]), *flat(robot.data.joint_vel[0]),
                                    *flat(motion.joint_pos[0]), *flat(motion.joint_vel[0]), *flat(action[0])]
                            row += [float((motion.anchor_pos_w - motion.robot_anchor_pos_w).norm()),
                                    float(squared_velocity.sqrt()), float(torch.exp(-squared_velocity / reward_std ** 2))]
                            observations, _, done, truncated, extras = raw.step(action)
                            obs = observations['policy']
                            terminated = bool(done[0] or truncated[0])
                            if terminated:
                                reason = 'native_environment_termination'
                                terminal_terms = {k: float(v) for k, v in extras.get('log', {}).items()
                                                  if k.startswith('Episode_Termination/')}
                            elif float(robot.data.root_pos_w[0, 2]) < protocol['abort_pelvis_height_m']:
                                terminated = True
                                reason = 'pelvis_height_abort'
                            elif tilt(robot.data.root_quat_w[0]) > protocol['abort_tilt_rad']:
                                terminated = True
                                reason = 'pelvis_tilt_abort'
                            row[7] = int(terminated)
                            writer.writerow(row)
                            completed += 1
                        if terminated:
                            break
                    stream.flush()
                    if terminated:
                        break
            manifest.update(completed_steps=completed, completed_protocol=completed == planned and not terminated,
                            terminated=terminated, termination_reason=reason, terminal_terms=terminal_terms)
            (target / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            metrics = score(target, gates)
            (target / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
            entry = {'case': case_name, 'completed': manifest['completed_protocol'],
                     'termination_reason': reason, 'terminal_terms': terminal_terms,
                     'quantitative_pass': metrics['quantitative_pass'], 'stage_metrics': metrics['stage_metrics'],
                     'stage_gates': metrics['quantitative_gate_checks']}
            results.append(entry)
            print('[OMA_NATIVE_CASE] ' + json.dumps(entry), flush=True)
    finally:
        raw.close()
    unchanged = all(torch.equal(initial_actor[k], v.detach().cpu()) for k, v in actor.state_dict().items())
    if not unchanged or sha(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError('Fixed actor/source changed during diagnostic evaluation')
    summary = {'scope': 'Fixed-weight native dynamics crosscheck; no PPO updates or behavior acceptance',
               'source_checkpoint_sha256': args.checkpoint_sha256, 'source_saved_iteration': checkpoint['iter'],
               'optimizer_updates': 0, 'actor_and_normalizer_unchanged': unchanged, 'cases': results}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    files = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ['.csv', '.json']:
            content = path.read_bytes()
            files[path.relative_to(out).as_posix()] = {'sha256': hashlib.sha256(content).hexdigest(),
                                                     'bytes': torch.tensor(list(content), dtype=torch.uint8)}
    bundle = Path(args.bundle_file).resolve()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': 'beyondmimic_native_diagnostic_v1', 'summary': summary, 'files': files}, bundle)
    print('[OMA] Native diagnostic CSV bundle saved to established SDK PT directory: ' + str(bundle), flush=True)
    print('[OMA] Native diagnostic finished; no new trained weights', flush=True)
    time.sleep(120)  # Existing G1 upload window; no runtime/cost estimate.


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
