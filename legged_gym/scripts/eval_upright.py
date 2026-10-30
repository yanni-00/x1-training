"""Audit G1 locomotion with every behavior command recorded and controlled.

This is an evaluation protocol, not a training reward change. ``sampled`` keeps
the sampled behavior commands for diagnosing the old velocity-only protocol;
``upright`` requests walking at nominal height, zero pitch and zero waist roll.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import isaacgym  # noqa: F401; must precede torch
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.isaacgym_utils import get_euler_xyz as euler_angles


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--protocol', choices=('sampled', 'height_only', 'upright', 'upright_reference'), default='upright')
    parser.add_argument('--eval-envs', type=int, default=16)  # smoke_r20.py
    parser.add_argument('--eval-steps', type=int, default=1000)  # play_diag.py
    parser.add_argument('--record-video', action='store_true')
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    output = Path(custom.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(custom.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if custom.record_video and (args.headless or custom.eval_envs != 1):
        raise ValueError('Video requires a viewer and --eval-envs=1')

    cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.seed is not None:
        train_cfg.seed = args.seed
        cfg.seed = args.seed
    cfg.env.num_envs = custom.eval_envs
    cfg.env.episode_length_s = 100000  # match old play; record actual terminations
    cfg.env.test = False
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = 100000
    cfg.noise.add_noise = False
    cfg.rewards.penalize_curriculum = False
    disabled_randomizations = []
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_') or name == 'push_robots':
            if isinstance(getattr(cfg.domain_rand, name), bool):
                setattr(cfg.domain_rand, name, False)
                disabled_randomizations.append(name)
    cfg.disturb.use_disturb = False
    cfg.terrain.curriculum = False
    cfg.terrain.mesh_type = 'trimesh'
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.max_init_terrain_level = 1
    cfg.terrain.selected = True
    cfg.terrain.selected_terrain_type = 'random_uniform'
    cfg.terrain.terrain_kwargs = {'random_uniform': {
        'min_height': 0.0, 'max_height': 0.0,
        'step': cfg.terrain.vertical_scale, 'downsampled_scale': 0.2}}

    base_class = task_registry.get_task_class(args.task)

    class EvaluationRobot(base_class):
        def _resample_commands(self, env_ids):
            super()._resample_commands(env_ids)
            if len(env_ids) == 0:
                return
            # Install commands before observations are built, including at reset.
            self.commands[env_ids, :3] = torch.tensor(
                [0.5, 0.0, 0.0], device=self.device)  # existing walk gate
            if custom.protocol == 'height_only':
                self.commands[env_ids, 7] = 0.0
            if custom.protocol in ('upright', 'upright_reference'):
                self.commands[env_ids, 3] = self.command_ranges['gait_frequency'][0]
                self.commands[env_ids, 4:6] = 0.5  # H1 sampler: walk, half-cycle stance
                self.commands[env_ids, 6] = self.command_ranges['foot_swing_height'][0]
                self.commands[env_ids, 7:10] = 0.0  # nominal height, level pelvis, neutral waist
                self.standing_envs_mask[env_ids] = False
            if hasattr(self, 'interrupt_mask'):
                self.interrupt_mask[env_ids] = False
                self.disturb_masks[env_ids] = False
                self.commands[env_ids, self.command_interrupt_flag_dim] = 0.0

    task_registry.task_classes[args.task] = EvaluationRobot
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    reference=env.reference_controller_metadata() if hasattr(env,'reference_controller_metadata') else None
    if (custom.protocol == 'upright_reference') != bool(reference and reference['active']):
        raise ValueError('Reference-arm controller requires the explicit hybrid evaluation protocol')
    train_cfg.runner.resume = True
    train_cfg.runner.resume_path = str(checkpoint)
    runner, _ = task_registry.make_alg_runner(
        env, args=args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    obs, critic_obs = env.reset()
    torso_index = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 'torso_link')
    if torso_index < 0:
        raise RuntimeError('G1 torso_link is required for world-frame posture measurements')
    joint_names = list(env.dof_names)
    columns = ['step', 'env', 'valid', 'terminated', 'time_s',
               'vx', 'vy', 'wz', 'base_z_world', 'base_height',
               'pelvis_roll', 'pelvis_pitch', 'pelvis_yaw',
               'torso_roll', 'torso_pitch', 'torso_yaw',
               'contact_left', 'contact_right', 'standing_mask']
    columns += ['cmd_' + str(i) for i in range(env.commands.shape[1])]
    columns += ['q_' + name for name in joint_names]
    # Additive diagnostics only; existing protocol/commands and columns retain
    # their meanings. Save enough state to recompute both clearance references.
    columns += ['foot_height_left', 'foot_height_right',
                'desired_contact_left', 'desired_contact_right',
                'foot_index_left', 'foot_index_right']
    rows = []
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    frame_dir = output / 'frames'
    if custom.record_video:
        frame_dir.mkdir(exist_ok=True)
    video_frames = 0
    for step in range(custom.eval_steps):
        if reference and (env.use_disturb or env.disturb_masks.any() or env.interrupt_mask.any()):
            raise RuntimeError('Hybrid evaluation cannot suppress contact termination or reward masks')
        commands_used = env.commands.clone()
        with torch.inference_mode():
            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
        obs, critic_obs, _, dones, _ = env.step(actions)
        terminations = dones.bool() & ~env.time_out_buf.bool()
        valid = ~failed & ~terminations
        relative_height = (env.root_states[:, 2:3] - env.heights_below_base).mean(-1)
        batch = torch.cat((
            torch.full((env.num_envs, 1), step, device=env.device),
            torch.arange(env.num_envs, device=env.device).unsqueeze(1),
            valid.unsqueeze(1), terminations.unsqueeze(1),
            torch.full((env.num_envs, 1), (step + 1) * env.dt, device=env.device),
            env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3],
            env.root_states[:, 2:3], relative_height.unsqueeze(1), env.rpy,
            euler_angles(env.rigid_body_states[:, torso_index, 3:7]),
            (env.contact_forces[:, env.feet_indices, 2] > 1.0),
            env.standing_envs_mask.unsqueeze(1), commands_used, env.dof_pos,
            env.foot_pos_world[:, :, 2].view(env.num_envs, -1) - env.measured_foot_scan.mean(-1),
            env.desired_contact_states, env.foot_indices,
        ), dim=1)
        rows.append(batch.cpu().numpy())
        failed |= terminations
        if custom.record_video and step % 2 == 0:  # play_video.py capture cadence
            angle = 2 * np.pi * step / custom.eval_steps
            target = env.root_states[0, :3].cpu().numpy().astype(float)
            camera = target + 2.0 * np.array([np.cos(angle), np.sin(angle), 0.3])
            env.gym.viewer_camera_look_at(env.viewer, env.envs[0],
                gymapi.Vec3(*camera), gymapi.Vec3(*target))
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / ('f_%04d.png' % video_frames)))
            video_frames += 1
        if failed.all():
            break

    data = np.concatenate(rows)
    csv_path = output / 'rollout.csv'
    np.savetxt(str(csv_path), data, delimiter=',', header=','.join(columns), comments='')
    summary = {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'task': args.task, 'protocol': custom.protocol, 'protocol_version': 3 if reference else 2, 'seed': cfg.seed,
        'dt': env.dt, 'requested_steps': custom.eval_steps, 'envs': env.num_envs,
        'base_height_target': cfg.rewards.base_height_target,
        'disabled_randomizations': disabled_randomizations,
        'interventions_enabled': bool(env.use_disturb or reference),
        'legacy_disturbances_enabled': bool(env.use_disturb),
        'reference_controller': reference,
        'nontermination_ratio': (~failed).float().mean().item(), 'per_env': []}
    col = {name: i for i, name in enumerate(columns)}
    steady = data[(data[:, col['step']] >= 200) & (data[:, col['valid']] == 1)]
    metrics = ['vx', 'vy', 'wz', 'base_height', 'pelvis_roll', 'pelvis_pitch',
               'torso_roll', 'torso_pitch', 'foot_height_left', 'foot_height_right']
    metrics += ['q_' + n for n in joint_names if any(
        part in n for part in ('waist_', 'knee', 'hip_pitch', 'ankle_pitch', 'shoulder_', 'elbow'))]
    for i in range(env.num_envs):
        segment = steady[steady[:, col['env']] == i]
        result = {'env': i, 'survived': not bool(failed[i]), 'steady_samples': len(segment)}
        for name in metrics:
            values = segment[:, col[name]]
            result[name] = None if len(values) == 0 else {
                'mean': float(values.mean()), 'mean_abs': float(np.abs(values).mean()),
                'min': float(values.min()), 'max': float(values.max()),
                'p95_abs': float(np.percentile(np.abs(values), 95))}
        result['commands_first'] = data[data[:, col['env']] == i][0,
            col['cmd_0']:col['cmd_0'] + env.commands.shape[1]].tolist()
        summary['per_env'].append(result)
    for name in metrics:
        values = steady[:, col[name]]
        summary[name] = None if len(values) == 0 else {
            'mean': float(values.mean()), 'mean_abs': float(np.abs(values).mean()),
            'p95_abs': float(np.percentile(np.abs(values), 95))}
    for side in ('left', 'right'):
        contact = steady[steady[:, col['contact_' + side]] == 1]
        knee = contact[:, col['q_' + side + '_knee_joint']]
        summary[side + '_stance_knee_mean'] = None if len(knee) == 0 else float(knee.mean())
        swing = steady[steady[:, col['desired_contact_' + side]] < 0.5]
        summary[side + '_swing_foot_origin_height_mean'] = None if len(swing) == 0 else float(
            swing[:, col['foot_height_' + side]].mean())
    urdf_path = ROOT / 'resources/robots/g1/g1_29dof_anneal_23dof.urdf'
    urdf = ET.parse(urdf_path).getroot()
    offsets = []
    for side in ('left', 'right'):
        for collision in urdf.find("link[@name='%s_ankle_roll_link']" % side).findall('collision'):
            z = float(collision.find('origin').get('xyz').split()[2])
            offsets.append(-z + float(collision.find('geometry/sphere').get('radius')))
    g1_reference = max(offsets)
    foot_height = steady[:, [col['foot_height_left'], col['foot_height_right']]]
    foot_index = steady[:, [col['foot_index_left'], col['foot_index_right']]]
    desired = steady[:, [col['desired_contact_left'], col['desired_contact_right']]]
    phases = 1 - np.abs(1 - np.clip(foot_index * 2 - 1, 0, 1) * 2)
    native_reference = getattr(cfg.rewards, 'feet_clearance_reference_height', 0.07)
    counterfactual = {}
    for name, reference in [('native', native_reference), ('legacy_h1', 0.07), ('g1_geometry', g1_reference)]:
        target = steady[:, col['cmd_6']:col['cmd_6']+1] * phases + reference
        cost = np.clip(np.sum(np.square(target-foot_height) * (1-desired), axis=1), 0, .1)
        counterfactual[name] = {'reference_height': reference, 'mean_raw_cost': float(cost.mean()),
                               'max_raw_cost': float(cost.max())}
    summary['clearance_diagnostics'] = {'source_urdf': str(urdf_path),
        'urdf_sha256': hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        'counterfactual': counterfactual,
        'scope': 'Same recorded steady trajectory; linear clearance formula, original contact mask and clip. Not a learning-outcome prediction. G1 reference is level collision geometry.'}
    if custom.record_video:
        video = output / 'play.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate',
            str(1.0 / (2 * env.dt)), '-i', str(frame_dir / 'f_%04d.png'),
            '-pix_fmt', 'yuv420p', '-crf', '23', str(video)], check=True)
        summary['video'] = str(video)
        summary['video_frames'] = video_frames
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    compact = {key: summary[key] for key in (
        'checkpoint_sha256', 'protocol', 'seed', 'nontermination_ratio',
        'base_height_target', 'vx', 'base_height', 'pelvis_pitch',
        'q_waist_pitch_joint', 'q_left_knee_joint', 'q_right_knee_joint')}
    print('UPRIGHT_SUMMARY ' + json.dumps(compact))
    env.gym.destroy_viewer(env.viewer) if env.viewer is not None else None
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
