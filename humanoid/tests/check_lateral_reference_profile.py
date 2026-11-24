"""Offline checks for the lateral hip-roll reference profile (no simulator required).

Run: python humanoid/tests/check_lateral_reference_profile.py

Covers three things the training configuration depends on:
  1. direction and amplitude of the phase-locked profile,
  2. the pure-lateral gate (sagittal / rotate / stand / omnidirectional unchanged),
  3. the full reference trajectory on the training URDF: leading-foot placement,
     double-support ankle separation and reference continuity.

The profile module is loaded by path so the check does not import the isaacgym
environment package.
"""

import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'humanoid' / 'envs' / 'f1' / 'f1_lateral_reference_profile.py'
URDF = ROOT / 'resources' / 'robots' / 'f1' / 'urdf' / 'f1_12dof_owner_f0_d005.urdf'
CONTRACT = ROOT / 'configs' / 'robot_contract.json'

_spec = importlib.util.spec_from_file_location('f1_lateral_reference_profile', PROFILE)
profile = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile)

AMPLITUDE = 0.126
COMMAND_SCALE = 0.4
STAND_THRESHOLD = 0.05  # f1_config.py commands.stand_com_threshold
FOOT_MIN_DIST = 0.2     # f1_config.py rewards.foot_min_dist
# Steady touchdown phases measured in exp-20261123-002 cold-start replays.
TOUCHDOWN_PHASE = {'left': 0.07, 'right': 0.57}
SHORT = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']


def offset_at(phase, command_x, command_y):
    return float(profile.lateral_hip_roll_offset(
        torch.tensor([phase]), torch.tensor([command_x]), torch.tensor([command_y]),
        AMPLITUDE, COMMAND_SCALE, STAND_THRESHOLD))


def load_kinematics():
    root = ET.parse(URDF).getroot()
    joints = {}
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        axis = joint.find('axis')
        joints[joint.get('name')] = {
            'xyz': np.array([float(x) for x in (origin.get('xyz') or '0 0 0').split()]),
            'rpy': [float(x) for x in (origin.get('rpy') or '0 0 0').split()],
            'axis': np.array([float(x) for x in (axis.get('xyz') or '0 0 0').split()])
            if axis is not None else None,
        }
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    defaults = contract['initial_pose']['default_joint_angles']
    deltas = contract['swing_joint_delta']
    sole_offset = contract['initial_pose']['feet_to_ankle_distance']
    return joints, defaults, deltas, sole_offset


def _rot_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _rot_axis(axis, angle):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def foot_point(joints, side, angles, sole_offset):
    transform = np.eye(4)
    for short in SHORT:
        joint = joints[f'{side}_{short}_joint']
        origin = np.eye(4)
        origin[:3, :3] = _rot_rpy(joint['rpy'])
        origin[:3, 3] = joint['xyz']
        transform = transform @ origin
        rotation = np.eye(4)
        rotation[:3, :3] = _rot_axis(joint['axis'], angles[f'{side}_{short}_joint'])
        transform = transform @ rotation
    return transform[:3, 3] + transform[:3, :3] @ np.array([0.0, 0.0, -sole_offset])


def reference_angles(defaults, deltas, phase, amplitude, lateral_command):
    sin_pos = math.sin(2 * math.pi * phase)
    cos_pos = math.cos(2 * math.pi * phase)
    blank = abs(sin_pos) < 0.1
    left = {f'left_{short}_joint': defaults[f'left_{short}_joint'] for short in SHORT}
    right = {f'right_{short}_joint': defaults[f'right_{short}_joint'] for short in SHORT}
    sin_l = min(sin_pos, 0.0)
    sin_r = max(sin_pos, 0.0)
    if not blank:
        for index, short in enumerate(SHORT):
            left[f'left_{short}_joint'] += -sin_l * deltas[index]
            right[f'right_{short}_joint'] += sin_r * deltas[6 + index]
    hip = amplitude * lateral_command * cos_pos
    left['left_hip_roll_joint'] += hip
    left['left_ankle_roll_joint'] -= hip
    right['right_hip_roll_joint'] -= hip
    right['right_ankle_roll_joint'] += hip
    return left, right


def check_profile():
    left_td = offset_at(TOUCHDOWN_PHASE['left'], 0.0, COMMAND_SCALE)
    right_td = offset_at(TOUCHDOWN_PHASE['right'], 0.0, COMMAND_SCALE)
    toe_off = offset_at(0.5, 0.0, COMMAND_SCALE)
    stance_sweep = left_td - toe_off
    foot_travel = stance_sweep * (0.1329 / 0.24)
    print(f'profile: left touchdown {left_td:+.4f} rad, right touchdown {-right_td:+.4f} rad')
    print(f'profile: stance sweep {stance_sweep:.4f} rad, foot travel {foot_travel:.4f} m/step')
    assert left_td > 0.11 and -right_td > 0.11
    assert math.isclose(left_td, -right_td, rel_tol=1e-6)
    assert 0.9 * 0.4 * 0.35 < foot_travel < 1.1 * 0.4 * 0.35

    # Pure-lateral gate: only zero vx with a real vy may change the reference.
    assert offset_at(0.07, 0.0, COMMAND_SCALE) != 0.0
    assert offset_at(0.07, 0.0, -COMMAND_SCALE) == -left_td
    for command_x, command_y, label in (
            (0.4, 0.4, 'omnidirectional'), (0.4, 0.0, 'sagittal'),
            (0.0, 0.0, 'stand'), (0.0, 0.04, 'near-zero lateral'),
            (0.0, STAND_THRESHOLD, 'exactly at threshold'), (0.06, 0.4, 'mixed')):
        assert offset_at(0.07, command_x, command_y) == 0.0, f'{label} must not move'
    small = offset_at(0.07, 0.0, 0.06)
    assert 0.0 < small < left_td, 'small lateral command keeps a proportionally small offset'
    print('profile: pure-lateral gate holds; omnidirectional/near-zero references unchanged')

    # Continuity of the added term across the cycle (no touchdown jump of its own).
    phases = torch.arange(0.0, 1.0, 0.001)
    command_y = torch.full_like(phases, COMMAND_SCALE)
    command_x = torch.zeros_like(phases)
    values = profile.lateral_hip_roll_offset(phases, command_x, command_y,
                                             AMPLITUDE, COMMAND_SCALE, STAND_THRESHOLD)
    jump = float((values - torch.roll(values, 1)).abs().max())
    print(f'profile: max per-0.001-phase change {jump:.5f} rad')
    assert jump < 0.01, 'added term must be continuous'


def check_geometry():
    joints, defaults, deltas, sole_offset = load_kinematics()
    default_left = foot_point(joints, 'left', defaults, sole_offset)[1]
    default_right = foot_point(joints, 'right', defaults, sole_offset)[1]

    left_touch, _ = reference_angles(defaults, deltas, TOUCHDOWN_PHASE['left'], AMPLITUDE, 1.0)
    _, right_touch = reference_angles(defaults, deltas, TOUCHDOWN_PHASE['right'], AMPLITUDE, 1.0)
    left_y = foot_point(joints, 'left', left_touch, sole_offset)[1]
    right_y = foot_point(joints, 'right', right_touch, sole_offset)[1]
    print(f'geometry: leading left foot {left_y - default_left:+.4f} m, '
          f'trailing right foot {right_y - default_right:+.4f} m (left command)')
    assert left_y - default_left > 0.05, 'leading foot must land on the commanded side'
    assert right_y - default_right > 0.05, 'trailing foot must close toward the commanded side'

    minimum = math.inf
    minimum_phase = None
    for step in range(2000):
        phase = step / 2000
        left, right = reference_angles(defaults, deltas, phase, AMPLITUDE, 1.0)
        separation = abs(foot_point(joints, 'left', left, sole_offset)[1]
                         - foot_point(joints, 'right', right, sole_offset)[1])
        if separation < minimum:
            minimum, minimum_phase = separation, phase
    print(f'geometry: minimum ankle separation {minimum:.4f} m at phase {minimum_phase:.3f} '
          f'(reward foot_min_dist {FOOT_MIN_DIST})')
    assert minimum > 0.17, 'double support must keep the feet clear of each other'
    if minimum < FOOT_MIN_DIST:
        print(f'note: reference dips below the {FOOT_MIN_DIST} m distance reward at double support; '
              'soft term, reference weight is an order above it')


def main():
    check_profile()
    check_geometry()
    print('all lateral reference checks passed')


if __name__ == '__main__':
    main()