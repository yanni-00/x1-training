"""Potential-based timing reshaping of the r7 lateral landing reward (pure torch).

The r7 touchdown settlement stays in the normal reward pipeline, including the
positive-only clip. This module adds, after that clip, the potential difference

    r_t = gamma * Phi(s_t) - Phi(s_{t-1}),   Phi = (|target| - |proj - target|) / (cmd_max * cycle)

where Phi is the r7 settlement score that would be earned if the foot landed at
the current position. Phi is zero whenever the foot is not in a live swing, so a
touchdown, a cancelled swing or a terminal step pays -Phi(s_{t-1}) and the
prepaid credit is clawed back. Over any complete swing the discounted sum is
exactly zero, so the discounted return of a trajectory equals the r7 clipped
return; only the timing of the feedback changes.

The swing state machine is the r7 one (landing_step); this module keeps a second
copy of that state for the potential and never touches the r7 settlement state.
"""

import importlib.util
from pathlib import Path

import torch

_MODULE = Path(__file__).resolve().parent / 'f1_lateral_landing_metric.py'
_spec = importlib.util.spec_from_file_location('f1_lateral_landing_metric_r7', _MODULE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
landing_step = _base.landing_step
new_landing_state = _base.new_landing_state


def new_timing_state(num_envs, device):
    state = new_landing_state(num_envs, device)
    state['potential'] = torch.zeros(num_envs, 2, device=device)
    return state


def timing_step(state, foot_xy, fz, base_yaw, cmd_x, cmd_y, terminal, gamma, weight, dt,
                touchdown_n, liftoff_n, min_air_steps, max_gap_steps, stand_com_threshold,
                cycle_time, command_scale):
    """Advance the timing state and return the unscaled-by-weight/DT post-clip term."""
    num_envs = fz.shape[0]
    unarmed = ~state['armed']
    landing_step(state, foot_xy, fz, base_yaw, cmd_x, cmd_y, touchdown_n, liftoff_n,
                 min_air_steps, max_gap_steps, stand_com_threshold, cycle_time, command_scale)
    terminal_mask = terminal.bool().view(num_envs, 1)
    if terminal_mask.any():
        state['in_swing'] = state['in_swing'] & ~terminal_mask

    axis = torch.stack((-torch.sin(state['yaw']), torch.cos(state['yaw'])), dim=-1)
    projected = ((foot_xy - state['baseline_xy']) * axis).sum(-1)
    distance = (projected - state['target']).abs()
    phi = torch.where(state['in_swing'],
                      (state['target'].abs() - distance) / (command_scale * cycle_time),
                      torch.zeros_like(distance))
    term = gamma * phi - state['potential']
    term = torch.where(unarmed.view(num_envs, 1), torch.zeros_like(term), term)
    state['potential'] = torch.where(unarmed.view(num_envs, 1), torch.zeros_like(phi), phi)
    return (term * weight * dt).sum(-1)
