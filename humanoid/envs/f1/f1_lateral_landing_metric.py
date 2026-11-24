"""Touchdown-settled lateral landing metric (pure torch, no simulator imports).

For every completed swing, score the swing's lateral displacement at the first
valid touchdown:

    advance = (touchdown_xy - liftoff_xy) . lateral_axis
    lateral_axis = R_z(yaw at liftoff) @ (0, 1)
    target  = vy_cmd at liftoff * cycle_time
    shaped  = (|target| - |advance - target|) / (lateral_command_max * cycle_time)

Both the heading axis and the speed target are frozen at liftoff, so a command
change during the swing cannot shrink the target. Using the liftoff position
removes stance slip from the measurement. The signed linear form is deliberately
unclamped: a backward landing deducts exactly what a forward landing earns, so a
foot that advances then returns nets zero regardless of step size.

Swing validity requires a clean airborne run: at least min_air_steps consecutive
airborne steps, at most max_gap_steps non-airborne steps immediately before the
touchdown (the force ramp), and no return to the air after a mid-level contact.
When a swing is cancelled (the pure-lateral gate or the commanded direction
changes, or a brush contact interrupts it) the foot must make firm contact again
before a new liftoff can start, so a cancelled swing cannot be restarted in the
air.

Touchdown-settled scoring is sparse; the reward scale is a declared design input
(see the experiment pre-registration), not a tuned value.
"""

import torch


def new_landing_state(num_envs, device):
    return {
        'armed': torch.zeros(num_envs, dtype=torch.bool, device=device),
        'baseline_xy': torch.zeros(num_envs, 2, 2, device=device),
        'in_swing': torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        'must_land': torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        'air_run': torch.zeros(num_envs, 2, dtype=torch.long, device=device),
        'gap_run': torch.zeros(num_envs, 2, dtype=torch.long, device=device),
        'yaw': torch.zeros(num_envs, 2, device=device),
        'target': torch.zeros(num_envs, 2, device=device),
        'prev_fz': torch.zeros(num_envs, 2, device=device),
        'gate_code': torch.zeros(num_envs, dtype=torch.long, device=device),
    }


def landing_step(state, foot_xy, fz, base_yaw, cmd_x, cmd_y, touchdown_n, liftoff_n,
                 min_air_steps, max_gap_steps, stand_com_threshold, cycle_time, command_scale):
    """Update the per-foot landing state and return the unscaled per-env reward."""
    num_envs = fz.shape[0]
    zero_long = torch.zeros_like(state['air_run'])

    pure_lateral = (cmd_x.abs() <= stand_com_threshold) & (cmd_y.abs() > stand_com_threshold)
    sign = torch.where(cmd_y >= 0, torch.ones_like(state['gate_code']),
                       -torch.ones_like(state['gate_code']))
    gate_code = torch.where(pure_lateral, sign, torch.zeros_like(state['gate_code']))

    unarmed = ~state['armed']
    if unarmed.any():
        m3 = unarmed.view(num_envs, 1, 1)
        m2 = unarmed.view(num_envs, 1)
        state['baseline_xy'] = torch.where(m3, foot_xy, state['baseline_xy'])
        state['prev_fz'] = torch.where(m2, fz, state['prev_fz'])
        state['in_swing'] = state['in_swing'] & ~m2
        state['must_land'] = state['must_land'] & ~m2
        state['air_run'] = torch.where(m2, zero_long, state['air_run'])
        state['gap_run'] = torch.where(m2, zero_long, state['gap_run'])
        state['yaw'] = torch.where(m2, base_yaw.view(num_envs, 1), state['yaw'])
        state['target'] = torch.where(m2, torch.zeros_like(state['target']), state['target'])
        state['gate_code'] = torch.where(unarmed, gate_code, state['gate_code'])
        state['armed'] = state['armed'] | unarmed

    was_swinging = state['in_swing']
    changed = (gate_code != state['gate_code']).view(num_envs, 1)
    state['gate_code'] = gate_code
    # A swing that straddles a command change is discarded; because the foot is
    # still airborne it must make firm contact before a new liftoff can start.
    must_land = state['must_land'] | (changed & was_swinging)
    in_swing = was_swinging & ~changed
    air_run = torch.where(changed, zero_long, state['air_run'])
    gap_run = torch.where(changed, zero_long, state['gap_run'])
    baseline = state['baseline_xy']
    yaw = state['yaw']
    target = state['target']

    airborne = fz < liftoff_n
    mid = (fz >= liftoff_n) & (fz < touchdown_n)
    touchdown = (fz >= touchdown_n) & (state['prev_fz'] < touchdown_n)
    firm_contact = fz >= touchdown_n

    liftoff = airborne & ~in_swing & ~must_land & ~changed
    in_swing = in_swing | liftoff
    air_run = torch.where(liftoff, torch.ones_like(air_run), air_run)
    gap_run = torch.where(liftoff, zero_long, gap_run)
    baseline = torch.where(liftoff.unsqueeze(-1), foot_xy, baseline)
    yaw = torch.where(liftoff, base_yaw.view(num_envs, 1), yaw)
    target = torch.where(liftoff, (cmd_y * cycle_time).view(num_envs, 1), target)

    # A return to the air after a mid-level contact was a brush, not the landing
    # ramp: the swing is discarded and must be followed by a real landing.
    invalid = airborne & in_swing & ~liftoff & (gap_run > 0)
    must_land = must_land | invalid
    in_swing = in_swing & ~invalid
    air_run = torch.where(invalid, zero_long, air_run)
    gap_run = torch.where(invalid, zero_long, gap_run)

    advancing = airborne & in_swing & ~liftoff & ~invalid
    air_run = torch.where(advancing, air_run + 1, air_run)
    gap_run = torch.where(advancing, zero_long, gap_run)
    gap_run = torch.where(mid & in_swing, gap_run + 1, gap_run)

    landing = touchdown & in_swing
    valid = landing & (air_run >= min_air_steps) & (gap_run <= max_gap_steps) & pure_lateral.view(num_envs, 1)

    axis = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    advance = ((foot_xy - baseline) * axis).sum(-1)
    # Distance-to-target shaping normalised by the full-command target, so a
    # large step under a small command is penalised instead of over-credited.
    shaped = (target.abs() - (advance - target).abs()) / (command_scale * cycle_time)
    reward = torch.where(valid, shaped, torch.zeros_like(shaped)).sum(-1)
    reward = torch.where(unarmed, torch.zeros_like(reward), reward)

    must_land = must_land & ~(touchdown | firm_contact)
    in_swing = in_swing & ~landing
    stale = firm_contact & in_swing
    in_swing = in_swing & ~stale
    air_run = torch.where(landing | stale, zero_long, air_run)
    gap_run = torch.where(landing | stale, zero_long, gap_run)

    state['in_swing'] = in_swing
    state['must_land'] = must_land
    state['air_run'] = air_run
    state['gap_run'] = gap_run
    state['baseline_xy'] = baseline
    state['yaw'] = yaw
    state['target'] = target
    state['prev_fz'] = fz
    return reward