"""Pure-torch terms for the late-swing lateral direction reward."""

import torch


def swing_direction_step(swing_mask, toward_cmd_vy, air_time, lateral_gate,
                         v_ref, ramp_s):
    """Signed late-swing lateral foot velocity reward, one term per swing leg.

    swing_mask: (N, 2) float 1 while the foot is airborne (contact <= liftoff).
    toward_cmd_vy: (N, 2) foot lateral velocity in the base frame multiplied by
        sign(cmd_y); positive means the swing foot is moving toward the command.
    air_time: (N, 2) accumulated swing time BEFORE this step.
    lateral_gate: (N,) bool, true when the command is lateral (|vx| small, |vy|
        above the stand threshold).
    Returns the per-environment raw term (N,). The framework multiplies every
    reward term by its scale and ONE control dt, so this function must not
    apply dt itself; the swing air-time accumulator below still does.
    """
    ramp = (air_time / ramp_s).clamp(min=0.0, max=1.0) ** 2
    progress = (toward_cmd_vy / v_ref).clamp(min=0.0, max=1.0)
    terms = swing_mask * ramp * progress
    gate = lateral_gate.to(terms.dtype).unsqueeze(1)
    return (terms * gate).sum(dim=1)


def update_air_time(air_time, swing_mask, dt):
    """Accumulate swing time while airborne; reset to zero on contact."""
    return (air_time + dt) * swing_mask