"""Phase-locked lateral hip-roll reference profile (pure torch, no simulator imports).

The inherited stage-one reference only moves the swing leg through a mid-swing arc
that returns to the default at touchdown, so its hip-roll offset is zero at every
steady touchdown and raising it only widens the mid-swing sway.  This profile adds
a lateral term whose peak sits on the touchdown phase instead: the swing foot
reaches the commanded side at touchdown and sweeps back across stance.

    offset(phase) = amplitude * clamp(cmd_y / command_scale, -1, 1) * cos(2*pi*phase)

Both F1 hip-roll axes are +x, so the same offset moves both feet toward +y.
Positive hip roll is abduction on the left leg and adduction on the right leg,
which is why the right leg uses the negated value one half-cycle later.

The term is gated to pure lateral commands with the same condition as the TASK054
lateral progress reward, so sagittal, rotate, stand and omnidirectional commands
keep their original reference exactly.

The reference is a body-relative joint target, not a world foot trajectory.
"""

import torch

# Audited dof order: left [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
# ankle_roll], then the right side at +6.
LEFT_HIP_ROLL = 1
LEFT_ANKLE_ROLL = 5
RIGHT_HIP_ROLL = 7
RIGHT_ANKLE_ROLL = 11


def lateral_hip_roll_offset(phase, command_x, command_y, amplitude, command_scale,
                            threshold):
    """Return the body-relative hip-roll offset for pure lateral commands only."""
    if command_scale <= 0.0:
        raise ValueError('lateral reference command_scale must be positive')
    if amplitude < 0.0:
        raise ValueError('lateral reference amplitude must be non-negative')
    if threshold < 0.0:
        raise ValueError('lateral reference threshold must be non-negative')
    pure_lateral = (command_x.abs() <= threshold) & (command_y.abs() > threshold)
    ratio = torch.clamp(command_y / command_scale, -1.0, 1.0)
    offset = amplitude * ratio * torch.cos(2.0 * torch.pi * phase)
    return torch.where(pure_lateral, offset, torch.zeros_like(offset))