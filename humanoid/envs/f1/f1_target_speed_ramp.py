"""Strengthen sagittal progress across the inherited low-speed reward plateau."""

import torch

from .f1_low_speed_ramp import F1LowSpeedRampCfg, F1LowSpeedRampEnv


class F1TargetSpeedRampCfg(F1LowSpeedRampCfg):
    class env(F1LowSpeedRampCfg.env):
        reward_semantics = 'longitudinal_progress_with_target_speed_ramp_v1'


class F1TargetSpeedRampEnv(F1LowSpeedRampEnv):
    def _reward_low_speed(self):
        previous = super()._reward_low_speed()
        command_x = self.commands[:, 0]
        threshold = self.cfg.commands.stand_com_threshold
        sagittal = (command_x > threshold) & (self.commands[:, 1].abs() <= threshold)
        ratio = self.base_lin_vel[:, 0] / command_x.clamp_min(threshold)

        # Inherit the half-command and overspeed limits from F1WalkingEnv.
        lower, upper = 0.5, 1.2
        toward_target = (ratio - lower) / (1.0 - lower)
        beyond_target = (upper - ratio) / (upper - 1.0)
        taper = torch.minimum(toward_target, beyond_target).clamp_min(0.0)
        # Use the inherited +1.2 plateau value for the peak increment.
        return previous + torch.where(sagittal, 1.2 * taper, torch.zeros_like(previous))
