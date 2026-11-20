"""Restore forward-speed discrimination inside the existing low-speed interval."""
import torch
from .f1_sagittal_target import F1SagittalTargetCfg, F1SagittalTargetEnv


class F1LowSpeedRampCfg(F1SagittalTargetCfg):
    class env(F1SagittalTargetCfg.env):
        reward_semantics = 'longitudinal_progress_with_low_speed_ramp_v1'


class F1LowSpeedRampEnv(F1SagittalTargetEnv):
    def _reward_low_speed(self):
        previous = super()._reward_low_speed()
        threshold = self.cfg.commands.stand_com_threshold
        command_vx = self.commands[:, 0]
        body_vx = self.base_lin_vel[:, 0]
        # F1WalkingEnv: zero speed gives -2, and half the command gives +1.2.
        lower_edge = 0.5 * command_vx
        active = ((command_vx > threshold) & (self.commands[:, 1].abs() <= threshold)
                  & (body_vx > 0) & (body_vx < lower_edge))
        fraction = body_vx / (0.5 * command_vx.clamp_min(threshold))
        ramp = torch.lerp(torch.full_like(body_vx, -2.), torch.full_like(body_vx, 1.2), fraction)
        return torch.where(active, ramp, previous)
