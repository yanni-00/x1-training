"""Extend the existing command-relative progress reward to pure lateral motion."""

import torch

from .f1_target_speed_ramp import F1TargetSpeedRampCfg, F1TargetSpeedRampEnv


class F1LateralProgressCfg(F1TargetSpeedRampCfg):
    class env(F1TargetSpeedRampCfg.env):
        reward_semantics = 'target_speed_ramp_with_lateral_progress_v1'


class F1LateralProgressEnv(F1TargetSpeedRampEnv):
    def _reward_tracking_lin_vel(self):
        original = super()._reward_tracking_lin_vel()
        threshold = self.cfg.commands.stand_com_threshold
        target_y = self.commands[:, 1]
        lateral = (self.commands[:, 0].abs() <= threshold) & (target_y.abs() > threshold)
        relative_error = (target_y - self.base_lin_vel[:, 1]).abs() / target_y.abs().clamp_min(threshold)
        progress = self.cfg.rewards.forward_progress_bonus_scale * (1 - relative_error).clamp_min(0)
        return original + torch.where(lateral, progress, torch.zeros_like(progress))
