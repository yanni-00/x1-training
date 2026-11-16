"""One reward delta: continuous forward progress, with the old policy intact."""
import torch
from .f1_directional_posture import F1DirectionalPostureCfg, F1DirectionalPostureEnv


class F1SpeedTrackingCfg(F1DirectionalPostureCfg):
    class env(F1DirectionalPostureCfg.env):
        goal_revision = 'F1-WALK-v3'
        reward_semantics = 'forward_relative_progress_v1'

    class rewards(F1DirectionalPostureCfg.rewards):
        # Source: existing X1 linear_grad and the native track_vel_hard scale.
        forward_progress_bonus_scale = F1DirectionalPostureCfg.rewards.scales.track_vel_hard


class F1SpeedTrackingEnv(F1DirectionalPostureEnv):
    def _forward_progress_bonus(self):
        threshold = self.cfg.commands.stand_com_threshold
        forward = ((self.commands[:, 0] > threshold)
                   & (self.commands[:, 1].abs() <= threshold))
        error = torch.norm(self.commands[:, :2]-self.base_lin_vel[:, :2], dim=1)
        denominator = self.commands[:, 0].clamp(min=threshold)
        bonus = self.cfg.rewards.forward_progress_bonus_scale * (1-error/denominator).clamp(min=0)
        return torch.where(forward, bonus, torch.zeros_like(bonus))

    def _reward_tracking_lin_vel(self):
        return super()._reward_tracking_lin_vel()+self._forward_progress_bonus()
