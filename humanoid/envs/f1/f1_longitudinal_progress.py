"""Keep the contact-motion recipe and align the extra progress bonus with body vx."""
import torch
from .f1_contact_motion import F1ContactMotionCfg, F1ContactMotionEnv


class F1LongitudinalProgressCfg(F1ContactMotionCfg):
    class env(F1ContactMotionCfg.env):
        reward_semantics = 'longitudinal_progress_contact_linear_plus_angular_v1'


class F1LongitudinalProgressEnv(F1ContactMotionEnv):
    def _forward_progress_bonus(self):
        threshold = self.cfg.commands.stand_com_threshold
        target_vx = self.commands[:, 0]
        forward = (target_vx > threshold) & (self.commands[:, 1].abs() <= threshold)
        error = (target_vx - self.base_lin_vel[:, 0]).abs()
        relative_error = error / target_vx.clamp(min=threshold)
        bonus = self.cfg.rewards.forward_progress_bonus_scale * (1 - relative_error).clamp(min=0)
        return torch.where(forward, bonus, torch.zeros_like(bonus))
