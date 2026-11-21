"""Shape sustained rearward pitch during positive sagittal walking."""

import torch

from .f1_target_speed_ramp import F1TargetSpeedRampCfg, F1TargetSpeedRampEnv


class F1RearwardPitchCfg(F1TargetSpeedRampCfg):
    class env(F1TargetSpeedRampCfg.env):
        reward_semantics = 'target_speed_ramp_with_sagittal_rearward_pitch_v1'


class F1RearwardPitchEnv(F1TargetSpeedRampEnv):
    def _reward_orientation(self):
        original = super()._reward_orientation()
        threshold = self.cfg.commands.stand_com_threshold
        forward = (self.commands[:, 0] > threshold) & (self.commands[:, 1].abs() <= threshold)
        # With gravity along world -z, its body x component is sin(base pitch).
        rearward = (-self.projected_gravity[:, 0]).clamp_min(0.0)
        return original - torch.where(forward, rearward, torch.zeros_like(rearward))
