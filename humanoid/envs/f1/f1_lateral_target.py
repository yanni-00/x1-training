"""Expose the dedicated lateral mode to both commanded speed endpoints."""

import torch

from .f1_lateral_progress import F1LateralProgressCfg, F1LateralProgressEnv


class F1LateralTargetCfg(F1LateralProgressCfg):
    class env(F1LateralProgressCfg.env):
        command_semantics = 'stage1_sagittal_and_lateral_target_v1'


class F1LateralTargetEnv(F1LateralProgressEnv):
    def _resample_walk_lateral_command(self, env_ids):
        super()._resample_walk_lateral_command(env_ids)
        command = self.commands[env_ids, 1]
        lower, upper = self.command_ranges['lin_vel_y']
        endpoint = torch.where(command > 0, upper, lower)
        moving = command.abs() > self.cfg.commands.stand_com_threshold
        self.commands[env_ids, 1] = torch.where(moving, endpoint, command)
