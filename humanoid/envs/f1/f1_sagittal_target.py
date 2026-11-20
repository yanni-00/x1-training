"""Train the positive sagittal mode at the declared forward target."""
import torch
from .f1_config import F1LowLiftCfg
from .f1_longitudinal_progress import F1LongitudinalProgressCfg, F1LongitudinalProgressEnv


class F1SagittalTargetCfg(F1LongitudinalProgressCfg):
    class env(F1LongitudinalProgressCfg.env):
        command_semantics = 'stage1_positive_sagittal_target_v1'

    class commands(F1LongitudinalProgressCfg.commands):
        # Matches the frozen forward evaluation command and the Q3 target range endpoint.
        positive_sagittal_target = F1LowLiftCfg.commands.ranges.lin_vel_x[1]


class F1SagittalTargetEnv(F1LongitudinalProgressEnv):
    def _resample_walk_sagittal_command(self, env_ids):
        super()._resample_walk_sagittal_command(env_ids)
        vx = self.commands[env_ids, 0]
        forward = vx > self.cfg.commands.stand_com_threshold
        self.commands[env_ids, 0] = torch.where(
            forward, self.cfg.commands.positive_sagittal_target, vx)
