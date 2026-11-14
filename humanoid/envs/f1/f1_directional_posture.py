"""One stage-one reward delta: permit explicitly commanded roll/yaw motion."""
import torch
from .f1_command_coverage import F1CommandCoverageCfg, F1CommandCoverageEnv


class F1DirectionalPostureCfg(F1CommandCoverageCfg):
    class env(F1CommandCoverageCfg.env):
        command_semantics = 'stage1_directional_posture_v1'


class F1DirectionalPostureEnv(F1CommandCoverageEnv):
    def _reward_default_joint_pos(self):
        diff = self.dof_pos - self.default_joint_pd_target
        # Existing native stand threshold is reused as an explicit command
        # deadband, not as a measured hardware quantity.
        threshold = self.cfg.commands.stand_com_threshold
        lateral = self.commands[:, 1].abs() > threshold
        turning = self.commands[:, 2].abs() > threshold
        posture = diff[:, [1, 2, 5, 7, 8, 11]]
        free = torch.stack((lateral, turning, lateral,
                            lateral, turning, lateral), dim=1)
        constrained = torch.where(free, torch.zeros_like(posture), posture)
        yaw_roll = (torch.norm(constrained[:, :3], dim=1)
                    + torch.norm(constrained[:, 3:], dim=1))
        # All constants and the residual all-joint penalty are unchanged from
        # F1WalkingEnv._reward_default_joint_pos; limits/termination also stay.
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(diff, dim=1)
