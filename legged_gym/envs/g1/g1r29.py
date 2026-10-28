"""G1 alternating arm reference inside the existing posture-cost component."""
import math
import torch
from legged_gym.envs.g1.g1r23 import G1R23Robot


class G1R29Robot(G1R23Robot):
    def _create_envs(self):
        super()._create_envs()
        self.swing_shoulder_pitch_inds = [self.dof_names.index(side + '_shoulder_pitch_joint')
                                          for side in ('left', 'right')]

    def _arm_rest_error(self):
        error = super()._arm_rest_error()
        # The opposite foot's clock is already part of the HugWBC observation.
        # The offset is calibrated from the accepted input's recorded foot-x
        # phase, rather than assuming that hip pitch is a foot-position proxy.
        phase = self.foot_indices[:, [1, 0]]
        reference = -self.cfg.rewards.arm_swing_pitch_amplitude * torch.cos(
            2 * math.pi * phase - self.cfg.rewards.arm_swing_phase_offset)
        # Preserve neutral arms for standing/zero forward command. Reverse the
        # reference for backward commands, and bound it by the chosen amplitude.
        speed_scale = (self.commands[:, 0] / self.cfg.rewards.arm_swing_speed_reference).clamp(-1., 1.)
        reference = reference * speed_scale.unsqueeze(1) * (~self.standing_envs_mask).unsqueeze(1)
        error[:, self.swing_shoulder_pitch_inds] -= reference
        return error

    def _reward_shoulder_deviation(self):
        error = self._arm_rest_error()
        # Existing component and external scale are retained. Defaults preserve
        # R29/R30 exactly; R31 isolates the residual elbow-coordinate priority.
        cost = (2 * torch.square(error[:, self.shoulder_inds]).sum(-1)
                + self.cfg.rewards.arm_swing_pitch_weight * torch.square(
                    error[:, self.swing_shoulder_pitch_inds]).sum(-1)
                + getattr(self.cfg.rewards, 'arm_rest_elbow_weight', .25)
                * torch.square(error[:, self.arm_rest_elbow_inds]).sum(-1))
        return cost * (~self.interrupt_mask)
