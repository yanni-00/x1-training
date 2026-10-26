"""R23: use a G1-specific hanging-arm reference in existing posture costs."""
import torch

from legged_gym.envs.g1.g1r21 import G1R21Robot


class G1R23Robot(G1R21Robot):
    def _create_envs(self):
        super()._create_envs()
        self.arm_rest_elbow_inds = [self.dof_names.index(side + '_elbow_joint')
                                    for side in ('left', 'right')]

    def _arm_rest_error(self):
        # Keep action offsets, observations, initialization and checkpoint
        # semantics unchanged. Only the posture-cost reference is translated.
        error = self.dof_pos - self.default_dof_pos
        error[:, self.arm_rest_elbow_inds] -= self.cfg.rewards.arm_rest_elbow_rad
        return error

    def _reward_shoulder_deviation(self):
        error = self._arm_rest_error()
        # Retain H1's groups and weights: shoulder roll/yaw 2; pitch/elbow .25.
        cost = (2 * torch.square(error[:, self.shoulder_inds]).sum(-1)
                + 0.25 * torch.square(error[:, self.elbow_inds]).sum(-1))
        return cost * (~self.interrupt_mask)

    def _reward_standing_joint_deviation(self):
        # The standing cost must use the same rest pose as the walking cost.
        error = self._arm_rest_error()[:, self.reward_arm_inds]
        return (torch.square(error).sum(-1) * self.standing_envs_mask
                * (~self.interrupt_mask))
