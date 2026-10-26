"""HugWBC G1 upright-walk stage with robot-specific reward joint groups."""

import torch

from legged_gym.envs.g1.g1r20 import G1R20Robot


class G1R21Robot(G1R20Robot):
    """Repair H1 joint-index assumptions without adding reward components.

    H1 has ten leg joints, one torso joint and eight arm joints. G1 has twelve
    leg joints, three waist joints and eight arm joints. Its command index 9
    retains the legacy configuration name ``waist_roll``, but the H1 URDF axis
    and the HugWBC paper identify that command as waist yaw.
    """

    def _create_envs(self):
        super()._create_envs()
        names = {name: i for i, name in enumerate(self.dof_names)}
        arm_names = [name for name in self.dof_names
                     if 'shoulder_' in name or 'elbow' in name]
        arm_set = set(arm_names)
        self.reward_arm_inds = [names[name] for name in arm_names]
        self.reward_lower_inds = [i for i, name in enumerate(self.dof_names)
                                  if name not in arm_set]
        self.reward_knee_inds = [names['left_knee_joint'], names['right_knee_joint']]
        self.waist_yaw_inds = [names['waist_yaw_joint']]
        if len(self.reward_arm_inds) != self.cfg.disturb.disturb_dim:
            raise RuntimeError('G1 reward arm group must match the intervention group')
        print('[g1r21] reward arms:', self.reward_arm_inds,
              'lower:', self.reward_lower_inds, 'knees:', self.reward_knee_inds,
              'waist command:', self.waist_yaw_inds)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Upright walking is the current stage; standing samples remain standing.
        self.commands[env_ids, self.command_gait_phase_dim] = 0.5

    def _reward_waist_control(self):
        error = self.dof_pos[:, self.waist_yaw_inds] - self.commands[:, 9:10]
        return torch.square(error).sum(dim=-1)

    def _action_rate_for_group(self, indices):
        first = self.actions[:, indices] - self.last_actions[:, indices]
        second = (self.actions[:, indices] - 2 * self.last_actions[:, indices]
                  + self.last_last_actions[:, indices])
        return torch.square(first).sum(-1) + torch.square(second).sum(-1)

    def _reward_action_rate_upper(self):
        return self._action_rate_for_group(self.reward_arm_inds) * (~self.interrupt_mask)

    def _reward_action_rate_lower(self):
        return self._action_rate_for_group(self.reward_lower_inds)

    def _reward_standing_joint_deviation(self):
        delta = self.dof_pos[:, self.reward_arm_inds] - self.default_dof_pos[:, self.reward_arm_inds]
        return (torch.square(delta).sum(-1) * self.standing_envs_mask
                * (~self.interrupt_mask))

    def _reward_joint_power_distribution(self):
        power = (torch.abs(self.torques) * torch.abs(self.dof_vel))[:, self.reward_knee_inds]
        return (torch.square(power.var(dim=-1)) * 1.e-8).clip(max=1)

    def _reward_dof_pos_limits(self):
        # H1InterruptRobot excludes arms unconditionally; preserve that policy.
        limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return limits[:, self.reward_lower_inds].sum(dim=1)

    def _reward_dof_acc(self):
        acceleration = torch.square((self.last_dof_vel - self.dof_vel) / self.dt)
        return acceleration[:, self.reward_lower_inds].sum(dim=1)

    def _reward_dof_vel_limits(self):
        limits = torch.clip(10 * self.velocity_level.unsqueeze(-1), min=10, max=20)
        error = (torch.abs(self.dof_vel[:, self.reward_arm_inds]) - limits).clip(min=0., max=15.).sum(dim=1)
        return 1 - torch.exp(-error)
