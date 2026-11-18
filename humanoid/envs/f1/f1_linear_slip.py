"""Use horizontal foot translation for the existing contact-slip penalty."""
import torch
from .f1_random_order import F1RandomOrderCfg, F1RandomOrderEnv


class F1LinearSlipCfg(F1RandomOrderCfg):
    class env(F1RandomOrderCfg.env):
        reward_semantics = 'forward_progress_contact_linear_slip_v1'


class F1LinearSlipEnv(F1RandomOrderEnv):
    def _reward_foot_slip(self):
        # Retain the original40N contact gate, norm, sum and reward weight.
        # Isaac Gym rigid-body state: position0:3, quaternion3:7,
        # linear velocity7:10, angular velocity10:13.
        contact = self.contact_forces[:, self.feet_indices, 2] > 40.
        speed = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        return torch.sum(speed * contact, dim=1)
