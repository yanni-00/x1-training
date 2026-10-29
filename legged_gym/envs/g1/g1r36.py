"""G1 arm-rest mapping with sampled collision clearance; no action override."""
import torch
from legged_gym.envs.g1.g1r32 import G1R32Robot


class G1R36Robot(G1R32Robot):
    def _create_envs(self):
        super()._create_envs()
        self.arm_rest_shoulder_roll_inds = [self.dof_names.index(side+'_shoulder_roll_joint')
                                           for side in ('left','right')]
        magnitude = self.cfg.rewards.arm_rest_shoulder_roll_rad
        self.arm_rest_shoulder_roll_reference = torch.tensor(
            [magnitude,-magnitude],device=self.device,dtype=torch.float)

    def _arm_rest_error(self):
        # [OMA] design-hugwbc-g1-r36: change only the existing pose reference.
        # The inherited walking/standing components share this same geometry.
        # Observation/action offsets, initialization, collisions and termination
        # remain untouched; the policy must actually learn the reference.
        error = super()._arm_rest_error()
        error[:,self.arm_rest_shoulder_roll_inds] -= self.arm_rest_shoulder_roll_reference
        return error
