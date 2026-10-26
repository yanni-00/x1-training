"""G1 R20 robot hooks for the waist-upright recovery trial."""

import torch

from legged_gym.envs.g1.g1interrupt import G1InterruptRobot


class G1R20Robot(G1InterruptRobot):
    """G1 interrupt robot with a thresholded waist posture cost.

    The R19 torso_deviation term charged every nonzero waist angle. R20 instead
    leaves a tolerance band for gait and penalizes only waist roll/pitch outside
    the upright band registered for this experiment.
    """

    def _create_envs(self):
        super()._create_envs()
        dof_index = {name.lower(): i for i, name in enumerate(self.dof_names)}
        required = ("waist_roll_joint", "waist_pitch_joint")
        missing = [name for name in required if name not in dof_index]
        if missing:
            raise RuntimeError(f"R20 requires G1 waist joints in the URDF: {missing}")
        self.waist_posture_inds = [dof_index[name] for name in required]
        print("[g1r20] waist posture joints =", [self.dof_names[i] for i in self.waist_posture_inds])

    def _reward_waist_posture_excess(self):
        angles = self.dof_pos[:, self.waist_posture_inds] - self.default_dof_pos[:, self.waist_posture_inds]
        tolerance = self.cfg.rewards.waist_posture_tolerance_rad
        excess = (torch.abs(angles) - tolerance).clamp(min=0.0)
        return torch.sum(torch.square(excess), dim=-1)
