"""Change the existing arm cost reference only; all actions remain learned."""
from legged_gym.envs.g1.g1r36 import G1R36Robot


class G1R39Robot(G1R36Robot):
    def _arm_rest_error(self):
        # [OMA] design-hugwbc-g1-r39: posterior upper arm with a bent elbow.
        # The inherited opposite-leg swing remains centered on this rest pose,
        # including the shared standing cost. No action/observation offsets.
        error = super()._arm_rest_error()
        error[:, self.swing_shoulder_pitch_inds] -= self.cfg.rewards.arm_rest_shoulder_pitch_rad
        return error
