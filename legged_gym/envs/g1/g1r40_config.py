"""R40: one finite waist-posture tolerance probe, matched to R39."""
from legged_gym.envs.g1.g1r39_config import G1R39Cfg, G1R39CfgPPO


class G1R40Cfg(G1R39Cfg):
    class rewards(G1R39Cfg.rewards):
        # [OMA] design-hugwbc-g1-r40: half the existing8deg deadband.
        # Explicit single design point, not an optimal tolerance estimate.
        # Existing excess-square formula/scales and all arm mapping stay intact.
        waist_posture_tolerance_rad = G1R39Cfg.rewards.waist_posture_tolerance_rad / 2


class G1R40CfgPPO(G1R39CfgPPO):
    class runner(G1R39CfgPPO.runner):
        experiment_name = 'g1_r40_waist_tolerance_band'
