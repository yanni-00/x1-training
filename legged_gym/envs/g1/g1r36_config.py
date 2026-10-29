"""R36: collision-aware G1 shoulder-roll rest reference, matched to R35."""
from legged_gym.envs.g1.g1r35_config import G1R35Cfg,G1R35CfgPPO


class G1R36Cfg(G1R35Cfg):
    class rewards(G1R35Cfg.rewards):
        # Source: R35 arm-collision-reference.json, actual G1 collision asset.
        # Minimum sampled clear roll.25 + one tested.01 grid buffer, not a
        # human-norm/optimal-value estimate or dynamic safety guarantee.
        arm_rest_shoulder_roll_rad = 0.26


class G1R36CfgPPO(G1R35CfgPPO):
    class runner(G1R35CfgPPO.runner):
        experiment_name = 'g1_r36_collision_clear_arm_reference'
