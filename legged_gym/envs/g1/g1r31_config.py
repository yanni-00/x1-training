"""R31: isolate elbow-reference priority after the recorded R30 residual audit."""
from legged_gym.envs.g1.g1r30_config import G1R30Cfg, G1R30CfgPPO


class G1R31Cfg(G1R30Cfg):
    class rewards(G1R30Cfg.rewards):
        # Design probe, not an inferred required coefficient. Same existing
        # component, hanging reference, pitch priority and all body protection.
        arm_rest_elbow_weight = G1R30Cfg.rewards.arm_swing_pitch_weight


class G1R31CfgPPO(G1R30CfgPPO):
    class runner(G1R30CfgPPO.runner):
        experiment_name = 'g1_r31_elbow_tracking_priority'
