"""R30: one internal pitch-tracking weight probe with the original R29 input."""
from legged_gym.envs.g1.g1r29_config import G1R29Cfg,G1R29CfgPPO


class G1R30Cfg(G1R29Cfg):
    class rewards(G1R29Cfg.rewards):
        # Explicit sensitivity design after actual tracking/FK audit; not an
        # estimate of the required coefficient. No new reward component.
        arm_swing_pitch_weight = 10 * G1R29Cfg.rewards.arm_swing_pitch_weight


class G1R30CfgPPO(G1R29CfgPPO):
    class runner(G1R29CfgPPO.runner):
        experiment_name = 'g1_r30_arm_tracking_priority'
